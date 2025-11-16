#!/usr/bin/env python3
"""
Evaluate an Azure OpenAI chat completion deployment on a JSONL validation set.

The script:
  * Reads conversation-style examples (with the final assistant turn as the label).
  * Calls the specified deployment while requesting log probabilities.
  * Derives class predictions from the textual output only.
  * Uses token log probabilities separately to compute probability scores.
  * Logs per-example outputs and aggregated classification metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests


class ContentFilterError(RuntimeError):
    """Raised when the Azure content filter blocks a request."""


@dataclass
class Example:
    index: int
    prompt_messages: List[Dict[str, str]]
    label: str


@dataclass
class TruncationConfig:
    enabled: bool
    max_chars: int
    min_head: int
    min_tail: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run evaluation requests against an Azure OpenAI deployment."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to the validation JSONL file.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("logs/model_outputs.jsonl"),
        help="Destination for per-example JSONL logs.",
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=Path("logs/metrics.json"),
        help="Where to write the aggregate metrics JSON.",
    )
    parser.add_argument(
        "--positive-label",
        default=None,
        help="Label to treat as the positive class for metrics (matched case-insensitively).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold reported alongside the metrics (decisions come from text responses).",
    )
    parser.add_argument(
        "--rpm",
        type=float,
        default=10.0,
        help="Target requests per minute to respect while looping through the dataset.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Number of times to retry transient API failures.",
    )
    parser.add_argument(
        "--retry-wait",
        type=float,
        default=15.0,
        help="Base wait time (seconds) before retrying a failed request.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout (seconds) for each API call.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print a progress line after each request (ping-pong style feedback).",
    )
    parser.add_argument(
        "--truncate-prompts",
        action="store_true",
        help="Enable prompt truncation to reduce sequence length before sending requests.",
    )
    parser.add_argument(
        "--truncate-max-chars",
        type=int,
        default=1200,
        help="Maximum number of characters to keep in truncated prompts (split equally across head/tail).",
    )
    parser.add_argument(
        "--truncate-min-head",
        type=int,
        default=400,
        help="Minimum number of leading characters to preserve when truncating prompts.",
    )
    parser.add_argument(
        "--truncate-min-tail",
        type=int,
        default=400,
        help="Minimum number of trailing characters to preserve when truncating prompts.",
    )
    return parser.parse_args()


def load_examples(path: Path) -> Tuple[List[Example], Dict[str, str], List[str]]:
    examples: List[Example] = []
    label_lookup: Dict[str, str] = {}
    label_order: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            record = json.loads(line)
            messages = record.get("messages")
            if not messages or not isinstance(messages, list):
                raise ValueError(f"Example {idx} has no 'messages' array.")
            if messages[-1].get("role") != "assistant":
                raise ValueError(f"Example {idx} does not end with an assistant label.")
            label_raw = register_label(messages[-1].get("content", ""), label_lookup, label_order)
            prompt_messages = messages[:-1]
            examples.append(
                Example(index=idx, prompt_messages=prompt_messages, label=label_raw)
            )
    if len(label_lookup) < 2:
        raise ValueError("Expected at least two distinct labels in the dataset.")
    return examples, label_lookup, label_order


def register_label(text: str, label_lookup: Dict[str, str], label_order: List[str]) -> str:
    content = (text or "").strip()
    if not content:
        raise ValueError("Encountered empty assistant label in dataset.")
    first_token = content.split()[0]
    normalized = first_token.lower()
    canonical = label_lookup.get(normalized)
    if canonical is None:
        canonical = first_token
        label_lookup[normalized] = canonical
        label_order.append(canonical)
    return canonical


def apply_truncation(
    messages: Sequence[Dict[str, str]],
    config: TruncationConfig,
) -> Tuple[List[Dict[str, str]], bool]:
    if not config.enabled:
        return [dict(message) for message in messages], False
    truncated_any = False
    prepared: List[Dict[str, str]] = []
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        new_content, truncated = truncate_content(
            content, config
        ) if role != "system" else (content, False)
        if truncated:
            truncated_any = True
        prepared.append({"role": role, "content": new_content})
    return prepared, truncated_any


def truncate_content(content: str, config: TruncationConfig) -> Tuple[str, bool]:
    if not content or len(content) <= config.max_chars:
        return content, False
    max_chars = max(config.max_chars, config.min_head + config.min_tail)
    head = max(config.min_head, max_chars // 2)
    tail = max(config.min_tail, max_chars - head)
    if head + tail > len(content):
        head = len(content)
        tail = 0
    truncated_section = len(content) - head - tail
    head_text = content[:head]
    tail_text = content[len(content) - tail :] if tail else ""
    marker = f"...[TRUNCATED {truncated_section} chars]..."
    return f"{head_text}{marker}{tail_text}", True


def normalize_label(text: str, label_lookup: Dict[str, str], default_label: str) -> str:
    """Map the model text output to canonical labels drawn from the dataset."""
    content = (text or "").strip()
    if not content:
        return default_label
    tokens = content.split()
    if tokens:
        canonical = label_lookup.get(tokens[0].lower())
        if canonical:
            return canonical
    lowered = content.lower()
    for key, value in label_lookup.items():
        if key in lowered:
            return value
    return default_label


def build_request_url(endpoint: str, deployment: str, api_version: str) -> str:
    base = endpoint.rstrip("/")
    return f"{base}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"


def send_chat_completion(
    session: requests.Session,
    url: str,
    payload: Dict,
    api_key: str,
    timeout: float,
    max_retries: int,
    retry_wait: float,
) -> Dict:
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    for attempt in range(1, max_retries + 1):
        response = session.post(url, headers=headers, json=payload, timeout=timeout)
        if response.ok:
            return response.json()
        if response.status_code == 400:
            try:
                detail_json = response.json()
            except ValueError:
                detail_json = None
            else:
                error_info = detail_json.get("error", {})
                inner_error = error_info.get("innererror") or {}
                code = (
                    error_info.get("code")
                    or inner_error.get("code")
                    or ""
                )
                if "content_filter" in code.lower() or inner_error.get(
                    "content_filter_result"
                ):
                    message = error_info.get(
                        "message",
                        "Request blocked by Azure content moderation policy.",
                    )
                    raise ContentFilterError(message)
        if response.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
            wait_time = retry_wait
            retry_after = response.headers.get("retry-after")
            if retry_after:
                try:
                    wait_time = max(float(retry_after), retry_wait)
                except ValueError:
                    pass
            time.sleep(wait_time)
            continue
        detail = response.text
        raise RuntimeError(
            f"Request failed with status {response.status_code}: {detail}"
        )
    raise RuntimeError("Exceeded retry attempts without success.")


def extract_model_output(choice: Dict, label_lookup: Dict[str, str], default_label: str) -> Tuple[str, str]:
    message = choice.get("message") or {}
    content = message.get("content", "")
    label = normalize_label(content, label_lookup, default_label)
    return content, label


def normalize_token(token: Optional[str]) -> str:
    if token is None:
        return ""
    cleaned = token.replace("Ġ", " ").replace("▁", " ")
    return cleaned.strip().lower()


def extract_label_probabilities(
    logprob_payload: Optional[Dict], label_lookup: Dict[str, str]
) -> Dict[str, float]:
    """Return probabilities keyed by canonical label names."""
    if not logprob_payload:
        return {}
    content = logprob_payload.get("content", [])
    for item in content:
        probabilities: Dict[str, float] = {}
        token_label = resolve_label_from_token(item.get("token"), label_lookup)
        if token_label:
            logprob = item.get("logprob")
            if logprob is not None:
                probabilities[token_label] = math.exp(logprob)
        for candidate in item.get("top_logprobs") or []:
            candidate_label = resolve_label_from_token(
                candidate.get("token"), label_lookup
            )
            if candidate_label:
                probabilities[candidate_label] = max(
                    probabilities.get(candidate_label, 0.0),
                    math.exp(candidate.get("logprob", float("-inf"))),
                )
        if probabilities:
            return probabilities
    return {}


def resolve_label_from_token(token: Optional[str], label_lookup: Dict[str, str]) -> Optional[str]:
    normalized = normalize_token(token)
    if not normalized:
        return None
    direct = label_lookup.get(normalized)
    if direct:
        return direct
    for key, canonical in label_lookup.items():
        if key.startswith(normalized) or normalized.startswith(key):
            return canonical
    return None


def compute_confusion_counts(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> Tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for actual, predicted in zip(y_true, y_pred):
        if actual == 1 and predicted == 1:
            tp += 1
        elif actual == 0 and predicted == 0:
            tn += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        elif actual == 1 and predicted == 0:
            fn += 1
    return tn, fp, fn, tp


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def probability_auc(y_true: Sequence[int], y_scores: Sequence[float]) -> float:
    positives = [score for score, label in zip(y_scores, y_true) if label == 1]
    negatives = [score for score, label in zip(y_scores, y_true) if label == 0]
    if not positives or not negatives:
        return float("nan")
    wins = 0.0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    total = len(positives) * len(negatives)
    return wins / total


def precision_recall_auc(y_true: Sequence[int], y_scores: Sequence[float]) -> float:
    positives = sum(y_true)
    if positives == 0:
        return float("nan")
    pairs = sorted(zip(y_scores, y_true), key=lambda item: item[0], reverse=True)
    tp = fp = 0
    recalls = [0.0]
    precisions = [1.0]
    for score, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / positives
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recalls.append(recall)
        precisions.append(precision)
    area = 0.0
    for i in range(1, len(recalls)):
        area += precisions[i] * (recalls[i] - recalls[i - 1])
    return area


def compute_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    y_scores: Sequence[Optional[float]],
    positive_label: str,
    threshold: float,
) -> Dict[str, float]:
    labels = {positive_label}
    labels.update(y_true)
    labels.update(y_pred)
    if len(labels) != 2:
        raise ValueError(
            f"Expected exactly two labels for binary metrics, got {sorted(labels)}"
        )
    negative_label = next(iter(sorted(lab for lab in labels if lab != positive_label)))
    to_binary = {positive_label: 1, negative_label: 0}
    y_true_binary = [to_binary.get(label, 0) for label in y_true]
    y_pred_binary = [to_binary.get(label, 0) for label in y_pred]
    tn, fp, fn, tp = compute_confusion_counts(y_true_binary, y_pred_binary)
    accuracy = safe_divide(tp + tn, len(y_true_binary))
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    scores = [
        score if score is not None else to_binary[pred]
        for score, pred in zip(y_scores, y_pred)
    ]
    auroc = probability_auc(y_true_binary, scores)
    auprc = precision_recall_auc(y_true_binary, scores)
    return {
        "N": len(y_true_binary),
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Recall": recall,
        "Specificity": specificity,
        "Precision": precision,
        "F1": f1,
        "AUROC": auroc,
        "AUPRC": auprc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def choose_default_positive_label(label_order: Sequence[str]) -> str:
    heuristics = ("yes", "compute", "positive", "true", "1")
    for label in label_order:
        lowered = label.lower()
        if any(keyword in lowered for keyword in heuristics):
            return label
    if label_order:
        return label_order[-1]
    raise ValueError("No labels available to select a positive class.")


def main() -> None:
    args = parse_args()
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
    missing = [
        name
        for name, value in [
            ("AZURE_OPENAI_ENDPOINT", endpoint),
            ("AZURE_OPENAI_API_KEY", api_key),
            ("AZURE_OPENAI_DEPLOYMENT", deployment),
            ("AZURE_OPENAI_API_VERSION", api_version),
        ]
        if not value
    ]
    if missing:
        joined = ", ".join(missing)
        raise EnvironmentError(f"Missing required environment variables: {joined}")

    examples, label_lookup, label_order = load_examples(args.dataset)
    default_label = label_order[0]
    if args.positive_label:
        desired = args.positive_label.lower()
        positive_label = label_lookup.get(desired)
        if not positive_label:
            matches = [label for label in label_order if label.lower() == desired]
            if matches:
                positive_label = matches[0]
        if not positive_label:
            raise ValueError(
                f"Positive label '{args.positive_label}' is not present in the dataset labels {label_order}."
            )
    else:
        positive_label = choose_default_positive_label(label_order)

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_file.parent.mkdir(parents=True, exist_ok=True)

    url = build_request_url(endpoint, deployment, api_version)
    payload_base = {
        "temperature": 0,
        "max_completion_tokens": 16,
        "logprobs": True,
        "top_logprobs": 5,
    }

    interval = 60.0 / args.rpm if args.rpm > 0 else 0.0
    last_call: Optional[float] = None

    predictions: List[str] = []
    scores: List[Optional[float]] = []
    truncation_config = TruncationConfig(
        enabled=args.truncate_prompts,
        max_chars=args.truncate_max_chars,
        min_head=args.truncate_min_head,
        min_tail=args.truncate_min_tail,
    )

    with requests.Session() as session, args.log_file.open("w", encoding="utf-8") as log_handle:
        for example in examples:
            if last_call:
                elapsed = time.time() - last_call
                if interval > 0 and elapsed < interval:
                    time.sleep(interval - elapsed)
            payload = dict(payload_base)
            prepared_messages, truncated = apply_truncation(
                example.prompt_messages, truncation_config
            )
            payload["messages"] = prepared_messages
            try:
                response = send_chat_completion(
                    session=session,
                    url=url,
                    payload=payload,
                    api_key=api_key,
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                    retry_wait=args.retry_wait,
                )
            except ContentFilterError as exc:
                log_entry = {
                    "index": example.index,
                    "ground_truth": example.label,
                    "predicted_label": default_label,
                    "model_output": "",
                    "label_probabilities": {},
                    "logprobs": None,
                    "error": str(exc),
                    "filtered": True,
                    "prompt_truncated": truncated,
                }
                log_handle.write(json.dumps(log_entry))
                log_handle.write("\n")
                predictions.append(default_label)
                scores.append(None)
                last_call = time.time()
                if args.progress:
                    print(
                        f"[{example.index}] filtered by policy → default '{default_label}' "
                        f"(truncated={truncated})",
                        flush=True,
                    )
                continue
            choice = (response.get("choices") or [{}])[0]
            raw_output, predicted_label = extract_model_output(
                choice, label_lookup, default_label
            )
            label_probs = extract_label_probabilities(
                choice.get("logprobs"), label_lookup
            )
            score = label_probs.get(positive_label)
            log_entry = {
                "index": example.index,
                "ground_truth": example.label,
                "predicted_label": predicted_label,
                "model_output": raw_output,
                "label_probabilities": label_probs,
                "logprobs": choice.get("logprobs"),
                "prompt_truncated": truncated,
            }
            log_handle.write(json.dumps(log_entry))
            log_handle.write("\n")
            predictions.append(predicted_label)
            scores.append(score)
            last_call = time.time()
            if args.progress:
                print(
                    f"[{example.index}] predicted='{predicted_label}' "
                    f"truth='{example.label}' truncated={truncated} filtered=False",
                    flush=True,
                )

    metrics = compute_metrics(
        y_true=[example.label for example in examples],
        y_pred=predictions,
        y_scores=scores,
        positive_label=positive_label,
        threshold=args.threshold,
    )

    with args.metrics_file.open("w", encoding="utf-8") as metrics_handle:
        json.dump(metrics, metrics_handle, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
