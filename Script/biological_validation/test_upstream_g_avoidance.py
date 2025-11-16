#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import copy
import re
from typing import Optional, Tuple


def extract_L_R_struct(user_content: str) -> Tuple[str, str, str]:
    """
    Expecting format:
      'L:<...>, A:A, R:<...>, Alu Vienna Structure:<...>'
    Returns (L, R, structure).
    Falls back to a regex if the simple split fails.
    """
    try:
        parts = [p.strip() for p in user_content.split(", ")]
        L = parts[0].split("L:")[-1].strip()
        R = parts[2].split("R:")[-1].strip()
        return L, R
    except Exception:
        m = re.search(
            r"L:([ACGTN]+).*?A:A.*?R:([ACGTN]+)",
            user_content
        )
        if not m:
            raise ValueError(f"Failed to parse user content: {user_content[:120]}...")
        return m.group(1), m.group(2)


def rebuild_user_content(L: str, R: str) -> str:
    """
    Rebuild the user message content string after sequence modifications.
    """
    return f"L:{L}, A:A, R:{R}"


def get_first_user_message_index(messages: list) -> Optional[int]:
    """
    Return the index of the first message with role == 'user'.
    """
    for i, m in enumerate(messages):
        if isinstance(m, dict) and m.get("role") == "user":
            return i
    return None


def process_file(input_path: str, out_original_path: str, out_mutated_path: str):
    """
    - Read JSONL with chat-style 'messages' objects.
    - Select records where:
        * L is NOT empty, and
        * the base immediately before the central 'A' (i.e., last base of L) is NOT 'G'.
    - Write:
        1) The original records to out_original_path (unchanged).
        2) A mutated copy to out_mutated_path where that preceding base is forced to 'G'
           by changing the last character of L to 'G' (length preserved).
    - Records with empty L are fully excluded (not written to any output).
    """
    total = 0
    selected = 0
    mutated = 0
    skipped_parse = 0
    skipped_emptyL = 0
    skipped_hadG = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(out_original_path, "w", encoding="utf-8") as f_orig, \
         open(out_mutated_path, "w", encoding="utf-8") as f_mut:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1

            # Parse JSON line
            try:
                obj = json.loads(line)
            except Exception:
                skipped_parse += 1
                continue

            # Find user message
            msgs = obj.get("messages", [])
            uidx = get_first_user_message_index(msgs)
            if uidx is None:
                skipped_parse += 1
                continue

            content = msgs[uidx].get("content", "")
            try:
                L, R = extract_L_R_struct(content)
            except Exception:
                skipped_parse += 1
                continue

            # Exclude empty L entirely
            if not L:
                skipped_emptyL += 1
                continue

            # Select only if last base of L is NOT 'G'
            if L[-1] == 'G':
                skipped_hadG += 1
                continue

            # (1) Write original unchanged
            f_orig.write(json.dumps(obj, ensure_ascii=False) + "\n")
            selected += 1

            # (2) Mutate: force last base of L to 'G'
            new_L = L[:-1] + 'G'
            mutated_obj = copy.deepcopy(obj)
            mutated_content = rebuild_user_content(new_L, R)
            mutated_obj["messages"][uidx]["content"] = mutated_content
            f_mut.write(json.dumps(mutated_obj, ensure_ascii=False) + "\n")
            mutated += 1

    print("Done.")
    print(f"Total lines read: {total}")
    print(f"Selected (non-empty L and no 'G' before A): {selected}")
    print(f"Mutated (forced last base of L to 'G'): {mutated}")
    print(f"Skipped (parse failures): {skipped_parse}")
    print(f"Skipped (empty L, fully excluded): {skipped_emptyL}")
    print(f"Skipped (already had 'G' before A): {skipped_hadG}")
    print(f"Outputs:\n - {out_original_path}\n - {out_mutated_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Select records with non-empty L and NO 'G' immediately before the central A; also write copies forcing that base to 'G'."
    )
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("out_original", help="Output JSONL of originals (no G before A)")
    parser.add_argument("out_mutated", help="Output JSONL of mutated copies (force G before A)")
    args = parser.parse_args()
    process_file(args.input, args.out_original, args.out_mutated)


if __name__ == "__main__":
    main()
