#!/usr/bin/env bash
# Export Azure OpenAI credentials for the evaluation script.
# Usage: source set_azure_credentials.sh

export AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT}"
export AZURE_OPENAI_DEPLOYMENT="${AZURE_OPENAI_DEPLOYMENT}"
export AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION}"
export AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY:-REPLACE_WITH_YOUR_API_KEY}"

echo "Azure OpenAI environment variables set."
