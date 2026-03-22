#!/bin/bash
# Format Python files with ruff (for lefthook pre-commit)

SCOPE="${1:-staged}"

if [ "$SCOPE" = "staged" ]; then
    files=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$')
elif [ "$SCOPE" = "all" ]; then
    files=$(find . -name "*.py" -not -path "./.venv/*")
else
    echo "Error: SCOPE must be 'staged' or 'all', got '$SCOPE'" >&2
    exit 1
fi

if [ -z "$files" ]; then
    echo "No Python files to format."
    exit 0
fi

count=$(echo "$files" | wc -l)
echo "Files to format:"
echo "$files" | sed 's|^\./||' | while read -r f; do echo "  - $f"; done

echo "Formatting $count Python files..."

echo "$files" | xargs ruff check --fix
echo "$files" | xargs ruff format

echo "Done."
exit 0
