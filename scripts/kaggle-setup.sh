#!/usr/bin/env bash
# Kaggle environment setup — install CLI, validate API key, test connection.
# Windows-compatible: uses `python -m kaggle` instead of bare `kaggle`.
set -euo pipefail

echo "[kaggle-setup] Checking environment..."

# 1. Check Python
if ! python --version &>/dev/null; then
    echo "ERROR: Python not found. Install Python 3.8+."
    exit 1
fi
echo "[kaggle-setup] Python: $(python --version 2>&1)"

# 2. Install kaggle CLI if missing
if ! python -m kaggle --version &>/dev/null; then
    echo "[kaggle-setup] Installing kaggle CLI..."
    python -m pip install --quiet kaggle
fi
echo "[kaggle-setup] Kaggle CLI: $(python -m kaggle --version 2>&1)"

# 3. Check API key (handle paths with spaces)
KAGGLE_JSON="$USERPROFILE/.kaggle/kaggle.json"
if [ ! -f "$KAGGLE_JSON" ]; then
    # Fallback for Git Bash $HOME
    KAGGLE_JSON="$HOME/.kaggle/kaggle.json"
fi
if [ ! -f "$KAGGLE_JSON" ]; then
    echo ""
    echo "ERROR: kaggle.json not found."
    echo "  Checked: $USERPROFILE/.kaggle/kaggle.json"
    echo ""
    echo "Steps to fix:"
    echo "  1. Go to https://www.kaggle.com/settings"
    echo "  2. Scroll to Legacy API Credential"
    echo "  3. Click 'Create New Token'"
    echo "  4. Save kaggle.json to: $USERPROFILE/.kaggle/"
    echo ""
    exit 1
fi

# 4. Validate key format
if ! grep -q '"username"' "$KAGGLE_JSON" || ! grep -q '"key"' "$KAGGLE_JSON"; then
    echo "ERROR: kaggle.json has invalid format."
    exit 1
fi

# 5. Set permissions (kaggle CLI warns if too open)
chmod 600 "$KAGGLE_JSON" 2>/dev/null || true

# 6. Test API connection
echo "[kaggle-setup] Testing API connection..."
KAGGLE_USERNAME=$(grep -o '"username":"[^"]*"' "$KAGGLE_JSON" | cut -d'"' -f4)
echo "[kaggle-setup] Username: $KAGGLE_USERNAME"

if python -m kaggle kernels list --mine --page-size 1 &>/dev/null; then
    echo "[kaggle-setup] API connection OK"
else
    echo "ERROR: API connection failed. Check kaggle.json."
    exit 1
fi

echo ""
echo "[kaggle-setup] Ready. You can now use:"
echo "  scripts/kaggle-push.sh    — push and run notebook on GPU"
echo "  scripts/kaggle-status.sh  — check status and download results"
