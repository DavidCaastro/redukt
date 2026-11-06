#!/usr/bin/env bash
# Kaggle environment setup — install CLI, validate API key, test connection.
set -euo pipefail

echo "[kaggle-setup] Checking environment..."

# 1. Check Python
if ! command -v python &>/dev/null && ! command -v python3 &>/dev/null; then
    echo "ERROR: Python not found. Install Python 3.8+."
    exit 1
fi
PYTHON=$(command -v python3 2>/dev/null || command -v python)
echo "[kaggle-setup] Python: $($PYTHON --version)"

# 2. Install kaggle CLI if missing
if ! command -v kaggle &>/dev/null; then
    echo "[kaggle-setup] Installing kaggle CLI..."
    $PYTHON -m pip install --quiet kaggle
fi
echo "[kaggle-setup] Kaggle CLI: $(kaggle --version 2>/dev/null || echo 'install failed')"

# 3. Check API key
KAGGLE_JSON="$HOME/.kaggle/kaggle.json"
if [ ! -f "$KAGGLE_JSON" ]; then
    echo ""
    echo "ERROR: $KAGGLE_JSON not found."
    echo ""
    echo "Steps to fix:"
    echo "  1. Go to https://www.kaggle.com/settings"
    echo "  2. Scroll to API section"
    echo "  3. Click 'Create New Token'"
    echo "  4. Save kaggle.json to: $KAGGLE_JSON"
    echo ""
    exit 1
fi

# 4. Validate key format
if ! grep -q '"username"' "$KAGGLE_JSON" || ! grep -q '"key"' "$KAGGLE_JSON"; then
    echo "ERROR: $KAGGLE_JSON has invalid format. Expected {\"username\":...,\"key\":...}"
    exit 1
fi

# 5. Set permissions (kaggle CLI warns if too open)
chmod 600 "$KAGGLE_JSON" 2>/dev/null || true

# 6. Test API connection
echo "[kaggle-setup] Testing API connection..."
KAGGLE_USERNAME=$(grep -o '"username":"[^"]*"' "$KAGGLE_JSON" | cut -d'"' -f4)
echo "[kaggle-setup] Username: $KAGGLE_USERNAME"

if kaggle kernels list --mine --page-size 1 &>/dev/null; then
    echo "[kaggle-setup] API connection OK"
else
    echo "ERROR: API connection failed. Check your kaggle.json credentials."
    exit 1
fi

echo ""
echo "[kaggle-setup] Ready. You can now use:"
echo "  scripts/kaggle-push.sh    — push and run notebook on GPU"
echo "  scripts/kaggle-status.sh  — check status and download results"
