#!/data/data/com.termux/files/usr/bin/bash
# One-shot installer for revapi on Termux (Android).
set -e
echo "[revapi] installing system packages…"
pkg update -y
pkg install -y python git x11-repo
pkg install -y chromium
# optional runtimes for other output languages (uncomment what you need)
# pkg install -y nodejs golang php ruby openjdk-17 clang
echo "[revapi] installing revapi…"
cd "$(dirname "$0")"
pip install --upgrade pip >/dev/null 2>&1 || true
pip install .
# generated Python clients usually use requests
pip install requests
echo
revapi doctor || true
echo
echo "[revapi] done. Set a model key, e.g.:  export NVIDIA_API_KEY=nvapi-...   (or ANTHROPIC_API_KEY, OPENROUTER_API_KEY, GEMINI_API_KEY, …)"
echo "[revapi] then run:  revapi"
