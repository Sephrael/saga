# setup.sh
# Root-level script to set up the Python environment

set -euo pipefail

python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install ruff mypy pytest-cov

echo "Environment ready. Activate with: source venv/bin/activate"
