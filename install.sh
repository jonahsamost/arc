apt update
apt install -y zip vim
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv venv
source .venv/bin/activate
uv pip install -r pyproject.toml