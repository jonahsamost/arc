apt update
apt install -y zip vim pv tmux

curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv venv
source .venv/bin/activate

# install s5cmd
wget -q https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz
tar -xzf s5cmd_2.2.2_Linux-64bit.tar.gz
mv s5cmd /usr/local/bin/
rm s5cmd_2.2.2_Linux-64bit.tar.gz CHANGELOG.md LICENSE README.md

uv pip install -r pyproject.toml