# myLLM
Install and config Volta
bash
curl https://get.volta.sh | bash


bash
grep -q "VOLTA" ~/.zshrc 2>/dev/null || cat << 'EOF' >> ~/.zshrc
# VOLTA
export VOLTA_HOME="$HOME/.volta"
export PATH="$VOLTA_HOME/bin:$PATH"
# VOLTA END
EOF


bash
source ~/.zshrc
