export COMMIT_ID=7d842fb85a0275a4a8e4d7e040d2625abbf7f084
export BASE_PATH=/mnt/shared-storage-user/huanghaian/.vscode-server
mkdir -P $BASE_PATH
tar -xzvf ./vscode-server-linux-x64.tar.gz -C ./
mkdir -p $BASE_PATH/cli/servers/Stable-$COMMIT_ID/
mv ./vscode-server-linux-x64/ $BASE_PATH/cli/servers/Stable-$COMMIT_ID/server

tar -xzvf ./vscode_cli_alpine_x64_cli.tar.gz
mv ./code $BASE_PATH/code-$COMMIT_ID