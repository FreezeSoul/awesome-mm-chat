# export HOME=/mnt/shared-storage-user/huanghaian

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/mnt/shared-storage-user/huanghaian/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/mnt/shared-storage-user/huanghaian/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/mnt/shared-storage-user/huanghaian/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/mnt/shared-storage-user/huanghaian/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
export PATH=/usr/local/nvidia/bin/:/mnt/shared-storage-user/huanghaian/.local/bin:/mnt/shared-storage-user/huanghaian/opt/codex-npm/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

source /mnt/shared-storage-user/huanghaian/proxy_file

export NVM_DIR="/mnt/shared-storage-user/huanghaian/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion
