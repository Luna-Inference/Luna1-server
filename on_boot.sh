#!/bin/bash
{
  echo "=== Boot at $(date) ==="
  cd /home/thomas/Luna1-server || exit 1
  /usr/bin/tmux new-session -Ad -s server "bash -c 'source myenv/bin/activate && python server.py'"
  sleep 3
  /usr/bin/tmux send-keys -t server "thomas" C-m
  echo "RKLLM Server started successfully on port 1306"
  echo "Use 'tmux attach -t rkllm-server' to view the server"
} >> /home/thomas/luna_boot.log 2>&1