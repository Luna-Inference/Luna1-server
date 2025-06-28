#!/bin/bash
{
  echo "=== Boot at $(date) ==="
  cd /home/orangepi/Luna1-server || exit 1
  /usr/bin/tmux new-session -Ad -s server "bash -c 'source myenv/bin/activate && python server.py'"
  sleep 2
  /usr/bin/tmux send-keys -t server "orangepi" C-m
  echo "Server started successfully"
} >> /home/orangepi/luna_boot.log 2>&1
