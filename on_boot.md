# Auto-Start Python Server in tmux on Boot (Orange Pi)

## Quick Setup Guide

### 1. Create startup script

Make file:
```bash
chmod +x /home/orangepi/Luna1-server/on_boot.sh
```

### 2. Create systemd service

Create `/etc/systemd/system/luna.service`:

```ini
[Unit]
Description=Start Luna server inside tmux
After=network.target user@1000.service

[Service]
Type=forking
ExecStart=/home/orangepi/Luna1-server/on_boot.sh
RemainAfterExit=no
User=orangepi
WorkingDirectory=/home/orangepi
# Environment=TMUX=/usr/bin/tmux

[Install]
WantedBy=multi-user.target
```

### 3. Enable and start service

```bash
sudo systemctl daemon-reload
sudo systemctl enable luna.service
sudo systemctl start luna.service
```

### 4. Verify it works

Check tmux sessions:
```bash
tmux ls
```

Attach to server:
```bash
tmux attach -t server
```

View logs:
```bash
cat /home/orangepi/luna_boot.log
```

## Notes

- Replace `orangepi` with your username if different
- Replace `Luna1-server` with your actual directory name
- The service runs as root to avoid permission issues
- Logs are saved to `/home/orangepi/luna_boot.log`
- To detach from tmux session: Press `Ctrl+B` then `D`

## Troubleshooting

If the service fails, check:
```bash
sudo systemctl status luna.service
journalctl -u luna.service
```

To manually test the script:
```bash
bash -x /home/orangepi/Luna1-server/on_boot.sh
```