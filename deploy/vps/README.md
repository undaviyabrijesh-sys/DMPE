# Always-on DMPE recorder (VPS)

This deployment runs **public market-data capture only**. It does not need Delta API keys and cannot submit orders. Keep the host patched and restrict SSH access.

## Suggested small server

A 1 vCPU / 1 GB RAM Linux VPS is sufficient for the current public BTCUSD recorder and daily JSONL files. Choose a nearby India/Singapore region if latency matters for monitoring. Start with a monthly instance; measure CPU, disk and reconnect behaviour before scaling.

## Install

Run as root on a fresh Ubuntu LTS host:

```bash
apt-get update && apt-get install -y git python3 python3-venv python3-pip
useradd --system --create-home --home-dir /var/lib/dmpe --shell /usr/sbin/nologin dmpe
mkdir -p /var/lib/dmpe/live
git clone https://github.com/undaviyabrijesh-sys/DMPE.git /opt/dmpe
chown -R dmpe:dmpe /opt/dmpe /var/lib/dmpe
runuser -u dmpe -- python3 -m venv /opt/dmpe/.venv
runuser -u dmpe -- /opt/dmpe/.venv/bin/pip install --upgrade pip
runuser -u dmpe -- /opt/dmpe/.venv/bin/pip install -r /opt/dmpe/requirements.txt
cp /opt/dmpe/deploy/vps/dmpe-live-recorder.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now dmpe-live-recorder
```

Check service and logs:

```bash
systemctl status dmpe-live-recorder
journalctl -u dmpe-live-recorder -f
ls -lh /var/lib/dmpe/live
```

The recorder rotates JSONL files daily and reconnects after a WebSocket disconnect. systemd restarts the process if it exits.

## Operations and data safety

- Monitor file growth, last event timestamp, reconnects, event counts and missing-minute coverage daily.
- Back up `/var/lib/dmpe/live` to durable object storage or another host; do not rely on the VPS disk as the only copy.
- Configure log rotation/retention and disk alerts. Keep at least several weeks of raw trades, L2, funding and system-status events.
- Update code only after a reviewed commit; then run `git pull`, reinstall dependencies if changed, and restart the service.
- Do not place exchange credentials on this recorder host. No private endpoints or order permissions are required.
- GitHub Actions remains useful for finite reproducible jobs; it is not the always-on WebSocket host.

## Important limitation

The current feature builder aggregates trade and L2 snapshots. Funding-rate events are recorded in the raw JSONL stream, but their `next_funding_realization` timestamp and rate must be retained as explicit settlement observations before using them for historical funding P&L. The recorder running continuously is the first step, not proof of a fill model or trading edge.
