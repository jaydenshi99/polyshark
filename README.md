# polyshark
Simulator and engine for The Battle of Polytopia.

## Match Lab

```bash
cmake -S . -B build          # once
python3 lab/lab.py serve     # http://127.0.0.1:8780
```

Run matches between bots, analyse positions and run native self-play. See [lab/README.md](lab/README.md),
the bot protocol in [lab/PROTOCOL.md](lab/PROTOCOL.md), and [bots/](bots) to write a bot.
