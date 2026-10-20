#!/usr/bin/env python3
"""Polyshark Match Lab.

    python3 lab/lab.py serve [--port 8780] [--max-workers N]   # website + scheduler
    python3 lab/lab.py build                                    # build the native runtime now
"""
import argparse
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

from server.util import DEFAULT_DATA  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lab.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the website and game scheduler")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8780)
    s.add_argument("--data", type=Path, default=DEFAULT_DATA, help="matches, builds and caches (default lab/data)")
    s.add_argument("--max-workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                   help="simultaneous games across all matches (default: CPUs - 2)")
    b = sub.add_parser("build", help="snapshot and build the current engine runtime")
    b.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = ap.parse_args(argv)

    if args.cmd == "serve":
        from server.web import serve
        serve(args.data.resolve(), args.host, args.port, args.max_workers)
    elif args.cmd == "build":
        from server.native import BuildError, Native
        native = Native(args.data.resolve())
        try:
            h = native.ensure_engine(progress=print)
        except BuildError as e:
            print(e.log[-6000:], file=sys.stderr)
            raise SystemExit(f"build failed: {e}")
        print(f"engine {h} ready in {native.engine_dir(h)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
