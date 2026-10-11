"""
download_data.py: fetch the six SEC Form ADV snapshots used by the analysis into data/raw/.

The SEC requires a descriptive User-Agent with a contact address
(https://www.sec.gov/os/accessing-edgar-data). Set SEC_USER_AGENT, e.g.
    SEC_USER_AGENT="Jane Doe jane@example.com" python scripts/download_data.py

Files that already exist are skipped, so the script is safe to re-run.
"""
from __future__ import annotations

import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ria_adv import SEC_URLS  # noqa: E402

RAW = ROOT / "data" / "raw"


def main() -> None:
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        sys.exit("Set SEC_USER_AGENT to 'Your Name your@email.com' (SEC fair-access policy).")
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in SEC_URLS.items():
        dest = RAW / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"skip  {name} (already downloaded)")
            continue
        print(f"fetch {name} ...", end=" ", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": ua})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(data)
        tmp.rename(dest)
        print(f"{len(data) / 1e6:.1f} MB")
        time.sleep(1)  # stay well under SEC's 10 requests/second limit


if __name__ == "__main__":
    main()
