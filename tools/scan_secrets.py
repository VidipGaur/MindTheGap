# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Scan files for credentials and personal data without printing their values.

Usage:
    python tools/scan_secrets.py [--strict] [--by-file] [--json OUT] PATH [PATH ...]

Each hit is reported as category, file, line and the first six characters of the
match. Secret-class hits always make the exit status 1. With --strict, e-mail
addresses, phone numbers and Windows user paths also fail the scan; that is the
mode used for anything meant to become public. Mentions of KAGGLE_KEY or
kaggle.json are listed for information only, since documentation has to name them.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

SECRET = {
    "hf_token": r"hf_[A-Za-z0-9]{20,}",
    "github_token": r"gh[pousr]_[A-Za-z0-9]{20,}",
    "sk_api_key": r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}",
    "aws_access_key": r"AKIA[0-9A-Z]{16}",
    "google_api_key": r"AIza[0-9A-Za-z_-]{35}",
    "slack_token": r"xox[baprs]-[A-Za-z0-9-]{10,}",
    "private_key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "kaggle_key_value": r"""KAGGLE_KEY\s*[=:]\s*["']?[0-9a-f]{20,}""",
    "kaggle_json_key": r""""key"\s*:\s*"[0-9a-f]{20,}\"""",
}
PERSONAL = {
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}",
    "phone": r"(?<![\w.])\+\d{1,3}[ -]?\(?\d{2,5}\)?[ -]?\d{3,5}[ -]?\d{3,5}(?![\w.])",
    "windows_user_path": r"[A-Za-z]:(?:\\{1,2}|/)Users(?:\\{1,2}|/)",
}
INFO = {
    "name_vidip": r"(?i)vidip",
    "mentions_kaggle_key": r"KAGGLE_KEY",
    "mentions_kaggle_json": r"kaggle\.json",
}
SKIP_DIRS = {".git", ".venv", "node_modules"}


def iter_files(paths):
    for p in map(Path, paths):
        if p.is_file():
            yield p
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and not SKIP_DIRS.intersection(f.parts):
                    yield f


def scan(paths):
    compiled = [(cls, name, re.compile(rx))
                for cls, table in (("secret", SECRET), ("personal", PERSONAL), ("info", INFO))
                for name, rx in table.items()]
    hits = []
    for f in iter_files(paths):
        text = f.read_bytes().decode("utf-8", errors="replace")
        # Split on "\n" only so line numbers agree with grep and editors;
        # str.splitlines() also breaks on form feeds and Unicode separators.
        for lineno, line in enumerate(text.split("\n"), 1):
            for cls, name, rx in compiled:
                for m in rx.finditer(line):
                    hits.append({"class": cls, "category": name, "file": str(f),
                                 "line": lineno, "prefix": m.group(0)[:6]})
    return hits


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--strict", action="store_true",
                    help="also fail on e-mail, phone and Windows user-path hits")
    ap.add_argument("--by-file", action="store_true",
                    help="print personal and info hits as per-file counts instead of lines")
    ap.add_argument("--json", help="write every hit (prefix only) to this file")
    args = ap.parse_args(argv)

    hits = scan(args.paths)
    secret_hits = [h for h in hits if h["class"] == "secret"]
    for h in secret_hits:
        print(f"SECRET  {h['category']:<18} {h['file']}:{h['line']}  prefix={h['prefix']}")
    for cls in ("personal", "info"):
        group = [h for h in hits if h["class"] == cls]
        if args.by_file:
            for (cat, f), n in sorted(Counter((h["category"], h["file"]) for h in group).items()):
                print(f"{cls.upper():<8}{cat:<22} {f}  count={n}")
        else:
            for h in group:
                print(f"{cls.upper():<8}{h['category']:<22} {h['file']}:{h['line']}  prefix={h['prefix']}")
    totals = Counter((h["class"], h["category"]) for h in hits)
    print("summary:", ", ".join(f"{c}/{k}={n}" for (c, k), n in sorted(totals.items())) or "no hits")
    if args.json:
        Path(args.json).write_text(json.dumps(hits, indent=1))

    failing = secret_hits
    if args.strict:
        failing = failing + [h for h in hits if h["class"] == "personal"]
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())
