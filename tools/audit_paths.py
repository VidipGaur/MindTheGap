# Copyright 2026 Vidip Gaur, Vidu Gaur
"""Report Windows-specific paths and commands left in scripts/.

Usage:
    python tools/audit_paths.py [--all] [SCRIPTS_DIR]

Every hit is classed as code; as prose when it sits in a comment or a
docstring; or as winbranch when it sits inside an `if os.name == "nt":` block,
where Windows behaviour was kept verbatim next to a POSIX equivalent. The exit
status is 1 when a script that is not listed in tools/dev_only.txt has a code
hit in any category except `id_split`. `id_split` (split("_")[0] used to
recover a slide id) is reported for information only; the occurrences are
listed in docs/LIMITATIONS.md and left unchanged.
"""

import argparse
import ast
import io
import re
import sys
import tokenize
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent

PATTERNS = {
    "windows_drive": r"(?<![A-Za-z0-9])[A-Z]:(?:\\|/)(?=[A-Za-z_.$])",
    "users_dir": r"/Users/",
    "username": r"(?i)vidip",
    "exe": r"\.exe\b",
    "ps_env": r"\$env:",
    "powershell": r"(?i)powershell",
    "windows_cmd": r"\b(?:taskkill|tasklist|Get-WmiObject|Get-Process|Get-ChildItem)\b",
    "startfile": r"os\.startfile",
    "appdata": r"AppData",
    "id_split": r"""\.split\(\s*["']_["']\s*\)\s*\[\s*0\s*\]""",
}
INFO_ONLY = {"id_split"}


def dev_only():
    lines = (HERE / "dev_only.txt").read_text().splitlines()
    return {s.strip() for s in lines if s.strip() and not s.startswith("#")}


def prose_spans(src):
    """(start, end) positions, as (line, col) pairs, of comments and docstrings."""
    spans = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            spans.append((tok.start, tok.end))
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                d = body[0]
                spans.append(((d.lineno, d.col_offset), (d.end_lineno, d.end_col_offset)))
    return spans


def windows_branch_lines(src):
    """Line numbers inside the body of `if os.name == "nt":` blocks."""
    lines = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            t = node.test
            if (isinstance(t.left, ast.Attribute) and t.left.attr == "name"
                    and isinstance(t.left.value, ast.Name) and t.left.value.id == "os"
                    and len(t.ops) == 1 and isinstance(t.ops[0], ast.Eq)
                    and isinstance(t.comparators[0], ast.Constant)
                    and t.comparators[0].value == "nt"):
                for stmt in node.body:
                    lines.update(range(stmt.lineno, stmt.end_lineno + 1))
    return lines


def backslash_joins(src):
    """String literals holding a backslash that are concatenated with `+` or
    passed to str.split, i.e. Windows path building by hand. The literal is
    evaluated, so "\\n" escapes do not count and raw strings do."""
    raw = [t for t in tokenize.generate_tokens(io.StringIO(src).readline)
           if t.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT,
                             tokenize.INDENT, tokenize.DEDENT)]
    # From Python 3.12 an f-string arrives as FSTRING_START/MIDDLE/END tokens;
    # fold each one back into a single (start, text, value) item.
    fstart = getattr(tokenize, "FSTRING_START", None)
    fend = getattr(tokenize, "FSTRING_END", None)
    items, i = [], 0
    while i < len(raw):
        t = raw[i]
        if fstart is not None and t.type == fstart:
            j, depth = i, 0
            while j < len(raw):
                depth += raw[j].type == fstart
                depth -= raw[j].type == fend
                if depth == 0:
                    break
                j += 1
            middle = "".join(x.string for x in raw[i:j + 1]
                             if x.type == getattr(tokenize, "FSTRING_MIDDLE", -1))
            raw_prefix = "r" in t.string.lower()
            has_bs = "\\" in middle if raw_prefix else bool(re.search(r"\\(?![ntr0'\"\\{}])", middle))
            items.append(("str", t.start, has_bs))
            i = j + 1
            continue
        if t.type == tokenize.STRING:
            try:
                value = ast.literal_eval(t.string)
            except (ValueError, SyntaxError):
                value = t.string
            items.append(("str", t.start, isinstance(value, str) and "\\" in value))
        else:
            items.append(("tok", t.start, t.string))
        i += 1
    out = []
    for k, (kind, start, payload) in enumerate(items):
        if kind != "str" or not payload:
            continue
        prev = items[k - 1][2] if k and items[k - 1][0] == "tok" else ""
        nxt = items[k + 1][2] if k + 1 < len(items) and items[k + 1][0] == "tok" else ""
        before_call = items[k - 2][2] if k >= 2 and items[k - 2][0] == "tok" else ""
        if prev == "+" or nxt == "+" or (prev == "(" and before_call == "split"):
            out.append(start)
    return out


def audit_file(path):
    src = path.read_text(encoding="utf-8")
    spans = prose_spans(src)
    winlines = windows_branch_lines(src)

    def kind(pos):
        if any(a <= pos < b for a, b in spans):
            return "prose"
        return "winbranch" if pos[0] in winlines else "code"

    hits = []
    for lineno, line in enumerate(src.split("\n"), 1):
        for cat, rx in PATTERNS.items():
            for m in re.finditer(rx, line):
                hits.append((cat, lineno, kind((lineno, m.start()))))
    for lineno, col in backslash_joins(src):
        hits.append(("backslash_join", lineno, kind((lineno, col))))
    return hits


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("scripts_dir", nargs="?", default=str(HERE.parent / "scripts"))
    ap.add_argument("--all", action="store_true", help="list every hit, not only per-file counts")
    args = ap.parse_args(argv)

    skip = dev_only()
    failing = 0
    totals = Counter()
    print(f"{'file':<36} {'status':<9} {'category':<15} {'code':>4} {'prose':>5} {'winbranch':>9}")
    for f in sorted(Path(args.scripts_dir).rglob("*.py")):
        hits = audit_file(f)
        status = "DEV-ONLY" if f.name in skip else "gated"
        for (cat, kind), n in Counter((c, k) for c, _, k in hits).items():
            totals[(status, cat, kind)] += n
        by_cat = Counter((c, k) for c, _, k in hits)
        for cat in sorted({c for c, _, _ in hits}):
            print(f"{f.name:<36} {status:<9} {cat:<15} {by_cat[(cat, 'code')]:>4} "
                  f"{by_cat[(cat, 'prose')]:>5} {by_cat[(cat, 'winbranch')]:>9}")
            if status == "gated" and cat not in INFO_ONLY and by_cat[(cat, "code")]:
                failing += by_cat[(cat, "code")]
        if args.all:
            for cat, lineno, kind in hits:
                print(f"    {f.name}:{lineno}  {cat}  {kind}")
    print()
    for status in ("gated", "DEV-ONLY"):
        parts = [f"{c}/{k}={n}" for (s, c, k), n in sorted(totals.items()) if s == status]
        print(f"{status} totals: " + (", ".join(parts) or "none"))
    print(f"gate: {failing} code hit(s) in gated scripts outside {sorted(INFO_ONLY)}")
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())
