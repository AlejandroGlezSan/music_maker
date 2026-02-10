#!/usr/bin/env python3
"""
Escanea el proyecto para:
- funciones/clases duplicadas por nombre
- archivos importados pero no referenciados (heurístico)
- definiciones fuera de clases por indentación incorrecta
Uso: python scripts/find_issues.py /ruta/al/proyecto
"""
import sys
import os
import re
from collections import defaultdict

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."

py_files = []
for dirpath, dirs, files in os.walk(ROOT):
    # ignorar virtualenv y .git
    if any(p in dirpath for p in (".git", "venv", "__pycache__")):
        continue
    for f in files:
        if f.endswith(".py"):
            py_files.append(os.path.join(dirpath, f))

defs = defaultdict(list)   # name -> [file:lineno:type]
imports = defaultdict(list) # module -> [file:lineno]
uses = defaultdict(list)    # name -> [file:lineno]

def_pattern = re.compile(r'^\s*(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(')
import_pattern = re.compile(r'^\s*(?:from\s+([A-Za-z0-9_.]+)\s+import|import\s+([A-Za-z0-9_.]+))')
name_use_pattern = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)\b')

for path in py_files:
    with open(path, 'r', encoding='utf-8') as fh:
        lines = fh.readlines()
    for i, line in enumerate(lines, start=1):
        m = def_pattern.match(line)
        if m:
            kind, name = m.groups()
            defs[name].append(f"{path}:{i}:{kind}")
        m2 = import_pattern.match(line)
        if m2:
            mod = m2.group(1) or m2.group(2)
            imports[mod].append(f"{path}:{i}")
        # collect simple uses (heuristic)
        for nm in name_use_pattern.findall(line):
            uses[nm].append(f"{path}:{i}")

# Report duplicates
print("=== Duplicated definitions (name -> locations) ===")
dups = {k:v for k,v in defs.items() if len(v) > 1}
if not dups:
    print("No duplicates found.")
else:
    for name, locs in dups.items():
        print(f"{name}:")
        for l in locs:
            print(f"  - {l}")

# Report likely unused defs (defined but rarely used)
print("\n=== Possibly unused definitions (defined but few uses) ===")
for name, locs in defs.items():
    use_count = len(uses.get(name, []))
    # if defined and used only at definition site or never used
    if use_count <= len(locs):
        for l in locs:
            print(f"{name} defined at {l} (uses: {use_count})")

# Report imports that may be unused (heuristic: module name not present in uses)
print("\n=== Possibly unused imports (heuristic) ===")
for mod, locs in imports.items():
    base = mod.split('.')[0]
    if base not in uses:
        for l in locs:
            print(f"{mod} imported at {l} -> no obvious usage found")

# Check for top-level defs outside classes (indentation issues)
print("\n\n=== Top-level indentation check (defs outside classes) ===")
for path in py_files:
    with open(path, 'r', encoding='utf-8') as fh:
        text = fh.read()
    # find 'class' blocks and their spans
    class_spans = []
    for m in re.finditer(r'^\s*class\s+[A-Za-z_][A-Za-z0-9_]*\s*[:\(]', text, flags=re.M):
        class_spans.append(m.start())
    # simple heuristic: look for 'def' at column 0 (no indent)
    for m in re.finditer(r'^\s*def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(', text, flags=re.M):
        line_start = text.rfind('\n', 0, m.start()) + 1
        col = m.start() - line_start
        if col == 0:
            # top-level def (ok) but we warn if file contains a class and also top-level defs that look like methods
            if class_spans:
                lineno = text.count('\n', 0, m.start()) + 1
                print(f"{path}:{lineno} -> top-level def (check indentation)")

print("\nScan complete.")
