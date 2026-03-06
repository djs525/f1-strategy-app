import os
import sys

def print_tree(directory, prefix="", ignore=None):
    if ignore is None:
        ignore = {".git", "__pycache__", "node_modules", ".venv", ".DS_Store"}
    
    entries = sorted(
        [e for e in os.scandir(directory) if e.name not in ignore],
        key=lambda e: (not e.is_dir(), e.name)
    )
    
    for i, entry in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        print(prefix + connector + entry.name)
        if entry.is_dir():
            extension = "    " if i == len(entries) - 1 else "│   "
            print_tree(entry.path, prefix + extension, ignore)

folder = sys.argv[1] if len(sys.argv) > 1 else "."
print(os.path.basename(os.path.abspath(folder)))
print_tree(folder)