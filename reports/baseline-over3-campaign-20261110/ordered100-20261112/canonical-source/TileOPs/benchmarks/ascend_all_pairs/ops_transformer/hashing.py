"""Hash individual source files and recursively selected source directories."""
from pathlib import Path


def source_hashes(*paths):
    from benchmarks.ascend_baselines import source_hashes as files_only
    files = []
    for value in paths:
        path = Path(value)
        files.extend(sorted(p for p in path.rglob('*') if p.is_file())
                     if path.is_dir() else [path])
    return files_only(*files)
