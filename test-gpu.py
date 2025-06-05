from pathlib import Path
import hashlib, os, collections

ROOT = Path("preprocessed_data")                         # split sonrası klasör
hash2split = collections.defaultdict(set)

for split in ("train", "val", "test"):
    for f in (ROOT / split).rglob("*.npy"):
        h = hashlib.md5(Path(f).read_bytes()).hexdigest()
        hash2split[h].add(split)

duplicates = {h:s for h,s in hash2split.items() if len(s) > 1}
print("Tam kopya çakışması:", len(duplicates))
