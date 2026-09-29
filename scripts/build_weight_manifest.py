import json
import sys
from pathlib import Path

from single_photon_mei.utils.io import sha256_file


root = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
records = []
for path in sorted(item for item in root.rglob("*") if item.is_file() and item != output):
    records.append({
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    })
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({"root": str(root), "files": records}, indent=2), encoding="utf-8")
print(output)
