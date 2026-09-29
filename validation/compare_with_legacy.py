import argparse
import json
from pathlib import Path

import numpy as np


def compare_array(name, old, new, atol, rtol, exact=False):
    old_array = np.load(old)
    new_array = np.load(new)
    if old_array.shape != new_array.shape:
        raise AssertionError(f"{name} shape differs: {old_array.shape} != {new_array.shape}")
    if exact:
        np.testing.assert_array_equal(old_array, new_array)
    else:
        np.testing.assert_allclose(old_array, new_array, atol=atol, rtol=rtol)
    return {"name": name, "shape": list(old_array.shape), "maximum_absolute_difference": float(np.max(np.abs(old_array - new_array)))}


parser = argparse.ArgumentParser()
parser.add_argument("--spec", required=True, help="JSON list of {name, old, new, exact}")
parser.add_argument("--output", required=True)
parser.add_argument("--atol", type=float, default=1e-6)
parser.add_argument("--rtol", type=float, default=1e-5)
args = parser.parse_args()
records = []
for item in json.loads(Path(args.spec).read_text(encoding="utf-8")):
    records.append(compare_array(item["name"], item["old"], item["new"], args.atol, args.rtol, item.get("exact", False)))
Path(args.output).write_text(json.dumps({"status": "pass", "comparisons": records}, indent=2), encoding="utf-8")
