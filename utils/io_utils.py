import csv
import json
from pathlib import Path
from typing import Any

import torch


def ensure_parent_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def save_json(data: Any, path: str | Path) -> Path:
    target = ensure_parent_dir(path)
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return target


def save_tensor_bundle(bundle: dict[str, Any], path: str | Path) -> Path:
    target = ensure_parent_dir(path)
    torch.save(bundle, target)
    return target


def load_tensor_bundle(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location="cpu")


def save_csv_rows(rows: list[dict[str, Any]], path: str | Path) -> Path:
    target = ensure_parent_dir(path)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return target
