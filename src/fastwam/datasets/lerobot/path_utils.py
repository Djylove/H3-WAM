from pathlib import Path
from typing import Iterable


def is_lerobot_dataset_dir(path: str | Path) -> bool:
    path = Path(path)
    return (path / "meta" / "info.json").is_file()


def has_lerobot_payload(path: str | Path) -> bool:
    path = Path(path)
    return (path / "data").is_dir()


def expand_lerobot_dataset_dirs(dataset_dirs: Iterable[str | Path]) -> list[str]:
    """Expand parent directories into concrete LeRobot dataset roots.

    LeRobot v3 exports are often stored as collections like
    ``task_xxx/batch_xxx/{data,meta,videos}``.  Training configs can point at the
    collection root and this helper resolves it to the batch-level dataset roots.
    """
    expanded: list[str] = []
    seen: set[str] = set()

    for raw_dir in dataset_dirs:
        root = Path(str(raw_dir)).expanduser()
        candidates: list[Path]
        if is_lerobot_dataset_dir(root):
            candidates = [root]
        elif root.exists():
            candidates = sorted(
                info.parent.parent
                for info in root.rglob("meta/info.json")
                if has_lerobot_payload(info.parent.parent)
            )
        else:
            candidates = [root]

        for candidate in candidates:
            candidate_str = str(candidate)
            if candidate_str not in seen:
                seen.add(candidate_str)
                expanded.append(candidate_str)

    return expanded
