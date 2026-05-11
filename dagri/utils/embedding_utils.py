"""
Simple embedding save/load utilities used by the notebooks.
"""
from typing import List, Optional, Sequence, Tuple, Any
import numpy as np
import json
from pathlib import Path


def save_embeddings(path: str, names: Sequence[str], embeddings: np.ndarray, metadata: Optional[Sequence[dict]] = None) -> None:
    """Save embeddings and optional metadata to a compressed .npz file.

    Args:
        path: target .npz file path
        names: list of string identifiers (same length as embeddings)
        embeddings: numpy array of shape (N, D)
        metadata: optional list of dicts (length N)
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    names = np.asarray(names, dtype=object)
    if metadata is None:
        meta_json = np.asarray([None] * len(names), dtype=object)
    else:
        # store JSON strings to avoid object-array issues
        meta_json = np.asarray([json.dumps(m) if m is not None else None for m in metadata], dtype=object)
    np.savez_compressed(p, names=names, embeddings=embeddings, metadata=meta_json)


def load_embeddings(path: str) -> Tuple[List[str], np.ndarray, Optional[List[dict]]]:
    """Load embeddings saved by `save_embeddings`.

    Returns:
        names: list of strings
        embeddings: numpy array
        metadata: list of dicts or None
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    data = np.load(p, allow_pickle=True)
    names = list(data['names'].tolist())
    embeddings = data['embeddings']
    meta_arr = data.get('metadata', None)
    if meta_arr is None:
        metadata = None
    else:
        metadata = []
        for j in meta_arr.tolist():
            if j is None:
                metadata.append(None)
            else:
                try:
                    metadata.append(json.loads(j))
                except Exception:
                    metadata.append(None)
    return names, embeddings, metadata


def exists(path: str) -> bool:
    return Path(path).exists()
