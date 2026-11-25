"""Remove duplicate chunk IDs from a FAISS IndexFlatIP store.

The vector and its chunk metadata are filtered by the same original row
positions so index.faiss and chunks.json remain aligned.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import faiss
import numpy as np


def deduplicate(store_dir: Path) -> tuple[int, int]:
    index_path = store_dir / "index.faiss"
    chunks_path = store_dir / "chunks.json"
    temporary_index_path = store_dir / "index.dedup.tmp.faiss"
    temporary_chunks_path = store_dir / "chunks.dedup.tmp.json"

    with chunks_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)
    index = faiss.read_index(str(index_path))

    if not isinstance(chunks, list):
        raise TypeError("chunks.json must contain a top-level JSON list")
    if index.ntotal != len(chunks):
        raise RuntimeError(
            "Index/chunk mismatch before repair: "
            f"{index.ntotal} vectors != {len(chunks)} chunks"
        )
    if not isinstance(index, faiss.IndexFlatIP):
        raise TypeError(
            "This repair script only supports IndexFlatIP, got "
            f"{type(index).__name__}"
        )

    seen_ids: set[str] = set()
    kept_positions: list[int] = []
    unique_chunks: list[dict] = []

    for position, chunk in enumerate(chunks):
        chunk_id = chunk["metadata"]["chunk_id"]
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)
        kept_positions.append(position)
        unique_chunks.append(chunk)

    vectors = np.ascontiguousarray(
        np.stack([index.reconstruct(position) for position in kept_positions]),
        dtype=np.float32,
    )
    repaired_index = faiss.IndexFlatIP(index.d)
    repaired_index.add(vectors)

    faiss.write_index(repaired_index, str(temporary_index_path))
    with temporary_chunks_path.open("w", encoding="utf-8") as file:
        json.dump(unique_chunks, file, ensure_ascii=False, indent=2)

    checked_index = faiss.read_index(str(temporary_index_path))
    with temporary_chunks_path.open("r", encoding="utf-8") as file:
        checked_chunks = json.load(file)
    checked_ids = [chunk["metadata"]["chunk_id"] for chunk in checked_chunks]

    if checked_index.ntotal != len(checked_chunks):
        raise RuntimeError("Temporary index and chunk counts do not match")
    if len(checked_ids) != len(set(checked_ids)):
        raise RuntimeError("Duplicate chunk IDs remain in temporary output")
    if checked_index.d != index.d:
        raise RuntimeError("Index dimension changed during repair")

    os.replace(temporary_index_path, index_path)
    os.replace(temporary_chunks_path, chunks_path)

    return len(chunks) - len(unique_chunks), len(unique_chunks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("store_dir", type=Path)
    args = parser.parse_args()

    removed, remaining = deduplicate(args.store_dir)
    print(f"Removed duplicate rows: {removed}")
    print(f"Remaining chunks/vectors: {remaining}")


if __name__ == "__main__":
    main()
