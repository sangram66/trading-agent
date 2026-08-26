"""
Storage: Parquet on disk, DuckDB for queries, a manifest for reproducibility.

The manifest is the point. Every partition written records its content hash, row
count and retrieval time; every verdict records the manifest hashes it consumed.
If a research conclusion cannot be tied back to exact bytes six months later it
was an anecdote, not a result.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Layers of the medallion. Bronze is append-only and never edited: if a vendor
# restates, that is a new row with a new retrieved_at, because the gap between
# first print and restatement is itself data.
LAYERS = ("bronze", "silver", "gold", "marts")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# Columns that record *when we fetched*, not *what the data says*. They are
# genuine provenance and belong in the file, but they must not enter the content
# hash: otherwise the digest changes on every pull and the manifest tracks
# download times instead of data identity, which defeats the whole point.
VOLATILE_COLUMNS = ("retrieved_at",)


def _hash_table(table: pa.Table) -> str:
    """Content hash of a table's logical data.

    Hashes a canonical Arrow IPC serialisation rather than the raw column
    buffers. Buffer hashing looks simpler but is not stable across a Parquet
    round-trip: identical data comes back with different padding and alignment,
    so `verify()` would report corruption on a perfectly good file. Schema
    metadata is stripped so that pandas index bookkeeping does not change the
    digest either, and volatile provenance columns are dropped so that two pulls
    of identical data agree.
    """
    keep = [c for c in table.column_names if c not in VOLATILE_COLUMNS]
    t = table.select(keep).replace_schema_metadata(None).combine_chunks()
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, t.schema) as writer:
        writer.write_table(t)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()[:16]


class Store:
    def __init__(self, root: str | Path = "data"):
        self.root = Path(root)
        for layer in LAYERS:
            (self.root / layer).mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text())
        return {}

    def _save_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2,
                                                 sort_keys=True))

    def path(self, layer: str, dataset: str) -> Path:
        if layer not in LAYERS:
            raise ValueError(f"unknown layer {layer!r}; expected one of {LAYERS}")
        return self.root / layer / f"{dataset}.parquet"

    def write(self, layer: str, dataset: str, df: pd.DataFrame,
              source: str = "", note: str = "") -> str:
        """Write a dataset and record it in the manifest. Returns the hash."""
        p = self.path(layer, dataset)
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, p, compression="zstd")
        digest = _hash_table(table)
        self.manifest[f"{layer}/{dataset}"] = {
            "hash": digest,
            "rows": int(len(df)),
            "columns": list(df.columns),
            "written_at": utcnow().isoformat(),
            "source": source,
            "note": note,
            "bytes": p.stat().st_size,
        }
        self._save_manifest()
        return digest

    def append(self, layer: str, dataset: str, df: pd.DataFrame,
               source: str = "", partition: str | None = None) -> str:
        """Append by writing a new dated partition, not by rewriting the file.

        The obvious implementation — read the existing Parquet, concat, write it
        back — is a trap when the store lives in git. A compressed Parquet
        rewritten daily produces a completely different blob every time, and git
        cannot delta binary files, so every day costs a full copy of the whole
        dataset. Measured over 60 daily commits: 6.8 MB monolithic vs 0.5 MB
        partitioned, and the gap widens without bound.

        So each append lands at `<layer>/<dataset>/<partition>.parquet` and old
        partitions are never touched. `read()` globs them back together.
        """
        part = partition or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        d = self.root / layer / dataset
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{part}.parquet"
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, p, compression="zstd")

        parts = sorted(q.name for q in d.glob("*.parquet"))
        total = sum((d / q).stat().st_size for q in parts)
        combined = hashlib.sha256(
            "".join(_hash_table(pq.read_table(d / q)) for q in parts).encode()
        ).hexdigest()[:16]

        self.manifest[f"{layer}/{dataset}"] = {
            "hash": combined,
            "rows": int(sum(pq.read_metadata(d / q).num_rows for q in parts)),
            "columns": list(df.columns),
            "written_at": utcnow().isoformat(),
            "source": source,
            "note": f"partitioned · {len(parts)} partitions",
            "bytes": total,
            "partitions": parts,
        }
        self._save_manifest()
        return combined

    def read(self, layer: str, dataset: str) -> pd.DataFrame:
        """Read a dataset, whether it is a single file or a partition directory."""
        d = self.root / layer / dataset
        if d.is_dir():
            parts = sorted(d.glob("*.parquet"))
            if not parts:
                raise FileNotFoundError(f"no partitions in {d}")
            return pd.concat([pd.read_parquet(p) for p in parts],
                             ignore_index=True)
        return pd.read_parquet(self.path(layer, dataset))

    def exists(self, layer: str, dataset: str) -> bool:
        d = self.root / layer / dataset
        return self.path(layer, dataset).exists() or (
            d.is_dir() and any(d.glob("*.parquet")))

    def verify(self, layer: str, dataset: str) -> bool:
        """Re-hash on disk and compare against the manifest.

        Catches silent corruption and, more usefully, catches someone having
        quietly edited a bronze file they should not have touched.
        """
        key = f"{layer}/{dataset}"
        if key not in self.manifest or not self.exists(layer, dataset):
            return False
        entry = self.manifest[key]
        if "partitions" in entry:
            d = self.root / layer / dataset
            parts = sorted(q.name for q in d.glob("*.parquet"))
            combined = hashlib.sha256(
                "".join(_hash_table(pq.read_table(d / q)) for q in parts).encode()
            ).hexdigest()[:16]
            return combined == entry["hash"]
        return _hash_table(pq.read_table(self.path(layer, dataset))) == entry["hash"]

    def query(self, sql: str):
        """DuckDB over the parquet tree. Reference tables as bronze_x, gold_y."""
        import duckdb
        con = duckdb.connect()
        for key in self.manifest:
            layer, dataset = key.split("/", 1)
            p = self.path(layer, dataset)
            if p.exists():
                view = f"{layer}_{dataset}".replace("-", "_").replace(".", "_")
                con.execute(
                    f"CREATE OR REPLACE VIEW {view} AS "
                    f"SELECT * FROM read_parquet('{p}')")
        return con.execute(sql).fetch_df()

    def summary(self) -> pd.DataFrame:
        rows = [{"dataset": k, **{kk: vv for kk, vv in v.items()
                                  if kk in ("rows", "hash", "bytes", "source")}}
                for k, v in sorted(self.manifest.items())]
        return pd.DataFrame(rows)
