"""Convert user-selected benchmark files to JSONL, preserving input order."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.parse import urlparse
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
# Import this standalone converter without masbench.__init__, which initializes
# the LLM runtime and would make local text preparation depend on API clients.
_module_spec = importlib.util.spec_from_file_location(
    "_messagesieve_preparation", ROOT / "masbench/datasets/preparation.py")
assert _module_spec is not None and _module_spec.loader is not None
_preparation = importlib.util.module_from_spec(_module_spec)
sys.modules[_module_spec.name] = _preparation
_module_spec.loader.exec_module(_preparation)
normalize, read_huggingface = _preparation.normalize, _preparation.read_huggingface
read_source = _preparation.read_source


def prepare(dataset: str, source: Path | None = None, output: Path | None = None, *,
            source_format: str = "auto",
            download: bool = False, source_url: str | None = None,
            hf_dataset: str | None = None, hf_config: str | None = None,
            hf_split: str | None = None, hf_revision: str | None = None) -> Path:
    """Convert all supplied records; evaluation configs control the sample limit."""
    spec = json.loads((ROOT / "data/splits.json").read_text(encoding="utf-8"))[dataset]
    output = Path(output) if output is not None else ROOT / spec["path"]
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if sum((source is not None, download, source_url is not None)) != 1:
        raise ValueError("Choose exactly one of --source, --download, --source-url")
    if any((hf_dataset, hf_config, hf_split, hf_revision)) and not download:
        raise ValueError("--hf-* options require --download")
    if source_format not in {"auto", "raw", "normalized"}:
        raise ValueError("source_format must be auto, raw or normalized")
    if source is not None:
        rows = read_source(dataset, Path(source))
    elif download and "file" not in spec:
        config = hf_config if hf_config is not None else (spec.get("config") if not hf_dataset else None)
        rows = read_huggingface(hf_dataset or spec["source"], config=config,
                                split=hf_split or spec["split"], revision=hf_revision)
    else:
        if download:
            if any((hf_dataset, hf_config, hf_split)):
                raise ValueError("LiveCodeBench uses a data file; use --source-url for another source")
            source_url = (
                f"https://huggingface.co/datasets/{spec['source']}"
                f"/resolve/{hf_revision or 'main'}/{spec['file']}"
            )
        parsed = urlparse(source_url or "")
        if parsed.scheme not in {"https", "http"}:
            raise ValueError("--source-url must be an HTTP(S) upstream data file")
        suffix = Path(parsed.path).suffix.lower()
        if suffix not in {".csv", ".json", ".jsonl", ".parquet"}:
            raise ValueError("URL must name a CSV, JSON, JSONL or parquet data file")
        with tempfile.TemporaryDirectory(prefix="messagesieve-upstream-") as temp:
            local = Path(temp) / Path(parsed.path).name
            with urlopen(source_url, timeout=60) as response, local.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            rows = read_source(dataset, local)
            # Keep durable provenance, not the temporary download filename.
            rows = [type(row)(row.record, row.index, source_url) for row in rows]
    records = normalize(dataset, rows, source_format=source_format)
    with tempfile.TemporaryFile() as staged:
        for record in records:
            line = (json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
            staged.write(line)
        staged.seek(0)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as handle:
            shutil.copyfileobj(staged, handle)
    return output


def main() -> None:
    datasets = json.loads((ROOT / "data/splits.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(
        description="Convert benchmark files to evaluation JSONL in input order.",
        epilog="The YAML dataset.limit controls evaluation size. Parquet requires pyarrow; HF downloads require datasets.",
    )
    parser.add_argument("--dataset", required=True, choices=datasets)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--source", type=Path, help="Local raw CSV/JSON/JSONL/parquet, split directory, or normalized JSONL")
    inputs.add_argument("--source-url", help="Upstream HTTP(S) data-file URL")
    inputs.add_argument("--download", action="store_true", help="Fetch the source listed in data/splits.json")
    parser.add_argument("--output", type=Path, help="New JSONL path (default: dataset path from data/splits.json); never overwritten")
    parser.add_argument("--source-format", choices=("auto", "raw", "normalized"), default="auto", help="Input schema (default: auto)")
    parser.add_argument("--hf-dataset", help="Override upstream HF dataset name")
    parser.add_argument("--hf-config", help="Override upstream HF configuration")
    parser.add_argument("--hf-split", help="Override upstream HF split")
    parser.add_argument("--hf-revision", help="Optional upstream revision for --download")
    args = parser.parse_args()
    try:
        print(prepare(**vars(args)))
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        parser.exit(2, f"prepare_data: {exc}\n")


if __name__ == "__main__":
    main()
