"""Serial SuiteSparse SpMM benchmark; run from the repository root."""

import argparse
import csv
import gc
import importlib.metadata
import json
import platform
import statistics
import subprocess
import sys
import time
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "yeast": "Pajek", "ca-GrQc": "SNAP", "p2p-Gnutella30": "SNAP",
    "web-Stanford": "SNAP", "citationCiteseer": "DIMACS10",
    "pdb1HYS": "Williams", "StocF-1465": "Janna", "ldoor": "GHS_psdef",
}
SCALES = {
    "yeast": "small", "ca-GrQc": "small", "p2p-Gnutella30": "small",
    "web-Stanford": "medium", "citationCiteseer": "medium",
    "pdb1HYS": "medium", "StocF-1465": "large", "ldoor": "large",
}
DOWNLOAD_URLS = {
    name: [f"https://www.cise.ufl.edu/research/sparse/mat/{group}/{name}.mat",
           f"https://sparse.tamu.edu/mat/{group}/{name}.mat"]
    for name, group in SOURCES.items()
}


def download_matrix(name, destination):
    """Stream to a temporary file; publish only a complete MATLAB download."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in DOWNLOAD_URLS[name]:
        temporary = None
        print(f"Downloading {name}: {url}", flush=True)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                expected = response.headers.get("Content-Length")
                received = 0
                next_report = 64 * 1024 * 1024
                with tempfile.NamedTemporaryFile(
                        dir=destination.parent, prefix=destination.name + ".",
                        suffix=".part", delete=False) as output:
                    temporary = Path(output.name)
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        received += len(chunk)
                        if received >= next_report:
                            print(f"  {received / 1024**2:.0f} MiB received", flush=True)
                            next_report += 64 * 1024 * 1024
                if expected is not None and received != int(expected):
                    raise ValueError(f"Incomplete download: {received}/{expected} bytes")
            with temporary.open("rb") as downloaded:
                header = downloaded.read(128)
            if len(header) != 128 or not header.startswith(b"MATLAB 5.0 MAT-file"):
                raise ValueError("Response is not a MATLAB v5 matrix file")
            temporary.replace(destination)
            print(f"Saved {destination} ({received:,} bytes)", flush=True)
            return
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            print(f"  Download failed: {exc}", flush=True)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise RuntimeError("All download sources failed:\n" + "\n".join(errors))


def prepare_data(args):
    paths = sorted(args.data_dir.rglob("*.mat"))
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError("Duplicate matrix names found under data-dir")
    existing = {p.stem: p for p in paths}
    selected = list(dict.fromkeys(args.matrices or SOURCES))
    result = []
    for name in selected:
        path = existing.get(name)
        if path is None:
            if name not in SOURCES:
                raise ValueError(f"No local matrix or configured download for {name}")
            path = args.data_dir / SCALES[name] / f"{name}.mat"
        if not path.is_file() or path.stat().st_size == 0:
            if name not in DOWNLOAD_URLS:
                raise ValueError(f"No download configured for {name}")
            download_matrix(name, path)
        else:
            print(f"Using existing {path}", flush=True)
        result.append(path)
    order = {"small": 0, "medium": 1, "large": 2}
    result.sort(key=lambda p: (order.get(p.parent.name, 3), p.name))
    return result
FIELDS = [
    "matrix", "source", "file", "scale", "m", "n", "k", "nnz",
    "stored_entries_before_cleanup", "dtype", "value_mode", "device",
    "seed", "warmup", "iterations", "rounds", "status", "max_abs_error",
    "mean_ms", "median_ms", "min_ms", "round_mean_ms", "error",
]


def command_output(command):
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True,
                                text=True, timeout=15)
        return (result.stdout + result.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=ROOT / "data/suitesparse/mat")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark/results"
                        / datetime.now().strftime("sdaa_%Y%m%d_%H%M%S_%f"))
    parser.add_argument("--matrices", nargs="+", help="File stems, without .mat")
    parser.add_argument("--download-only", action="store_true",
                        help="Download missing matrices and exit; no SDAA dependencies needed")
    parser.add_argument("--sizes", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--device", default="sdaa:0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--value-mode", choices=["original", "implicit-ones"],
                        default="original")
    args = parser.parse_args()
    if min(args.sizes + [args.warmup, args.iterations, args.rounds]) < 1:
        parser.error("sizes, warmup, iterations and rounds must be positive")
    return args


def main():
    args = arguments()
    paths = prepare_data(args)
    if args.download_only:
        print(f"Ready: {len(paths)} matrices under {args.data_dir}", flush=True)
        return
    # Import the device integration before loading this repository's extensions.
    import numpy as np
    import torch
    import torch_sdaa
    from scipy.io import loadmat
    from scipy.sparse import issparse

    sys.path.insert(0, str(ROOT))
    from torch_sparse import SparseTensor
    import torch_sparse

    device = torch.device(args.device)
    if device.type != "sdaa" or not torch.sdaa.is_available():
        raise RuntimeError("An available SDAA device is required")
    torch.sdaa.set_device(device)
    if not any((ROOT / "torch_sparse").glob("_spmm_sdaa*.so")):
        raise RuntimeError("No _spmm_sdaa*.so in this checkout; build/install "
                           "the SDAA extensions for this repository first")

    def synchronize():
        torch.sdaa.synchronize()

    synchronize()
    args.output.mkdir(parents=True, exist_ok=False)
    versions = {}
    for package in ["torch", "torch-sdaa", "torch-scatter", "scipy", "numpy"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "arguments": {k: str(v) if isinstance(v, Path) else v
                      for k, v in vars(args).items()},
        "python": sys.version, "platform": platform.platform(),
        "versions": versions, "torch_sparse_file": torch_sparse.__file__,
        "torch_sdaa_file": torch_sdaa.__file__,
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]),
        "teco_smi": command_output(["teco-smi"]),
        "timing": "Host wall clock, synchronized around each batch; includes "
                  "Python/operator dispatch and output allocation. Excludes "
                  "loading, conversion, transfers, validation and warmup.",
        "validation": "All output elements against SciPy float32 CSR product; "
                      "abs(error) <= 1e-4 + 1e-4 * (abs(A) @ abs(X)).",
    }
    (args.output / "environment.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    # Keep the exact runner used with the results, including uncommitted changes.
    (args.output / "runner.py").write_bytes(Path(__file__).read_bytes())

    def run_case(matrix, sparse, k):
        rng = np.random.default_rng(args.seed)
        x_cpu = rng.standard_normal((matrix.shape[1], k), dtype=np.float32)
        x = torch.from_numpy(x_cpu).to(device)
        with torch.no_grad():
            out = sparse @ x
            synchronize()
            max_error = 0.0
            # Check every output element, in row chunks to limit host memory.
            abs_x = np.abs(x_cpu)
            for start in range(0, matrix.shape[0], 4096):
                stop = min(start + 4096, matrix.shape[0])
                block = matrix[start:stop]
                expected = block @ x_cpu
                scale = abs(block) @ abs_x
                actual = out[start:stop].cpu().numpy()
                error = np.abs(actual.astype(np.float64) - expected)
                if (not np.isfinite(actual).all()
                        or not np.isfinite(expected).all()
                        or not np.isfinite(scale).all()
                        or not np.all(error <= 1e-4 + 1e-4 * scale)):
                    raise RuntimeError(f"Correctness check failed in rows {start}:{stop}")
                max_error = max(max_error, float(error.max(initial=0)))
            del out, abs_x, x_cpu
            for _ in range(args.warmup):
                out = sparse @ x
                del out
            synchronize()
            times = []
            for _ in range(args.rounds):
                synchronize()
                start = time.perf_counter()
                for _ in range(args.iterations):
                    out = sparse @ x
                    del out
                synchronize()
                times.append((time.perf_counter() - start) * 1000 / args.iterations)
        return max_error, times

    with (args.output / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        file.flush()
        for path in paths:
            print(f"Loading {path.name}", flush=True)
            matrix = loadmat(path, variable_names=["Problem"],
                             struct_as_record=False, squeeze_me=True)["Problem"].A
            if not issparse(matrix) or np.iscomplexobj(matrix.data):
                raise ValueError(f"Expected a real sparse matrix: {path}")
            stored_entries = matrix.nnz
            matrix = matrix.tocsr().astype(np.float32)
            matrix.sum_duplicates()
            matrix.eliminate_zeros()
            matrix.sort_indices()
            if not np.isfinite(matrix.data).all():
                raise ValueError(f"Non-finite float32 values: {path}")
            if args.value_mode == "implicit-ones":
                matrix.data.fill(1)
            rowptr = torch.from_numpy(matrix.indptr.astype(np.int64)).to(device)
            col = torch.from_numpy(matrix.indices.astype(np.int64)).to(device)
            value = (torch.from_numpy(matrix.data).to(device)
                     if args.value_mode == "original" else None)
            sparse = SparseTensor(rowptr=rowptr, col=col, value=value,
                                  sparse_sizes=matrix.shape, is_sorted=True,
                                  trust_data=True)
            scale = ("small" if matrix.nnz < 100000 else
                     "medium" if matrix.nnz <= 5000000 else "large")
            group = SOURCES.get(path.stem)
            for k in args.sizes:
                row = dict(matrix=path.stem,
                           source=f"https://sparse.tamu.edu/{group}/{path.stem}" if group else "unknown",
                           file=str(path.resolve()), scale=scale,
                           m=matrix.shape[0], n=matrix.shape[1], k=k, nnz=matrix.nnz,
                           stored_entries_before_cleanup=stored_entries,
                           dtype="float32", value_mode=args.value_mode,
                           device=str(device), seed=args.seed, warmup=args.warmup,
                           iterations=args.iterations, rounds=args.rounds)
                print(f"  {path.stem}: A={matrix.shape}, K={k}, nnz={matrix.nnz}", flush=True)
                try:
                    error, times = run_case(matrix, sparse, k)
                    row.update(status="ok", max_abs_error=error,
                               mean_ms=statistics.mean(times),
                               median_ms=statistics.median(times), min_ms=min(times),
                               round_mean_ms=json.dumps(times))
                except Exception as exc:
                    row.update(status="failed", error=str(exc))
                    writer.writerow(row)
                    file.flush()
                    raise  # Never publish a timing after failed validation/device errors.
                writer.writerow(row)
                file.flush()
                print(f"    mean={row['mean_ms']:.6f} ms, median={row['median_ms']:.6f} ms", flush=True)
            del sparse, rowptr, col, value, matrix
            gc.collect()
            torch.sdaa.empty_cache()
    print(f"Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
