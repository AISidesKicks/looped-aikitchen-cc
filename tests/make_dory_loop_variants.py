"""Make loop-count variants of the Dory GGUF by patching `dory.num_loops`.

llama.cpp dory support (PR 30306) reads the loop count from GGUF metadata
(`dory.num_loops`, uint32). The official GGUF ships with 4 loops baked in
and the server has no CLI flag for it yet, so this script makes cheap
variant copies with 1/2/3 loops. Weights are never touched.

The GLM GGUF layout note (verified 2026-10-30 on the downloaded file):
v3 header (magic, version u32, tensor_count u64, kv_count u64 = 24 bytes)
followed by metadata with NO alignment padding between keys or values.
String payloads carry their own u64 length. This deviates from some
padding-heavy GGUF writers, hence a hand-rolled scan instead of gguf-py.

Usage:
    python tests/make_dory_loop_variants.py            # makes 1 and 2 loop copies
    python tests/make_dory_loop_variants.py --cycles 3 # extra copies welcome
"""

from __future__ import annotations

import argparse
import shutil
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GGUF_DIR = ROOT / "models" / "gguf" / "dory1-4b"
SOURCE = GGUF_DIR / "NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf"
KEY = "dory.num_loops"
NAME_STEM = "NVIDIA-Nemotron-Labs-Dory1.0-4B"

SCALAR_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
TYPE_UINT32 = 4


def read_header(fh) -> tuple[int, int, int]:
    if fh.read(4) != b"GGUF":
        raise SystemExit("not a GGUF file")
    version = struct.unpack("<I", fh.read(4))[0]
    tensor_count = struct.unpack("<Q", fh.read(8))[0]
    kv_count = struct.unpack("<Q", fh.read(8))[0]
    return version, tensor_count, kv_count


def skip_value(fh, v_type: int) -> None:
    if v_type == 8:  # string: u64 length + bytes
        fh.seek(struct.unpack("<Q", fh.read(8))[0], 1)
    elif v_type == 9:  # array: u32 elem type, u64 count, then elements
        elem_type = struct.unpack("<I", fh.read(4))[0]
        count = struct.unpack("<Q", fh.read(8))[0]
        for _ in range(count):
            skip_value(fh, elem_type)
    elif v_type in SCALAR_SIZE:
        fh.seek(SCALAR_SIZE[v_type], 1)
    else:
        raise SystemExit(f"unknown GGUF value type {v_type}")


def find_loop_info(path: Path) -> tuple[int, int]:
    """Return (current num_loops, byte offset of the uint32 payload)."""
    fh = path.open("rb")
    try:
        version, _tensor_count, kv_count = read_header(fh)
        if version != 3:
            raise SystemExit(f"expected GGUF v3, got {version}")
        for _ in range(kv_count):
            k_len = struct.unpack("<Q", fh.read(8))[0]
            name = fh.read(k_len).decode("utf-8", "replace")
            v_type = struct.unpack("<I", fh.read(4))[0]
            if name == KEY:
                if v_type != TYPE_UINT32:
                    raise SystemExit(f"{KEY} has type {v_type}, expected uint32")
                return struct.unpack("<I", fh.read(4))[0], fh.tell() - 4
            skip_value(fh, v_type)
        raise SystemExit(f"{KEY} not found; layout changed upstream?")
    finally:
        fh.close()


def patch_variant(cycles: int) -> None:
    source_value, value_offset = find_loop_info(SOURCE)
    if cycles == source_value:
        out = SOURCE
        print(f"[skip] source already runs {source_value} loops")
        return None
    out = SOURCE.with_name(NAME_STEM + f"-{cycles}loop.gguf")
    shutil.copyfile(SOURCE, out)
    with out.open("r+b") as fh:
        fh.seek(value_offset)
        fh.write(struct.pack("<I", cycles))
    verify, _ = find_loop_info(out)
    if verify != cycles:
        raise SystemExit(f"patched {out.name} but re-read {verify}")
    print(f"[ok] {out.name}: {source_value} -> {verify} loops ({out.stat().st_size / (1 << 30):.2f} GiB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, nargs="+", default=[1, 2], help="loop counts to produce")
    args = ap.parse_args()
    if any(n < 1 or n > 4 for n in args.cycles):
        ap.error("llama.cpp dory support accepts 1..4 loops")
    for n in args.cycles:
        try:
            patch_variant(n)
        except OSError as exc:
            print(f"[fail] cycles={n}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
