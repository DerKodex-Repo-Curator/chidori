"""Chidori worker builder — compiles chidori-worker from bundled source.

On first `chidori serve`, detects CUDA toolkit and compiles the worker binary
from C source bundled inside the pip package. Caches at ~/.chidori/bin/.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _cache_dir() -> Path:
    """Return the binary cache directory, creating it if needed."""
    d = Path.home() / ".chidori" / "bin"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cached_binary() -> Path:
    """Path to the cached chidori-worker binary."""
    return _cache_dir() / "chidori-worker"


def _source_dir() -> Path:
    """Path to the bundled C source inside the pip package."""
    return Path(__file__).parent / "src"


def _find_cuda() -> str:
    """Find CUDA toolkit root. Returns path or raises."""
    # Check CUDA_PATH env first
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path and Path(cuda_path).is_dir():
        return cuda_path

    # Check common locations
    for p in ["/usr/local/cuda", "/opt/cuda", "/usr/local/cuda-12.8",
              "/usr/local/cuda-12.6", "/usr/local/cuda-12.4", "/usr/local/cuda-12.1"]:
        if Path(p).is_dir():
            return p

    # Try nvcc in PATH
    nvcc = shutil.which("nvcc")
    if nvcc:
        return str(Path(nvcc).parent.parent)

    raise FileNotFoundError(
        "CUDA toolkit not found. Set CUDA_PATH or install CUDA toolkit.\n"
        "  Example: export CUDA_PATH=/usr/local/cuda"
    )


def _find_compiler() -> str:
    """Find a C compiler. Returns path or raises."""
    for cc in ["gcc", "cc"]:
        path = shutil.which(cc)
        if path:
            return path
    raise FileNotFoundError(
        "C compiler not found. Install gcc.\n"
        "  apt install gcc   (Ubuntu/Debian)\n"
        "  dnf install gcc   (Fedora/RHEL)"
    )


def _needs_rebuild() -> bool:
    """Check if the cached binary needs to be (re)built."""
    binary = _cached_binary()
    if not binary.exists():
        return True
    # Check if source is newer than binary
    src = _source_dir()
    if not src.exists():
        return False
    binary_mtime = binary.stat().st_mtime
    for f in src.rglob("*"):
        if f.is_file() and f.stat().st_mtime > binary_mtime:
            return True
    return False


def build_worker() -> Path:
    """Build chidori-worker from bundled source. Returns path to binary.

    Compiles with gcc, links against CUDA libraries. Caches result.
    """
    binary = _cached_binary()

    if binary.exists() and not _needs_rebuild():
        return binary

    src = _source_dir()
    if not src.exists():
        raise FileNotFoundError(
            "Bundled source not found. Reinstall: pip install --force-reinstall chidori-gpu"
        )

    cuda_path = _find_cuda()
    cc = _find_compiler()

    src_server = src / "server"
    src_protocol = src / "protocol"
    src_include = src / "include"

    sources = [
        src_server / "chidori_worker.c",
        src_server / "cuda_dispatch.c",
        src_server / "replay.c",
        src_protocol / "chidori_net.c",
    ]

    # Verify all source files exist
    for s in sources:
        if not s.exists():
            raise FileNotFoundError(f"Missing source file: {s}")

    cmd = [
        cc,
        "-D_GNU_SOURCE", "-Wall", "-Wextra", "-O2",
        f"-I{src_include}",
        f"-I{src_server}",
        f"-I{cuda_path}/include",
        "-o", str(binary),
    ] + [str(s) for s in sources] + [
        "-lpthread", "-lzstd",
        f"-L{cuda_path}/lib64",
        "-lcuda", "-lcudart", "-lcublas", "-lcublasLt", "-lcudnn",
    ]

    print("Building chidori-worker for your GPU...", file=sys.stderr)
    print(f"  CC: {cc}", file=sys.stderr)
    print(f"  CUDA: {cuda_path}", file=sys.stderr)
    print(f"  Output: {binary}", file=sys.stderr)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        raise RuntimeError("Compilation timed out after 120s")

    if result.returncode != 0:
        raise RuntimeError(
            f"Compilation failed (exit {result.returncode}):\n{result.stderr}"
        )

    if result.stderr:
        # Show warnings
        for line in result.stderr.strip().split("\n"):
            if "warning:" in line:
                print(f"  {line}", file=sys.stderr)

    print(f"  Done. Binary cached at {binary}", file=sys.stderr)
    return binary


def ensure_worker() -> Path:
    """Return path to chidori-worker, building if needed."""
    binary = _cached_binary()
    if binary.exists() and not _needs_rebuild():
        return binary

    # Check if bundled source exists — if not, fall back to PATH
    src = _source_dir()
    if not src.exists():
        from shutil import which
        w = which("chidori-worker")
        if w:
            return Path(w)
        raise FileNotFoundError(
            "chidori-worker not found. Either:\n"
            "  1. Run 'chidori serve' on a machine with CUDA toolkit (auto-builds from source)\n"
            "  2. Build manually: cd chidori-worker && make all"
        )

    return build_worker()
