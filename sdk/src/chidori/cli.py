"""chidori run — execute any command with transparent remote GPU access."""

import argparse
import os
import sys
from pathlib import Path


def _find_libs():
    """Find the bundled .so libraries."""
    libs_dir = Path(__file__).parent / "libs"
    if libs_dir.exists() and any(libs_dir.glob("*.so*")):
        return libs_dir
    # Fallback: check if chidori-worker/build exists (dev mode)
    repo_root = Path(__file__).parent.parent.parent.parent.parent
    build_dir = repo_root / "chidori-worker" / "build"
    if build_dir.exists() and (build_dir / "libchidori.so").exists():
        return build_dir
    return None


def main():
    parser = argparse.ArgumentParser(
        prog="chidori",
        description="Run any PyTorch script on a remote GPU. Zero code changes.",
        usage="chidori run --server HOST:PORT [OPTIONS] COMMAND...",
    )
    sub = parser.add_subparsers(dest="subcmd")

    run_p = sub.add_parser("run", help="Execute a command with remote GPU access")
    run_p.add_argument("--server", "-s", required=True, action="append",
                       help="GPU server address (host:port). Can specify multiple.")
    run_p.add_argument("--optimize", choices=["inference", "training", "all"], default="all",
                       help="Which workloads to optimize (default: all)")
    run_p.add_argument("--verbose", "-v", action="store_true",
                       help="Enable Chidori debug logging")
    run_p.add_argument("--trace", action="store_true",
                       help="Enable RPC tracing with timestamps")
    run_p.add_argument("--libs", type=str, default=None,
                       help="Path to Chidori shared libraries (auto-detected)")
    run_p.add_argument("command", nargs=argparse.REMAINDER,
                       help="Command to execute (e.g., python train.py)")

    args = parser.parse_args()

    if args.subcmd != "run":
        parser.print_help()
        sys.exit(1)

    if not args.command:
        run_p.print_help()
        sys.exit(1)

    # Strip leading -- if present
    cmd = args.command
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]

    # Find libraries
    libs_dir = Path(args.libs) if args.libs else _find_libs()
    if not libs_dir:
        print("Error: Could not find Chidori libraries.", file=sys.stderr)
        print("Build them first: cd chidori-worker && make all", file=sys.stderr)
        print("Or specify --libs /path/to/build/", file=sys.stderr)
        sys.exit(1)

    libchidori = libs_dir / "libchidori.so"
    if not libchidori.exists():
        print(f"Error: {libchidori} not found", file=sys.stderr)
        sys.exit(1)

    # Build LD_PRELOAD — libchidori.so is the main interceptor
    preload_libs = [str(libchidori)]

    # Add libcuda.so.1 symlink if it exists (for CUDA driver interception)
    libcuda_sym = libs_dir / "libcuda.so.1"
    if libcuda_sym.exists():
        preload_libs = [str(libcuda_sym)]

    # Build server config
    servers = [s if ":" in s else f"{s}:43211" for s in args.server]
    server_list = ",".join(f'"{s}"' for s in servers)
    cfg = '{"id":"chidori-sdk","servers":[' + server_list + ']}'

    # Set environment
    env = os.environ.copy()
    env["LD_PRELOAD"] = " ".join(preload_libs)
    env["LD_LIBRARY_PATH"] = str(libs_dir) + ":" + env.get("LD_LIBRARY_PATH", "")
    env["CHIDORI_DIRECT_CONNECT"] = "1"
    env["CHIDORI_CFG_OVERRIDE"] = cfg

    if args.verbose:
        env["CHIDORI_DEBUG"] = "1"
    if args.trace:
        env["CHIDORI_TRACE"] = "1"

    # Enable inference optimization via sitecustomize hook
    if args.optimize in ("inference", "all"):
        env["CHIDORI_OPTIMIZE_INFERENCE"] = "1"

    # Execute the user's command
    os.execvpe(cmd[0], cmd, env)
