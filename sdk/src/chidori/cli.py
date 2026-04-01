"""chidori — GPU-over-IP CLI. Run scripts on remote GPUs or serve your GPU."""

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
    for parent in [Path(__file__).parent] + list(Path(__file__).parents):
        build_dir = parent / "chidori-worker" / "build"
        if build_dir.exists() and (build_dir / "libchidori.so").exists():
            return build_dir
    return None


def _find_worker():
    """Find the chidori-worker binary."""
    libs_dir = _find_libs()
    if libs_dir:
        worker = libs_dir / "chidori-worker"
        if worker.exists():
            return worker
    # Check PATH
    from shutil import which
    w = which("chidori-worker")
    if w:
        return Path(w)
    return None


def cmd_run(args):
    """Execute a command with transparent remote GPU access."""
    if not args.command:
        print("Error: no command specified", file=sys.stderr)
        print("Usage: chidori run --server HOST:PORT python train.py", file=sys.stderr)
        sys.exit(1)

    cmd = args.command
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]

    libs_dir = Path(args.libs) if args.libs else _find_libs()
    if not libs_dir:
        print("Error: Could not find Chidori libraries.", file=sys.stderr)
        print("Install with: pip install chidori-gpu", file=sys.stderr)
        sys.exit(1)

    # Use libcuda.so.1 symlink as LD_PRELOAD (resolves to libchidori.so)
    libcuda = libs_dir / "libcuda.so.1"
    if not libcuda.exists():
        libcuda = libs_dir / "libchidori.so"
    if not libcuda.exists():
        print(f"Error: no libchidori.so in {libs_dir}", file=sys.stderr)
        sys.exit(1)

    servers = [s if ":" in s else f"{s}:43211" for s in args.server]
    server_list = ",".join(f'"{s}"' for s in servers)
    cfg = '{"id":"chidori-sdk","servers":[' + server_list + ']}'

    env = os.environ.copy()
    env["LD_PRELOAD"] = str(libcuda)
    env["LD_LIBRARY_PATH"] = str(libs_dir) + ":" + env.get("LD_LIBRARY_PATH", "")
    env["CHIDORI_DIRECT_CONNECT"] = "1"
    env["CHIDORI_CFG_OVERRIDE"] = cfg

    if args.verbose:
        env["CHIDORI_DEBUG"] = "1"
    if args.trace:
        env["CHIDORI_TRACE"] = "1"
    if args.optimize in ("inference", "all"):
        env["CHIDORI_OPTIMIZE_INFERENCE"] = "1"

    os.execvpe(cmd[0], cmd, env)


def cmd_serve(args):
    """Start a GPU server that accepts remote CUDA connections."""
    from chidori.worker_build import ensure_worker

    try:
        worker = ensure_worker()
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    port = args.port
    print(f"Starting Chidori GPU server on port {port}...", file=sys.stderr)
    print(f"Binary: {worker}", file=sys.stderr)
    print(f"Clients connect with: chidori run --server {_get_ip()}:{port} python script.py", file=sys.stderr)

    cmd = [str(worker), "--standalone", str(port)]
    os.execv(str(worker), cmd)


def cmd_test(args):
    """Run the built-in test suite against a GPU server."""
    libs_dir = Path(args.libs) if args.libs else _find_libs()
    if not libs_dir:
        print("Error: Could not find Chidori libraries.", file=sys.stderr)
        sys.exit(1)

    libcuda = libs_dir / "libcuda.so.1"
    if not libcuda.exists():
        libcuda = libs_dir / "libchidori.so"

    servers = [s if ":" in s else f"{s}:43211" for s in args.server]
    server_list = ",".join(f'"{s}"' for s in servers)
    cfg = '{"id":"chidori-test","servers":[' + server_list + ']}'

    # Find the test suite module
    test_module = Path(__file__).parent / "test_suite.py"
    if not test_module.exists():
        print(f"Error: test_suite.py not found at {test_module}", file=sys.stderr)
        sys.exit(1)

    env = os.environ.copy()
    env["LD_PRELOAD"] = str(libcuda)
    env["LD_LIBRARY_PATH"] = str(libs_dir) + ":" + env.get("LD_LIBRARY_PATH", "")
    env["CHIDORI_DIRECT_CONNECT"] = "1"
    env["CHIDORI_CFG_OVERRIDE"] = cfg

    # Find python
    python = sys.executable
    os.execvpe(python, [python, str(test_module)], env)


def _get_ip():
    """Best-effort: get the machine's external IP."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "YOUR_IP"


def main():
    parser = argparse.ArgumentParser(
        prog="chidori",
        description="GPU-over-IP: run any CUDA workload on a remote GPU.",
    )
    sub = parser.add_subparsers(dest="subcmd")

    # chidori run
    run_p = sub.add_parser("run", help="Run a command on a remote GPU")
    run_p.add_argument("--server", "-s", required=True, action="append",
                       help="GPU server (host:port, default port 43211)")
    run_p.add_argument("--optimize", choices=["inference", "training", "all"], default="all",
                       help="Optimize workloads (default: all)")
    run_p.add_argument("--verbose", "-v", action="store_true")
    run_p.add_argument("--trace", action="store_true")
    run_p.add_argument("--libs", type=str, default=None,
                       help="Path to shared libraries (auto-detected)")
    run_p.add_argument("command", nargs=argparse.REMAINDER)

    # chidori serve
    serve_p = sub.add_parser("serve", help="Start a GPU server")
    serve_p.add_argument("--port", "-p", type=int, default=43211,
                         help="Port to listen on (default: 43211)")

    # chidori test
    test_p = sub.add_parser("test", help="Test your GPU connection")
    test_p.add_argument("--server", "-s", required=True, action="append",
                        help="GPU server (host:port)")
    test_p.add_argument("--libs", type=str, default=None)

    args = parser.parse_args()

    if args.subcmd == "run":
        cmd_run(args)
    elif args.subcmd == "serve":
        cmd_serve(args)
    elif args.subcmd == "test":
        cmd_test(args)
    else:
        parser.print_help()
        sys.exit(0)
