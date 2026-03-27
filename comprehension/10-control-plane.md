# 10. Go Control Plane

## Module Structure

- **Module path:** `github.com/fpolica91/chidori/control-plane`
- **Go version:** 1.20
- **Key dependencies:**
  - `gorilla/mux` -- HTTP routing
  - `go-nvml` -- NVIDIA Management Library bindings
  - `gorm` -- ORM (PostgreSQL + SQLite)
  - `go-jose` -- JWT token validation (Auth0 RS256)
  - `prometheus` -- metrics exposition
  - `zap` -- structured logging
  - `sentry-go` -- error reporting
  - `go-memdb` -- in-memory database

---

## cmd/agent/ -- GPU Host Agent

### Initialization (`app.NewAgent`)

The agent is the process that runs on every machine with GPUs. It starts up as follows:

1. **Optional TLS setup** -- loads certificate/key if configured.
2. **Loads `.env`** via `godotenv` for local configuration.
3. **HTTP server** on `--address` (default `0.0.0.0:43210`).
4. **Parses CLI flags:**
   - `--labels` -- comma-separated `key=value` pairs for agent labeling.
   - `--taints` -- comma-separated `key=value` pairs (similar to Kubernetes taints).
   - `--pool-id` -- assigns the agent to a named pool for session routing.
5. **Resolves `--bin-path`** -- defaults to the directory containing the agent executable; used to find `chidori-worker`.
6. **Detects GPUs** via `cmdgpu.DetectGpus()` (NVML).
7. **Creates `MetricsProvider`** for per-GPU telemetry polling.
8. **Registers HTTP endpoints** for status, session management, and Prometheus metrics.

### GPU Detection (`gpu/gpu.go`)

```
nvml.Init()
nvml.DeviceGetCount()
for each device index:
    nvml.DeviceGetHandleByIndex(i)
    collect: name, UUID, memory total, PCI bus ID,
             vendor ID, device ID, driver version
```

### GPU Metrics (`gpu/metrics.go`)

`MetricsProvider` polls every **1 second** and collects per-GPU:

- GPU utilization (%)
- Memory utilization (%)
- VRAM used (bytes)
- Temperature (C)
- Power draw / power limit (W)
- Clock speeds (SM, memory)
- Fan speed (%)

Metrics are distributed to both the Prometheus `/metrics` endpoint and the controller sync consumers.

### Session Management (`app/session.go`)

A `Session` holds:
- The set of GPUs selected for the session.
- A map of `Connection` objects (one per active worker process).

**`Session.Run()`:** Spawns a goroutine that waits on context cancellation. When cancelled, it releases the assigned GPUs back to the pool.

**`Session.Connect()`:** Creates a `Connection` if one does not already exist, then forwards the raw TCP socket to the worker process via SCM_RIGHTS.

### Worker Process Management (`app/connection.go`)

**`Connection.Start()`:**

1. Creates **two Unix socketpairs** for IPC (one write, one read).
2. Spawns the `chidori-worker` binary with arguments:
   - `--id <session_id>`
   - `--log_file <path>`
   - `--ipc_write <fd>`
   - `--ipc_read <fd>`
   - `--pcibus <pci_address>`
3. Inherits the IPC file descriptors into the child process via `cmd.ExtraFiles`.
4. Captures stderr to a log file.
5. Monitors the child process; closes IPC sockets on exit.

**`Connection.Connect()`:**

1. Extracts the raw TCP file descriptor from a `net.Conn`.
2. Sends the TCP fd to the worker process via `SCM_RIGHTS` (`unix.SendmsgN`).
3. Performs a **handshake:** reads 1 byte (worker ready), writes 1 byte (continue).
4. Retains a reference to the hijacked connection to prevent Go's garbage collector from closing the underlying fd.

### SCM_RIGHTS Implementation (`connection_linux.go`)

- **`setupIpc()`:** Creates an `AF_LOCAL` socketpair using `unix.Socketpair()`.
- **`inheritFiles()`:** Appends file descriptors to `cmd.ExtraFiles` so they are available in the child process.
- **`forwardSocket()`:** Uses `unix.UnixRights()` + `unix.SendmsgN()` to pass the TCP socket fd to the worker.

### Controller Communication (`app/controller.go`)

The agent maintains a persistent relationship with the controller:

1. **Registration:** `POST /v1/register/agent` with hostname, address, version, GPU list, labels, taints, and pool ID.
2. **1-second tick goroutine:**
   - Fetches current agent state from the controller.
   - Processes **assigned sessions** (state `SessionAssigned`) by starting worker processes.
   - Processes **canceling sessions** by shutting down workers.
   - Sends `UpdateAgent` with current state and GPU metrics.

### Agent HTTP Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/status` | Health check (no auth required) |
| POST | `/v1/request/session` | Request a session (removed when controller-connected) |
| GET | `/v1/session/{id}` | Session details |
| DELETE | `/v1/session/{id}` | Cancel a session |
| POST | `/v1/connect/session/{id}` | HTTP hijack, 101 Switching Protocols, SCM_RIGHTS to worker |
| GET | `/metrics` | Prometheus metrics exposition |

---

## cmd/controller/ -- Central Scheduler

### Storage Backends

The controller supports three storage backends:

1. **GORM:** Supports PostgreSQL and SQLite. Uses row-level locking for session assignment to prevent race conditions.
2. **PostgreSQL (direct SQL):** Direct SQL queries with aggregate percentile calculations for metrics.
3. **MemDB:** In-memory database. Pool and permission operations are marked TODO.

### Backend Scheduler (1-second ticker)

The scheduler runs every second and performs:

1. **Mark agents "missing"** if they have not updated in 30 seconds.
2. **Remove agents** that have been "missing" for 5 minutes.
3. **Iterate queued sessions** and attempt assignment to a matching agent:
   - **Pool ID** must match.
   - **Label matching:** session `MatchLabels` must be a subset of agent labels.
   - **Taint tolerance:** agent taints must be a subset of session tolerations.
   - **GPU capacity:** simulate assigning existing sessions, then find matching GPUs by VRAM requirements and PCI bus address.

### Frontend Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/status` | Legacy Node.js-era format |
| GET | `/v1/status` | JSON status |
| POST | `/v1/register/agent` | Agent registration |
| GET/PUT | `/v1/agent/{id}` | Agent details/update |
| GET | `/v1/agents` | List agents (optional `?pool_id=X` filter) |
| POST | `/v1/request/session` | Request a new session |
| GET/DELETE | `/v1/session/{id}` | Session details/cancel |
| PUT/GET/DELETE | `/v1/pool/{id}` | Pool management |
| GET | `/v1/pool/{id}/permissions` | Pool permissions |
| GET/DELETE | `/v1/user/permissions/{id}` | User permissions |
| PUT | `/v1/user/permissions` | Update user permissions |

### Database Models (GORM)

- **Agent:** UUID, state, hostname, address, version, GPUs (JSON), total VRAM, labels, taints, pool reference.
- **Session:** UUID, agent_id, state (`queued` / `assigned` / `active` / `canceling` / `closed`), address, GPUs, requirements, labels, tolerates, pool.
- **Connection:** UUID, session_id, exit_code, PID, process_name.
- **Permission:** user_id, pool_id, permission_type (`create_session` / `register_agent` / `admin`).
- **Pool:** UUID, name, max_agents.
- **KeyValue:** Generic key-value store used for labels, taints, and requirements.

---

## cmd/juicify/ -- Client CLI Launcher

### Configuration

- Reads `chidori.cfg` from the binary's directory (JSON format):
  ```json
  {
    "id": "session-id",
    "servers": ["gpu-server:43210"],
    "accessToken": "...",
    "requirements": { ... }
  }
  ```
- CLI flags: `--address`, `--access-token`, `--queue-timeout`.

### Session Lifecycle

1. **`requestSession()`:** `POST /v1/request/session` to the controller.
2. **`waitForSession()`:** Polls `GET /v1/session/{id}` until state becomes `active`.
3. On success, the assigned server address **replaces** the original config's server list.
4. **`cancelSession()`** is deferred for cleanup on exit.

### Application Launch

- **`runCommand()`:** Sets environment variables in the child process:
  - `LD_PRELOAD=libchidori.so`
  - `CHIDORI_CFG_OVERRIDE` (JSON with id and server address)
- **`--on-connection-error=continue`:** Falls back to running without Chidori if the connection to the GPU server fails.

---

## Authentication

- **JWT validation is DISABLED by default.** Enable with `ENABLE_TOKEN_VALIDATION=true`.
- When enabled:
  - Auth0 RS256 with JWKS caching (5-minute refresh).
  - Validates JWTs on protected endpoints.
- **Agent-to-controller:** Static access token via `Bearer` auth header.
- **Permissions model:** Pools contain agents; users have per-pool permissions (`create_session`, `register_agent`, `admin`).

---

## Shared Packages (pkg/)

| Package | Purpose |
|---------|---------|
| `task/` | Goroutine lifecycle management with context cancellation |
| `server/` | gorilla/mux + CORS + Sentry + Zap middleware stack |
| `restapi/` | REST types, HTTP/HTTPS auto-negotiating client, helpers |
| `gpu/` | `GpuSet` with first-fit GPU selection, PCI address parsing |
| `middleware/jwt.go` | JWT validation middleware (disabled by default) |
| `crypto/` | Self-signed ECDSA P256 certificate generation |
| `logger/` | Zap wrapper with custom "juice" log format |
| `errors/` | Custom `Error` type with ID, Message, Cause chain |
| `net/` | Generic HTTP request/response helpers |
| `utilities/` | `ConcurrentMap`, `ConcurrentVariable`, `LinkedList`, `CommaValue` |
