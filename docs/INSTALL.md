# Installing ragkit

`ragkit` runs anywhere Python 3.10+ runs. Pick your OS:

| OS | One-liner |
|---|---|
| **Linux** | `bash scripts/setup-linux.sh` (apt/dnf/pacman/apk/zypper auto-detected) |
| **macOS** | `bash scripts/setup-macos.sh` (Homebrew) |
| **Windows** | `powershell -f scripts/setup-windows.ps1` (winget) |
| **Any (pip)** | `pip install cognis-ragkit` |
| **Docker** | `docker run --rm ghcr.io/cognis-digital/ragkit:latest --help` |
| **Devcontainer** | open in VS Code → "Reopen in Container" |

All ports of the tool (Python/JS/Go/Rust) live in `ports/`.
