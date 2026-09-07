#!/bin/sh
# Install or update ailocal.
#   curl -fsSL https://zottiben.github.io/ai-local/install.sh | sh
# Once installed, `ailocal update` re-runs this same script for you.
#
# Options (pass them through `ailocal update` too, e.g. `ailocal update --check`):
#   --check           report whether an update is available, install nothing
#   --force           reinstall even when the latest version is already installed
#   --version <tag>   install a specific release tag (e.g. v0.1.0) instead of the latest
#   -h, --help        show this help
#
# Linux (x86_64, aarch64) and macOS (universal). Windows is out of scope: llama.cpp's
# GPU backends and the service managers this drives are Unix-only here.
set -eu

REPO="zottiben/ai-local"
# Records the installed tag so a later run can answer "already up to date" without
# downloading anything. `ailocal update` also passes AILOCAL_CURRENT_VERSION, which is
# authoritative (it is the running binary's own version) and wins over the receipt.
RECEIPT="${XDG_DATA_HOME:-$HOME/.local/share}/ailocal/version"

usage() {
  cat <<'EOF'
Install or update ailocal.
  curl -fsSL https://zottiben.github.io/ai-local/install.sh | sh
Once installed, `ailocal update` re-runs this same script for you.

Options:
  --check           report whether an update is available, install nothing
  --force           reinstall even when the latest version is already installed
  --version <tag>   install a specific release tag (e.g. v0.1.0) instead of the latest
  -h, --help        show this help
EOF
}

# Is version $1 strictly newer than $2? Both are plain numeric x.y.z, the only shape
# ailocal's release tags take.
newer_than() {
  [ "$1" != "$2" ] || return 1
  [ "$(printf '%s\n%s\n' "$1" "$2" | sort -t. -k1,1n -k2,2n -k3,3n | tail -1)" = "$1" ]
}

CHECK_ONLY=0
FORCE=0
VERSION=""
while [ $# -gt 0 ]; do
  case "$1" in
    --check) CHECK_ONLY=1 ;;
    --force) FORCE=1 ;;
    --version)
      shift
      [ $# -gt 0 ] || { echo "--version needs a release tag, e.g. --version v0.1.0" >&2; exit 2; }
      VERSION="$1"
      ;;
    --version=*) VERSION="${1#--version=}" ;;
    -h | --help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"

case "$OS" in
  linux)
    case "$ARCH" in
      x86_64 | amd64) PLATFORM="linux-x86_64" ;;
      arm64 | aarch64) PLATFORM="linux-aarch64" ;;
      *) echo "Unsupported architecture: $ARCH" >&2; exit 1 ;;
    esac
    ;;
  darwin)
    # One universal binary rather than per-arch: it is a few MB more and removes a
    # whole class of "wrong build for this Mac" reports.
    PLATFORM="macos-universal"
    ;;
  *) echo "Unsupported OS: $OS (ailocal targets Linux and macOS)" >&2; exit 1 ;;
esac

# Pick a bin dir on PATH without needing sudo when possible.
if echo "$PATH" | tr ':' '\n' | grep -qx "$HOME/.local/bin"; then
  BIN_DIR="$HOME/.local/bin"
else
  BIN_DIR="/usr/local/bin"
fi

# --- what is installed, and what is the latest -----------------------------------
CURRENT="${AILOCAL_CURRENT_VERSION:-}"
if [ -z "$CURRENT" ] && [ -r "$RECEIPT" ]; then
  CURRENT="$(cat "$RECEIPT")"
fi
CURRENT="${CURRENT#v}"

if [ -z "$VERSION" ]; then
  VERSION="$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
    | grep '"tag_name"' | head -1 | sed -E 's/.*"([^"]+)".*/\1/')"
  if [ -z "$VERSION" ]; then
    echo "Could not determine the latest release. Is one published yet?" >&2
    exit 1
  fi
fi
VERSION_NUM="${VERSION#v}"
BASE="https://github.com/${REPO}/releases/download/${VERSION}"

if [ -n "$CURRENT" ] && [ "$CURRENT" = "$VERSION_NUM" ]; then
  if [ "$CHECK_ONLY" -eq 1 ]; then
    echo "ailocal v${CURRENT} is up to date."
    exit 0
  fi
  if [ "$FORCE" -eq 0 ]; then
    echo "ailocal v${CURRENT} is already installed (use --force to reinstall)."
    exit 0
  fi
elif [ -n "$CURRENT" ] && newer_than "$CURRENT" "$VERSION_NUM"; then
  # A development build, or a hand-picked tag: report it rather than quietly downgrade.
  echo "ailocal v${CURRENT} is newer than ${VERSION}."
  if [ "$CHECK_ONLY" -eq 1 ]; then
    exit 0
  fi
  if [ "$FORCE" -eq 0 ]; then
    echo "Nothing to do (use --force to install ${VERSION} anyway)."
    exit 0
  fi
elif [ "$CHECK_ONLY" -eq 1 ]; then
  if [ -n "$CURRENT" ]; then
    echo "Update available: v${CURRENT} -> ${VERSION}"
  else
    echo "ailocal ${VERSION} is available."
  fi
  echo "Run 'ailocal update' to install it."
  exit 0
fi

# --- asset name (must match .github/workflows/release.yml) -----------------------
FILENAME="ailocal-v${VERSION_NUM}-${PLATFORM}.tar.gz"

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

echo "Downloading ailocal ${VERSION} for ${PLATFORM}..."
curl -fsSL "${BASE}/${FILENAME}" -o "${TMPDIR}/${FILENAME}"

# --- verify checksum (best-effort: only if published and a hasher is available) --
if curl -fsSL "${BASE}/checksums.txt" -o "${TMPDIR}/checksums.txt" 2>/dev/null; then
  expected="$(grep " ${FILENAME}\$" "${TMPDIR}/checksums.txt" | awk '{print $1}')"
  if [ -n "$expected" ]; then
    if command -v sha256sum >/dev/null 2>&1; then
      actual="$(sha256sum "${TMPDIR}/${FILENAME}" | awk '{print $1}')"
    elif command -v shasum >/dev/null 2>&1; then
      actual="$(shasum -a 256 "${TMPDIR}/${FILENAME}" | awk '{print $1}')"
    else
      actual=""
    fi
    if [ -n "$actual" ] && [ "$actual" != "$expected" ]; then
      echo "Checksum mismatch for ${FILENAME}" >&2
      exit 1
    fi
  fi
fi

tar xzf "${TMPDIR}/${FILENAME}" -C "$TMPDIR"

# Staged beside the target and renamed into place: a rename over a running executable is
# atomic and succeeds while the gateway is serving, whereas writing onto it fails
# (ETXTBSY).
staged="${BIN_DIR}/.ailocal.new.$$"
if mkdir -p "$BIN_DIR" 2>/dev/null && [ -w "$BIN_DIR" ]; then
  cp "${TMPDIR}/ailocal" "$staged"
  chmod +x "$staged"
  mv -f "$staged" "$BIN_DIR/ailocal"
else
  echo "Installing to ${BIN_DIR} (requires sudo)..."
  sudo mkdir -p "$BIN_DIR"
  sudo cp "${TMPDIR}/ailocal" "$staged"
  sudo chmod +x "$staged"
  sudo mv -f "$staged" "$BIN_DIR/ailocal"
fi

if [ "$OS" = "darwin" ]; then
  # curl downloads carry no quarantine attribute, but clear it in case of a re-host.
  xattr -d com.apple.quarantine "$BIN_DIR/ailocal" 2>/dev/null || true
fi

# Best-effort: the receipt only speeds up the next run's up-to-date check.
if mkdir -p "$(dirname "$RECEIPT")" 2>/dev/null; then
  printf '%s\n' "$VERSION_NUM" > "$RECEIPT" 2>/dev/null || true
fi

echo "Installed ailocal ${VERSION} to ${BIN_DIR}/ailocal"

if [ -n "$CURRENT" ] && [ "$CURRENT" != "$VERSION_NUM" ]; then
  echo "Updated from v${CURRENT}."
  # Service definitions embed the binary path, and a running gateway keeps the old
  # image mapped, so it has to be restarted to pick up the new one.
  if command -v systemctl >/dev/null 2>&1 &&
     systemctl --user is-active ailocal-gateway.service >/dev/null 2>&1; then
    echo "Restarting the gateway to pick it up..."
    systemctl --user restart ailocal-gateway.service || true
  elif command -v launchctl >/dev/null 2>&1 &&
       launchctl print "gui/$(id -u)/io.github.zottiben.ailocal.gateway" >/dev/null 2>&1; then
    echo "Restarting the gateway to pick it up..."
    launchctl kickstart -k "gui/$(id -u)/io.github.zottiben.ailocal.gateway" || true
  fi
fi

if ! echo "$PATH" | tr ':' '\n' | grep -qx "$BIN_DIR"; then
  echo
  echo "Note: ${BIN_DIR} is not on your PATH. Add it, e.g.:"
  echo "  export PATH=\"${BIN_DIR}:\$PATH\""
fi

if [ -z "$CURRENT" ]; then
  echo
  echo "Next: run 'ailocal setup' to check prerequisites, install a model,"
  echo "start the services and point your harnesses at them."
fi
