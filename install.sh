#!/bin/sh
# Install or update ailocal.
#   curl -fsSL https://zottiben.github.io/ai-local/install.sh | sh
# Once installed, `ailocal update` re-runs this same script for you.
#
# Options (pass them through `ailocal update` too, e.g. `ailocal update --check`):
#   --check           report whether an update is available, install nothing
#   --force           reinstall even when the latest version is already installed
#   --version <tag>   install a specific release tag (e.g. v0.1.0) instead of the latest
#   --extra <name>    also install an optional companion, e.g. --extra eval (repeatable)
#   --extras-only     install only the named extras, leaving ailocal itself alone
#   --app             also install the desktop app (the console in its own window)
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
  --extra <name>    also install an optional companion, e.g. --extra eval (repeatable)
  --extras-only     install only the named extras, leaving ailocal itself alone
  --app             also install the desktop app (the console in its own window)
  -h, --help        show this help

Extras are separate binaries released from the same tag, so that the core stays small
for the machines that only serve models. `ailocal update` refreshes whichever ones are
already installed, keeping their version in step with ailocal's.

The desktop app is not one of them: it is an application, not a subcommand, so it has
its own flag. It is published for Apple Silicon and for Linux x86_64. `ailocal update`
refreshes it too once it is installed.
EOF
}

# Is version $1 strictly newer than $2? Both are plain numeric x.y.z, the only shape
# ailocal's release tags take.
newer_than() {
  [ "$1" != "$2" ] || return 1
  [ "$(printf '%s\n%s\n' "$1" "$2" | sort -t. -k1,1n -k2,2n -k3,3n | tail -1)" = "$1" ]
}

# Extras this script knows how to install. An unknown name is rejected up front rather
# than turning into a confusing 404 halfway through.
KNOWN_EXTRAS="eval ui"

CHECK_ONLY=0
FORCE=0
VERSION=""
EXTRAS=""
EXTRAS_ONLY=0
APP=0

add_extra() {
  for known in $KNOWN_EXTRAS; do
    if [ "$known" = "$1" ]; then
      # Space-separated list, deduplicated: `ailocal update` appends every installed
      # extra, and a caller may have named one too.
      for already in $EXTRAS; do
        [ "$already" = "$1" ] && return 0
      done
      EXTRAS="${EXTRAS} $1"
      return 0
    fi
  done
  echo "Unknown extra: $1 (known: ${KNOWN_EXTRAS})" >&2
  # The likeliest wrong guess, because the app is released from the same tag and
  # installed by the same script - it is just not a subcommand.
  case "$1" in
    desktop | app | ui-app) echo "For the desktop app, use --app instead." >&2 ;;
  esac
  exit 2
}

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
    --extra)
      shift
      [ $# -gt 0 ] || { echo "--extra needs a name, e.g. --extra eval" >&2; exit 2; }
      add_extra "$1"
      ;;
    --extra=*) add_extra "${1#--extra=}" ;;
    --extras-only) EXTRAS_ONLY=1 ;;
    --app) APP=1 ;;
    -h | --help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [ "$EXTRAS_ONLY" -eq 1 ] && [ -z "$EXTRAS" ]; then
  echo "--extras-only needs at least one --extra <name>" >&2
  exit 2
fi

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

# The desktop app is built per architecture rather than universal, because it is opened
# once on the machine it was downloaded to. The release publishes two: Apple Silicon and
# Linux x86_64. An empty label means there is none for this machine.
case "$OS" in
  linux) case "$ARCH" in x86_64 | amd64) APP_PLATFORM="linux-x86_64" ;; *) APP_PLATFORM="" ;; esac ;;
  darwin) case "$ARCH" in arm64) APP_PLATFORM="macos-arm64" ;; *) APP_PLATFORM="" ;; esac ;;
  *) APP_PLATFORM="" ;;
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

# Whether the core binary itself needs replacing. Extras are handled separately below,
# so "ailocal is already current" must not skip installing a newly requested one.
NEED_CORE=1
[ "$EXTRAS_ONLY" -eq 1 ] && NEED_CORE=0

if [ -n "$CURRENT" ] && [ "$CURRENT" = "$VERSION_NUM" ]; then
  if [ "$CHECK_ONLY" -eq 1 ]; then
    echo "ailocal v${CURRENT} is up to date."
    exit 0
  fi
  if [ "$FORCE" -eq 0 ]; then
    NEED_CORE=0
    [ -z "$EXTRAS" ] && [ "$APP" -eq 0 ] && { echo "ailocal v${CURRENT} is already installed (use --force to reinstall)."; exit 0; }
  fi
elif [ -n "$CURRENT" ] && newer_than "$CURRENT" "$VERSION_NUM"; then
  # A development build, or a hand-picked tag: report it rather than quietly downgrade.
  echo "ailocal v${CURRENT} is newer than ${VERSION}."
  if [ "$CHECK_ONLY" -eq 1 ]; then
    exit 0
  fi
  if [ "$FORCE" -eq 0 ]; then
    NEED_CORE=0
    [ -z "$EXTRAS" ] && [ "$APP" -eq 0 ] && { echo "Nothing to do (use --force to install ${VERSION} anyway)."; exit 0; }
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

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

# --- fetching a binary ------------------------------------------------------------
# One routine for the core and for every extra, because they are released from the same
# tag as identically shaped assets: <binary>-v<version>-<platform>.tar.gz containing a
# single executable named <binary>. Asset names must match .github/workflows/release.yml.

# Best-effort checksum: only when one is published and a hasher is available.
verify() {
  filename="$1"
  [ -r "${TMPDIR}/checksums.txt" ] || return 0
  expected="$(grep " ${filename}\$" "${TMPDIR}/checksums.txt" | awk '{print $1}')"
  [ -n "$expected" ] || return 0
  if command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum "${TMPDIR}/${filename}" | awk '{print $1}')"
  elif command -v shasum >/dev/null 2>&1; then
    actual="$(shasum -a 256 "${TMPDIR}/${filename}" | awk '{print $1}')"
  else
    return 0
  fi
  if [ "$actual" != "$expected" ]; then
    echo "Checksum mismatch for ${filename}" >&2
    exit 1
  fi
}

install_binary() {
  binary="$1"
  filename="${binary}-v${VERSION_NUM}-${PLATFORM}.tar.gz"

  echo "Downloading ${binary} ${VERSION} for ${PLATFORM}..."
  if ! curl -fsSL "${BASE}/${filename}" -o "${TMPDIR}/${filename}"; then
    echo "Could not fetch ${BASE}/${filename}" >&2
    if [ "$binary" != "ailocal" ]; then
      # Much the likeliest cause: extras were added after that tag was cut.
      echo "Release ${VERSION} may predate the ${binary#ailocal-} extra." >&2
    fi
    exit 1
  fi
  verify "$filename"
  tar xzf "${TMPDIR}/${filename}" -C "$TMPDIR"

  # Staged beside the target and renamed into place: a rename over a running executable
  # is atomic and succeeds while the gateway is serving, whereas writing onto it fails
  # (ETXTBSY).
  staged="${BIN_DIR}/.${binary}.new.$$"
  if mkdir -p "$BIN_DIR" 2>/dev/null && [ -w "$BIN_DIR" ]; then
    cp "${TMPDIR}/${binary}" "$staged"
    chmod +x "$staged"
    mv -f "$staged" "${BIN_DIR}/${binary}"
  else
    echo "Installing to ${BIN_DIR} (requires sudo)..."
    sudo mkdir -p "$BIN_DIR"
    sudo cp "${TMPDIR}/${binary}" "$staged"
    sudo chmod +x "$staged"
    sudo mv -f "$staged" "${BIN_DIR}/${binary}"
  fi

  if [ "$OS" = "darwin" ]; then
    # curl downloads carry no quarantine attribute, but clear it in case of a re-host.
    xattr -d com.apple.quarantine "${BIN_DIR}/${binary}" 2>/dev/null || true
  fi
  echo "Installed ${binary} ${VERSION} to ${BIN_DIR}/${binary}"
}

# --- the desktop app -------------------------------------------------------------
# Not a binary on PATH and not a subcommand, so it cannot go through install_binary:
# macOS ships an .app inside a .dmg, Linux an AppImage.

install_app() {
  if [ -z "$APP_PLATFORM" ]; then
    echo "No desktop app is published for ${OS} ${ARCH}." >&2
    echo "The same console runs in a browser: ailocal extras install ui, then ailocal ui" >&2
    exit 1
  fi

  if [ "$OS" = "darwin" ]; then
    filename="ai-local-desktop-v${VERSION_NUM}-${APP_PLATFORM}.dmg"
  else
    filename="ai-local-desktop-v${VERSION_NUM}-${APP_PLATFORM}.AppImage"
  fi

  echo "Downloading the ai-local app ${VERSION} for ${APP_PLATFORM}..."
  if ! curl -fsSL "${BASE}/${filename}" -o "${TMPDIR}/${filename}"; then
    echo "Could not fetch ${BASE}/${filename}" >&2
    echo "Release ${VERSION} may predate the desktop app." >&2
    exit 1
  fi
  verify "$filename"

  if [ "$OS" = "darwin" ]; then
    # /Applications is group-writable by admin, so this usually needs no sudo.
    APP_DIR="/Applications"
    [ -w "$APP_DIR" ] || APP_DIR="${HOME}/Applications"
    mkdir -p "$APP_DIR"

    mount="${TMPDIR}/mnt"
    mkdir -p "$mount"
    hdiutil attach -nobrowse -readonly -mountpoint "$mount" "${TMPDIR}/${filename}" -quiet
    # Replaced rather than merged: a file left behind inside an old bundle would
    # outlive the update. A running copy keeps its own inode and is unaffected until
    # it is next launched.
    rm -rf "${APP_DIR}/ai-local.app"
    cp -R "${mount}/ai-local.app" "${APP_DIR}/ai-local.app"
    hdiutil detach "$mount" -quiet

    # The app is ad-hoc signed, so Gatekeeper refuses a quarantined copy outright
    # rather than offering the right-click-Open escape. curl sets no quarantine
    # attribute, but clear it in case the .dmg was re-hosted by something that does.
    xattr -dr com.apple.quarantine "${APP_DIR}/ai-local.app" 2>/dev/null || true
    echo "Installed the ai-local app ${VERSION} to ${APP_DIR}/ai-local.app"
  else
    # Staged and renamed, as for a binary: a rename over a running AppImage is atomic.
    staged="${BIN_DIR}/.ai-local-desktop.new.$$"
    if mkdir -p "$BIN_DIR" 2>/dev/null && [ -w "$BIN_DIR" ]; then
      cp "${TMPDIR}/${filename}" "$staged"
      chmod +x "$staged"
      mv -f "$staged" "${BIN_DIR}/ai-local-desktop"
    else
      echo "Installing to ${BIN_DIR} (requires sudo)..."
      sudo mkdir -p "$BIN_DIR"
      sudo cp "${TMPDIR}/${filename}" "$staged"
      sudo chmod +x "$staged"
      sudo mv -f "$staged" "${BIN_DIR}/ai-local-desktop"
    fi
    echo "Installed the ai-local app ${VERSION} to ${BIN_DIR}/ai-local-desktop"

    # A launcher entry, so it is an application rather than a command you have to
    # remember. Written by hand rather than extracted from the AppImage: extraction
    # needs the AppImage runtime to cooperate, and a desktop file is eight lines.
    apps="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
    if mkdir -p "$apps" 2>/dev/null; then
      cat > "${apps}/ai-local.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=ai-local
Comment=Manage local models, services and harnesses
Exec=${BIN_DIR}/ai-local-desktop
Icon=ai-local
Terminal=false
Categories=Development;
DESKTOP
      command -v update-desktop-database >/dev/null 2>&1 &&
        update-desktop-database "$apps" >/dev/null 2>&1 || true
      echo "Added a launcher entry at ${apps}/ai-local.desktop"
    fi
  fi
}

# Fetched once and reused for every asset.
curl -fsSL "${BASE}/checksums.txt" -o "${TMPDIR}/checksums.txt" 2>/dev/null || true

if [ "$NEED_CORE" -eq 1 ]; then
  install_binary ailocal

  # Best-effort: the receipt only speeds up the next run's up-to-date check.
  if mkdir -p "$(dirname "$RECEIPT")" 2>/dev/null; then
    printf '%s\n' "$VERSION_NUM" > "$RECEIPT" 2>/dev/null || true
  fi
fi

for extra in $EXTRAS; do
  install_binary "ailocal-${extra}"
done

if [ "$APP" -eq 1 ]; then
  install_app
fi

if [ "$NEED_CORE" -eq 1 ] && [ -n "$CURRENT" ] && [ "$CURRENT" != "$VERSION_NUM" ]; then
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

if [ "$NEED_CORE" -eq 1 ] && [ -z "$CURRENT" ]; then
  echo
  echo "Next: run 'ailocal setup' to check prerequisites, install a model,"
  echo "start the services and point your harnesses at them."
fi

for extra in $EXTRAS; do
  echo "Try: ailocal ${extra} --help"
done

if [ "$APP" -eq 1 ]; then
  if [ "$OS" = "darwin" ]; then
    echo "Open it from Launchpad, or: open -a ai-local"
  else
    echo "Open it from your launcher, or: ai-local-desktop"
  fi
fi
