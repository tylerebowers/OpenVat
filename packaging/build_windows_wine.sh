#!/usr/bin/env bash
# Build dist/OpenVat-<version>-setup.exe on Linux, through Wine - the same
# installer build_windows.ps1 makes on Windows.
#
# Needs: Wine 11 or newer with its 32-bit part (Inno Setup 7's 64-bit
# installer refuses to run on a system that cannot also run 32-bit programs:
# on Debian / Ubuntu that is the wine32:i386 package; Fedora's and Arch's wine
# have it), curl, tar and sha256sum.  Without a desktop session (CI, SSH)
# Inno Setup's installer is run under xvfb-run if that is installed.
#
# The Windows side is downloaded on first use and kept in a private Wine
# prefix under ~/.cache/openvat-build (set OPENVAT_BUILD_CACHE to move it;
# delete it to start over); your own ~/.wine is never touched:
#   Python        python-build-standalone's Windows build of CPython (a plain
#                 archive, so no Windows installer has to run under Wine)
#   Inno Setup 7  the 64-bit edition, installed in portable mode
# Every download is checked against its SHA-256.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_VERSION=${PYTHON_VERSION:-3.13.15}
PBS_RELEASE=${PBS_RELEASE:-20260924}       # github.com/astral-sh/python-build-standalone/releases
INNO_VERSION=${INNO_VERSION:-7.0.2}        # github.com/jrsoftware/issrc/releases
INNO_SHA256=${INNO_SHA256:-}
[ "$INNO_VERSION" = 7.0.2 ] && INNO_SHA256=${INNO_SHA256:-5ad54ca3def786f8f4212552e54cc6d8d61329e2d24a1cfee0571d42c2684ff1}

VERSION=$(sed -n 's/^__version__ = "\(.*\)"/\1/p' openvat/__init__.py)
CACHE=${OPENVAT_BUILD_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/openvat-build}
WINE=${WINE:-$(command -v wine || command -v wine64 || true)}
[ -n "$WINE" ] || { echo "wine not found - install it (e.g. apt install wine / dnf install wine)" >&2; exit 1; }
WINESERVER=${WINESERVER:-$(command -v wineserver || echo "$(dirname "$WINE")/wineserver")}
if [ "$("$WINE" --version | sed -n 's/^wine-\([0-9]*\).*/\1/p')" -lt 11 ] 2> /dev/null; then
    echo "$("$WINE" --version) is too old: Wine 11 or newer is needed (numpy calls C runtime" >&2
    echo "functions older Wines lack).  See https://gitlab.winehq.org/wine/wine/-/wikis/Download" >&2
    exit 1
fi

export WINEPREFIX="$CACHE/wine"
export WINEARCH=win64
export WINEDEBUG=${WINEDEBUG:--all}
export WINEDLLOVERRIDES="mscoree,mshtml="  # no Wine Mono / Gecko download prompts
export PYTHONUTF8=1
TOOLS="$WINEPREFIX/drive_c/openvat-build"   # = C:\openvat-build
PY='C:\openvat-build\python\python.exe'
ISCC='C:\openvat-build\inno-setup-7\ISCC.exe'

fetch() {   # fetch URL FILE SHA256 - download into $CACHE once, check the hash
    local url=$1 file="$CACHE/downloads/$2" sha=$3
    mkdir -p "$CACHE/downloads"
    if [ ! -f "$file" ]; then
        echo "   downloading $2"
        curl -fsSL --retry 3 -o "$file.part" "$url"
        mv "$file.part" "$file"
    fi
    if [ -n "$sha" ]; then
        echo "$sha  $file" | sha256sum -c --quiet - || { rm -f "$file"; echo "bad download: $2" >&2; exit 1; }
    else
        echo "   (no SHA-256 known for $2 - not checked; set INNO_SHA256 to check it)"
    fi
}

win() {     # run a Windows console program.  Its output goes through a pipe:
            # Python under Wine stops with "[WinError 6] Invalid handle" when its
            # output is a file (make exe-wine > build.log).
    "$WINE" "$@" < /dev/null 2>&1 | cat
}

gui() {     # run a Windows program that opens windows, on a virtual screen when there is no desktop
    if [ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] && command -v xvfb-run > /dev/null; then
        xvfb-run -a "$WINE" "$@"
    else
        "$WINE" "$@"
    fi
}

make_prefix() {
    rm -rf "$WINEPREFIX"
    mkdir -p "$WINEPREFIX"
    "$WINE" wineboot --init
    "$WINESERVER" --wait
}

echo "== Wine prefix ($WINEPREFIX)"
if [ ! -f "$WINEPREFIX/system.reg" ]; then
    make_prefix
elif [ ! -f "$WINEPREFIX/drive_c/windows/syswow64/kernel32.dll" ]; then
    make_prefix                                 # made before Wine's 32-bit part was installed
fi
if [ ! -f "$WINEPREFIX/drive_c/windows/syswow64/kernel32.dll" ]; then
    echo "This Wine cannot run 32-bit programs, and Inno Setup 7's installer needs that." >&2
    echo "Install Wine's 32-bit part - on Debian / Ubuntu:" >&2
    echo "    sudo dpkg --add-architecture i386 && sudo apt update && sudo apt install wine32:i386" >&2
    exit 1
fi

if [ ! -f "$TOOLS/python/.version-$PYTHON_VERSION-$PBS_RELEASE" ]; then
    echo "== Python $PYTHON_VERSION for Windows"
    PBS_FILE="cpython-$PYTHON_VERSION+$PBS_RELEASE-x86_64-pc-windows-msvc-install_only.tar.gz"
    PBS_URL="https://github.com/astral-sh/python-build-standalone/releases/download/$PBS_RELEASE"
    fetch "$PBS_URL/SHA256SUMS" "SHA256SUMS-$PBS_RELEASE" ""  > /dev/null
    PBS_SHA=$(awk -v f="$PBS_FILE" '$2 == f { print $1 }' "$CACHE/downloads/SHA256SUMS-$PBS_RELEASE")
    [ -n "$PBS_SHA" ] || { echo "$PBS_FILE is not in release $PBS_RELEASE" >&2; exit 1; }
    fetch "$PBS_URL/${PBS_FILE/+/%2B}" "$PBS_FILE" "$PBS_SHA"
    rm -rf "$TOOLS/python"
    mkdir -p "$TOOLS"
    tar -xzf "$CACHE/downloads/$PBS_FILE" -C "$TOOLS"          # unpacks python/
    touch "$TOOLS/python/.version-$PYTHON_VERSION-$PBS_RELEASE"
fi

if [ ! -f "$TOOLS/inno-setup-7/.version-$INNO_VERSION" ]; then
    echo "== Inno Setup $INNO_VERSION"
    INNO_FILE="innosetup-$INNO_VERSION-x64.exe"
    fetch "https://github.com/jrsoftware/issrc/releases/download/is-${INNO_VERSION//./_}/$INNO_FILE" "$INNO_FILE" "$INNO_SHA256"
    rm -rf "$TOOLS/inno-setup-7"
    gui "$CACHE/downloads/$INNO_FILE" /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /SP- /PORTABLE=1 \
        '/DIR=C:\openvat-build\inno-setup-7'
    [ -f "$TOOLS/inno-setup-7/ISCC.exe" ] || { echo "Inno Setup did not install" >&2; exit 1; }
    touch "$TOOLS/inno-setup-7/.version-$INNO_VERSION"
fi

echo "== PyInstaller (Windows)"
win "$PY" -m pip install -q --disable-pip-version-check -r requirements.txt pyinstaller
win "$PY" packaging/wine_stub_dlls.py        # Qt needs Windows' ICU to load; Wine has none
win "$PY" -m PyInstaller --noconfirm --clean --log-level WARN \
    --distpath build/windows/dist --workpath build/windows/work packaging/openvat.spec

# Wine's own DLLs and the ICU stand-ins only work inside Wine: none may be
# packaged, and Qt's Windows plugin must be.
if grep -rlaE "Wine (builtin|placeholder) DLL|OpenVat Wine stand-in" build/windows/dist/OpenVat \
        --include='*.dll' --include='*.pyd' --include='*.exe'; then
    echo "the files above only work inside Wine and were packaged by mistake" >&2
    exit 1
fi
if [ ! -f build/windows/dist/OpenVat/_internal/PySide6/plugins/platforms/qwindows.dll ]; then
    echo "Qt's Windows plugin (qwindows.dll) is missing from the build" >&2
    exit 1
fi

echo "== Inno Setup"
win "$ISCC" /Q "/DAppVersion=$VERSION" 'packaging\installer.iss'
"$WINESERVER" --wait
echo "built dist/OpenVat-$VERSION-setup.exe"
