#!/usr/bin/env bash
# Build dist/OpenVat-<version>-<arch>.AppImage on Linux.
# Needs: python3 (3.10+) and pip; appimagetool is used from PATH or downloaded
# once into ~/.cache/openvat-build (set OPENVAT_BUILD_CACHE to move it).  No
# FUSE needed (works in containers and CI).
# The AppImage runs on distributions at least as new as the one it is built on
# (glibc), so build on the oldest you support - the GitHub workflow uses
# Ubuntu 22.04.
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION=$(sed -n 's/^__version__ = "\(.*\)"/\1/p' openvat/__init__.py)
ARCH=$(uname -m)
CACHE=${OPENVAT_BUILD_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/openvat-build}
export APPIMAGE_EXTRACT_AND_RUN=1          # run appimagetool without FUSE

echo "== PyInstaller"
python3 -m pip install -q -r requirements.txt pyinstaller
python3 -m PyInstaller --noconfirm --clean --log-level WARN \
    --distpath build/linux/dist --workpath build/linux/work packaging/openvat.spec

echo "== AppDir"
APPDIR=build/linux/OpenVat.AppDir
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications" "$APPDIR/usr/share/icons/hicolor/256x256/apps"
cp -r build/linux/dist/OpenVat/* "$APPDIR/usr/bin/"
cp openvat.desktop "$APPDIR/openvat.desktop"
cp openvat.desktop "$APPDIR/usr/share/applications/"
sed -i 's/^Icon=.*/Icon=openvat/' "$APPDIR/openvat.desktop" "$APPDIR/usr/share/applications/openvat.desktop"
cp packaging/openvat.png "$APPDIR/openvat.png"
cp "$APPDIR/openvat.png" "$APPDIR/usr/share/icons/hicolor/256x256/apps/openvat.png"
# Qt on X11 (XWayland on Wayland desktops), native Wayland when there is no X
# server; QT_QPA_PLATFORM set by the user wins.  PyOpenGL is made to follow
# whichever Qt ends up using (openvat/ui/glplatform.py).
cat > "$APPDIR/AppRun" <<'EOF'
#!/usr/bin/env bash
HERE="$(dirname "$(readlink -f "$0")")"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb;wayland}"
exec "$HERE/usr/bin/OpenVat" "$@"
EOF
chmod +x "$APPDIR/AppRun"

echo "== appimagetool"
TOOL=$(command -v appimagetool || true)
if [ -z "$TOOL" ]; then
    TOOL="$CACHE/downloads/appimagetool-$ARCH.AppImage"
    if [ ! -x "$TOOL" ]; then
        mkdir -p "$(dirname "$TOOL")"
        curl -fsSL --retry 3 -o "$TOOL.part" \
            "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-$ARCH.AppImage"
        chmod +x "$TOOL.part"
        mv "$TOOL.part" "$TOOL"
    fi
fi
mkdir -p dist
export ARCH                                 # appimagetool reads it
"$TOOL" --no-appstream "$APPDIR" "dist/OpenVat-$VERSION-$ARCH.AppImage"
echo "built dist/OpenVat-$VERSION-$ARCH.AppImage"
