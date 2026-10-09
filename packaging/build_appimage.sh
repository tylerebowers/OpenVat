#!/usr/bin/env bash
# Build dist/OpenVat-<version>-x86_64.AppImage on Linux.
# Needs: python3 (3.10+), pip, and either `appimagetool` on PATH or network
# access (it is downloaded to build/ automatically).
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION=$(python3 -c "import openvat; print(openvat.__version__)")
ARCH=$(uname -m)

echo "== PyInstaller"
python3 -m pip install -q -r requirements.txt pyinstaller
python3 -m PyInstaller --noconfirm --clean packaging/openvat.spec

echo "== AppDir"
APPDIR=build/OpenVat.AppDir
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications" "$APPDIR/usr/share/icons/hicolor/256x256/apps"
cp -r dist/OpenVat/* "$APPDIR/usr/bin/"
cp openvat.desktop "$APPDIR/openvat.desktop"
cp openvat.desktop "$APPDIR/usr/share/applications/"
sed -i 's/^Icon=.*/Icon=openvat/' "$APPDIR/openvat.desktop" "$APPDIR/usr/share/applications/openvat.desktop"
if [ -f packaging/openvat.png ]; then
    cp packaging/openvat.png "$APPDIR/openvat.png"
    cp packaging/openvat.png "$APPDIR/usr/share/icons/hicolor/256x256/apps/openvat.png"
else
    python3 packaging/make_icon.py "$APPDIR/openvat.png"
    cp "$APPDIR/openvat.png" "$APPDIR/usr/share/icons/hicolor/256x256/apps/openvat.png"
fi
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
    TOOL=build/appimagetool-$ARCH.AppImage
    if [ ! -x "$TOOL" ]; then
        curl -L -o "$TOOL" "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-$ARCH.AppImage"
        chmod +x "$TOOL"
    fi
fi
mkdir -p dist
ARCH=$ARCH "$TOOL" --no-appstream "$APPDIR" "dist/OpenVat-$VERSION-$ARCH.AppImage"
echo "built dist/OpenVat-$VERSION-$ARCH.AppImage"
