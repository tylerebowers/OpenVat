#!/usr/bin/env bash
# Build dist/OpenVat-<version>.dmg on macOS.
# Needs: python3 (3.10+, e.g. from python.org or Homebrew) and Xcode command
# line tools (for iconutil / hdiutil, both ship with macOS).
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION=$(python3 -c "import openvat; print(openvat.__version__)")

echo "== PyInstaller"
python3 -m pip install -q -r requirements.txt pyinstaller
python3 packaging/make_icon.py packaging/openvat.icns
python3 -m PyInstaller --noconfirm --clean packaging/openvat.spec

echo "== DMG"
STAGING=build/dmg
rm -rf "$STAGING"
mkdir -p "$STAGING"
cp -R dist/OpenVat.app "$STAGING/"
ln -s /Applications "$STAGING/Applications"
mkdir -p dist
rm -f "dist/OpenVat-$VERSION.dmg"
hdiutil create -volname "OpenVat" -srcfolder "$STAGING" -ov -format UDZO "dist/OpenVat-$VERSION.dmg"
echo "built dist/OpenVat-$VERSION.dmg"
echo "(unsigned: users must right-click > Open the first time, or sign it with codesign/notarytool)"
