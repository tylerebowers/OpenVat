#!/usr/bin/env bash
# Build dist/OpenVat-<version>-<arch>.dmg on macOS (arm64 on Apple Silicon,
# x86_64 on Intel - each Mac builds for its own kind; the GitHub workflow
# builds both).
# Needs: python3 (3.10+, e.g. from python.org or Homebrew); hdiutil ships
# with macOS.  macOS apps can only be built on macOS: PyInstaller does not
# cross-compile.
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION=$(sed -n 's/^__version__ = "\(.*\)"/\1/p' openvat/__init__.py)
ARCH=$(uname -m)

echo "== PyInstaller"
python3 -m pip install -q -r requirements.txt pyinstaller
python3 -m PyInstaller --noconfirm --clean --log-level WARN \
    --distpath build/macos/dist --workpath build/macos/work packaging/openvat.spec

echo "== DMG"
STAGING=build/macos/dmg
rm -rf "$STAGING"
mkdir -p "$STAGING"
cp -R build/macos/dist/OpenVat.app "$STAGING/"
ln -s /Applications "$STAGING/Applications"
mkdir -p dist
DMG="dist/OpenVat-$VERSION-$ARCH.dmg"
rm -f "$DMG"
for try in 1 2 3 4 5; do            # hdiutil now and then fails with "Resource busy"
    if hdiutil create -volname "OpenVat" -srcfolder "$STAGING" -ov -format ULFO "$DMG"; then
        break
    fi
    [ "$try" = 5 ] && exit 1
    sleep 5
done
echo "built $DMG"
echo "(not notarized: on first start macOS refuses it - open System Settings > Privacy & Security"
echo " and click \"Open Anyway\", or run: xattr -dr com.apple.quarantine /Applications/OpenVat.app)"
