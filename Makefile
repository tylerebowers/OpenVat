# OpenVat build targets.  Run `make help` for the list.
#
# Nothing here requires installing OpenVat as a Python module: `make run`
# starts it from the source tree, and the packaged builds bundle Python.

PYTHON ?= python3
OPENVAT_BUILD_CACHE ?= $(or $(XDG_CACHE_HOME),$(HOME)/.cache)/openvat-build
export OPENVAT_BUILD_CACHE

# `make dist`: every package this machine can build
UNAME := $(shell uname -s 2>/dev/null)
ifeq ($(UNAME),Linux)
DIST_TARGETS := appimage exe-wine
else ifeq ($(UNAME),Darwin)
DIST_TARGETS := dmg
else
DIST_TARGETS := exe
endif

.PHONY: help deps run test dist appimage exe exe-wine dmg clean clean-cache

help:
	@echo "make deps         install Python dependencies (pip)"
	@echo "make run          run from source (python run.py)"
	@echo "make test         run the test suite"
	@echo "make dist         every package this machine can build ($(DIST_TARGETS))"
	@echo "make appimage     Linux:           dist/OpenVat-<ver>-<arch>.AppImage"
	@echo "make exe-wine     Linux, Wine 11+: dist/OpenVat-<ver>-setup.exe"
	@echo "make exe          Windows:         dist/OpenVat-<ver>-setup.exe   (PowerShell / Git Bash)"
	@echo "make dmg          macOS:           dist/OpenVat-<ver>-<arch>.dmg"
	@echo "make clean        remove build/ and dist/"
	@echo "make clean-cache  remove downloaded build tools and the Wine prefix ($(OPENVAT_BUILD_CACHE))"
	@echo "All of them at once, for every platform: the GitHub workflow (.github/workflows/build.yml)."

deps:
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) run.py

test:
	$(PYTHON) -m pytest -q tests

dist: $(DIST_TARGETS)

appimage:
	bash packaging/build_appimage.sh

exe-wine:
	bash packaging/build_windows_wine.sh

exe:
	powershell -ExecutionPolicy Bypass -File packaging/build_windows.ps1

dmg:
	bash packaging/build_macos.sh

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

clean-cache:
	rm -rf "$(OPENVAT_BUILD_CACHE)"
