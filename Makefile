# OpenVat build targets.  Run `make help` for the list.
#
# Nothing here requires installing OpenVat as a Python module: `make run`
# starts it from the source tree, and the packaged builds bundle Python.

PYTHON ?= python3

.PHONY: help deps run test appimage exe dmg clean

help:
	@echo "make deps      install Python dependencies (pip)"
	@echo "make run       run from source (python run.py)"
	@echo "make test      run the test suite"
	@echo "make appimage  Linux:   dist/OpenVat-<ver>-x86_64.AppImage"
	@echo "make exe       Windows: dist/OpenVat-<ver>-setup.exe   (run in PowerShell / Git Bash)"
	@echo "make dmg       macOS:   dist/OpenVat-<ver>.dmg"
	@echo "make clean     remove build artifacts"

deps:
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) run.py

test:
	$(PYTHON) -m pytest -q tests

appimage:
	bash packaging/build_appimage.sh

exe:
	powershell -ExecutionPolicy Bypass -File packaging/build_windows.ps1

dmg:
	bash packaging/build_macos.sh

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
