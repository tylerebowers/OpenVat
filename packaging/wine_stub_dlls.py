"""Give a Wine prefix the Windows system DLLs Qt links to but Wine lacks.

Run with the Windows Python inside Wine, after PySide6 and PyInstaller are
installed (build_windows_wine.sh does this):

    wine python.exe packaging/wine_stub_dlls.py

Qt6Core.dll links Windows' own ICU (icuuc.dll, in System32 since Windows 10
1703).  Wine has no ICU, so QtCore cannot even be imported - and PyInstaller,
which imports it to find Qt's plugins, would build an OpenVat without them.
This writes a stand-in for each missing ICU DLL into the prefix's System32,
exporting exactly the functions Qt imports, each returning 0.  Only
PyInstaller's look-up runs against it: it is never packaged (PyInstaller
leaves out DLLs from the Windows folder, and build_windows_wine.sh checks),
and on Windows Qt uses the real ICU.
"""

from __future__ import annotations

import importlib.util
import os
import struct
import sys
from pathlib import Path

import pefile                      # installed with PyInstaller on Windows

MARKER = b"OpenVat Wine stand-in for a Windows system DLL"
SYSTEM_ONLY = {"icuuc.dll", "icuin.dll", "icu.dll"}     # Windows ships them, Wine does not


def _align(n: int, a: int) -> int:
    return (n + a - 1) // a * a


def stub_dll(dll_name: str, names: list[str]) -> bytes:
    """A 64-bit DLL exporting ``names``, every one ``return 0``."""
    names = sorted(set(names))      # the loader binary-searches the name table
    n = len(names)
    rva, file_align, sect_align = 0x1000, 0x200, 0x1000

    body = bytearray(bytes.fromhex("b801000000c3"))     # DllMain: mov eax, 1; ret
    ret0 = len(body)
    body += bytes.fromhex("31c0c3")                     # xor eax, eax; ret
    body += b"\0" * (-len(body) % 8)
    exp = len(body)
    eat, npt = exp + 40, exp + 40 + 4 * n
    ords, strs = npt + 4 * n, npt + 6 * n
    strings, name_rvas = bytearray(), []
    dll_name_rva = rva + strs
    strings += dll_name.encode() + b"\0"
    for name in names:
        name_rvas.append(rva + strs + len(strings))
        strings += name.encode() + b"\0"
    strings += MARKER + b"\0"
    body += struct.pack("<IIHHIIIIIII", 0, 0, 0, 0, dll_name_rva, 1, n, n,
                        rva + eat, rva + npt, rva + ords)
    body += struct.pack(f"<{n}I", *[rva + ret0] * n)
    body += struct.pack(f"<{n}I", *name_rvas)
    body += struct.pack(f"<{n}H", *range(n))
    body += strings
    exp_size = len(body) - exp
    raw_size = _align(len(body), file_align)

    dos = bytearray(64)
    dos[:2] = b"MZ"
    struct.pack_into("<I", dos, 0x3C, len(dos))
    coff = struct.pack("<HHIIIHH", 0x8664, 1, 0, 0, 0, 240, 0x2022)   # x64; executable, large-address, DLL
    opt = struct.pack("<HBBIIIIIQIIHHHHHHIIIIHHQQQQII",
                      0x20B, 14, 0, raw_size, 0, 0, rva, rva,           # PE32+, entry = DllMain
                      0x180000000, sect_align, file_align,
                      6, 0, 0, 0, 6, 0, 0,
                      rva + _align(len(body), sect_align), file_align, 0,
                      2, 0x100,                                         # GUI subsystem, NX compatible
                      0x100000, 0x1000, 0x100000, 0x1000, 0, 16)
    dirs = struct.pack("<II", rva + exp, exp_size) + bytes(15 * 8)    # only the export table
    section = struct.pack("<8sIIIIIIHHI", b".text", len(body), rva, raw_size, file_align,
                          0, 0, 0, 0, 0x60000020)                       # code, execute, read
    headers = bytes(dos) + b"PE\0\0" + coff + opt + dirs + section
    return headers.ljust(file_align, b"\0") + bytes(body).ljust(raw_size, b"\0")


def imported_names(folders: list[Path]) -> dict[str, set[str]]:
    """{system-only DLL: names imported from it} over the DLLs in ``folders``."""
    wanted: dict[str, set[str]] = {}
    for folder in folders:
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() not in (".dll", ".pyd"):
                continue
            pe = pefile.PE(str(path), fast_load=True)
            pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
            for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []):
                dll = entry.dll.decode().lower()
                if dll in SYSTEM_ONLY:
                    wanted.setdefault(dll, set()).update(i.name.decode() for i in entry.imports if i.name)
            pe.close()
    return wanted


def main() -> int:
    folders = []
    for package in ("PySide6", "shiboken6"):
        spec = importlib.util.find_spec(package)         # without importing (that needs the DLLs)
        if spec and spec.submodule_search_locations:
            folders.append(Path(spec.submodule_search_locations[0]))
    system32 = Path(os.environ.get("SystemRoot", r"C:\windows")) / "System32"
    for dll, names in sorted(imported_names(folders).items()):
        target = system32 / dll
        if target.exists() and MARKER not in target.read_bytes():
            continue                                     # a real one (a newer Wine, say)
        target.write_bytes(stub_dll(dll, sorted(names)))
        print(f"   {target}: stand-in exporting {len(names)} functions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
