"""USB drives (and SD cards) to export print files to.

``removable_drives()`` lists the removable drives plugged in now:

* Linux: partitions of USB / removable disks and SD cards, read from
  /proc/self/mountinfo and /sys (no root, no extra packages).  A drive
  that is plugged in but not mounted yet (KDE and GNOME leave new sticks
  unmounted until something opens them) is listed too; ``mount`` mounts it
  with udisks (``udisksctl``, or ``gio``), as the file manager would.
* Windows: drive letters of type DRIVE_REMOVABLE that have media in them.
* macOS: the volumes under /Volumes other than the system disk.

``eject`` unmounts a drive (and powers it off on Linux) so it can be pulled
out safely.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Drive:
    label: str                 # what to call it: the volume label, or the device name
    path: str = ""             # where it is mounted ("" = not mounted yet, Linux only)
    device: str = ""           # /dev/sdb1 (Linux), E: (Windows), /dev/disk4s1 (macOS) when known

    @property
    def mounted(self) -> bool:
        return bool(self.path)

    def free_bytes(self) -> int | None:
        try:
            return shutil.disk_usage(self.path).free if self.path else None
        except OSError:
            return None

    def describe(self) -> str:
        """'KINGSTON (14.2 GB free)' / 'KINGSTON (not mounted)'."""
        free = self.free_bytes()
        if not self.mounted:
            return f"{self.label} (not mounted yet)"
        return f"{self.label} ({free / 1e9:.1f} GB free)" if free is not None else self.label


def removable_drives() -> list[Drive]:
    """The removable drives plugged in now (empty when none, or on error)."""
    try:
        if sys.platform.startswith("linux"):
            return linux_drives()
        if sys.platform == "win32":
            return _windows_drives()
        if sys.platform == "darwin":
            return _mac_drives()
    except Exception:                       # never let a probe break the app
        pass
    return []


# ------------------------------------------------------------------- Linux
_SKIP_FS = {"squashfs", "iso9660", "udf", "overlay", "tmpfs", "proc", "sysfs", "devtmpfs"}


def _unescape(s: str) -> str:
    """Mount points in mountinfo escape space, tab, newline and backslash as \\ooo."""
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), s)


def _mounts(mountinfo: str) -> list[tuple[str, str, str, list[str]]]:
    """(device, mount point, fs type, mount options) per line of mountinfo."""
    out = []
    try:
        with open(mountinfo, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return out
    for line in lines:
        left, sep, right = line.partition(" - ")
        lf, rf = left.split(), right.split()
        if not sep or len(lf) < 6 or len(rf) < 2:
            continue
        out.append((rf[1], _unescape(lf[4]), rf[0], lf[5].split(",")))
    return out


def _removable(name: str, sys_block: str) -> bool:
    """Whether the block device ``name`` (sdb1, mmcblk0p1) is on a USB bus,
    on a disk flagged removable, or an SD card."""
    node = os.path.join(sys_block, name)
    if not os.path.exists(node):
        return False
    real = os.path.realpath(node)
    if "/usb" in real or name.startswith("mmcblk"):
        return True
    disk = os.path.dirname(real) if os.path.exists(os.path.join(real, "partition")) else real
    try:
        with open(os.path.join(disk, "removable")) as f:
            return f.read().strip() == "1"
    except OSError:
        return False


def _links(folder: str) -> dict[str, str]:
    """Device name -> link name, for /dev/disk/by-label and by-uuid (udev
    writes a space in a label as \\x20)."""
    out = {}
    try:
        for entry in os.listdir(folder):
            name = os.path.basename(os.path.realpath(os.path.join(folder, entry)))
            out[name] = re.sub(r"\\x([0-9a-fA-F]{2})", lambda m: chr(int(m.group(1), 16)), entry)
    except OSError:
        pass
    return out


def _swaps(path: str = "/proc/swaps") -> set[str]:
    try:
        with open(path) as f:
            return {os.path.basename(line.split()[0]) for line in f.readlines()[1:] if line.strip()}
    except OSError:
        return set()


def linux_drives(mountinfo: str = "/proc/self/mountinfo", sys_block: str = "/sys/class/block",
                 dev_disk: str = "/dev/disk") -> list[Drive]:
    """Removable drives on Linux (the paths are parameters for the tests)."""
    drives: dict[str, Drive] = {}
    mounted = set()                       # every mounted device, also the ones left out (read-only)
    for source, mount, fstype, opts in _mounts(mountinfo):
        if not source.startswith("/dev/"):
            continue
        name = os.path.basename(os.path.realpath(source)) if os.path.exists(source) else os.path.basename(source)
        mounted.add(name)
        if fstype in _SKIP_FS or "ro" in opts:
            continue
        if name in drives or mount == "/" or not _removable(name, sys_block):
            continue
        drives[name] = Drive(label=os.path.basename(mount.rstrip("/")) or name, path=mount, device=f"/dev/{name}")
    # plugged in but not mounted: partitions with a file system (udev gives them a UUID)
    labels = _links(os.path.join(dev_disk, "by-label"))
    with_fs = _links(os.path.join(dev_disk, "by-uuid"))
    swaps = _swaps()
    for name in sorted(with_fs):
        if name in mounted or name in swaps or name.startswith(("loop", "ram", "zram", "dm-", "sr")):
            continue
        if _removable(name, sys_block) and not _has_partitions(name, sys_block):
            drives[name] = Drive(label=labels.get(name, name), path="", device=f"/dev/{name}")
    return list(drives.values())


def _has_partitions(name: str, sys_block: str) -> bool:
    """A whole disk whose partitions are listed on their own."""
    real = os.path.realpath(os.path.join(sys_block, name))
    try:
        return any(os.path.exists(os.path.join(real, e, "partition")) for e in os.listdir(real))
    except OSError:
        return False


def _run(args: list[str], timeout: float = 30.0) -> tuple[bool, str]:
    try:
        kw = {}
        if sys.platform == "win32":
            kw["creationflags"] = 0x08000000           # CREATE_NO_WINDOW
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout, **kw)
        return p.returncode == 0, (p.stdout + p.stderr).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)


def mount(drive: Drive) -> Drive:
    """The drive mounted (as the file manager would mount it); raises
    OSError with the reason when that fails."""
    if drive.mounted:
        return drive
    errors = []
    for tool, args in (("udisksctl", ["mount", "--no-user-interaction", "-b", drive.device]),
                       ("gio", ["mount", "-d", drive.device])):
        if not shutil.which(tool):
            continue
        ok, out = _run([tool, *args])
        for d in linux_drives():                      # where it went
            if d.device == drive.device and d.mounted:
                return d
        errors.append(out)
    raise OSError("Could not mount " + drive.device + (": " + "; ".join(e for e in errors if e) if any(errors)
                                                       else " (udisksctl is not installed)"))


def eject(drive: Drive) -> tuple[bool, str]:
    """Unmount the drive so it can be pulled out (Linux also powers it off).
    (True, "") when done, else (False, why)."""
    if sys.platform == "darwin":
        return _run(["diskutil", "eject", drive.path])
    if sys.platform == "win32":
        letter = drive.path.rstrip("\\/")
        script = (f"(New-Object -ComObject Shell.Application).Namespace(17).ParseName('{letter}')"
                  ".InvokeVerb('Eject')")
        return _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script])
    if shutil.which("udisksctl"):
        ok, out = _run(["udisksctl", "unmount", "--no-user-interaction", "-b", drive.device])
        if not ok:
            return False, out
        disk = _parent_disk(drive.device)
        if disk:
            _run(["udisksctl", "power-off", "--no-user-interaction", "-b", disk])   # best effort
        return True, ""
    if shutil.which("gio"):
        return _run(["gio", "mount", "-e", drive.path])
    return _run(["umount", drive.path])


def _parent_disk(device: str, sys_block: str = "/sys/class/block") -> str:
    name = os.path.basename(device)
    real = os.path.realpath(os.path.join(sys_block, name))
    if os.path.exists(os.path.join(real, "partition")):
        return "/dev/" + os.path.basename(os.path.dirname(real))
    return device


# ------------------------------------------------------------------- Windows
def _windows_drives() -> list[Drive]:
    import ctypes
    import string
    k = ctypes.windll.kernel32
    old = k.SetErrorMode(1)                    # SEM_FAILCRITICALERRORS: no "insert a disk" boxes
    try:
        mask = k.GetLogicalDrives()
        out = []
        for i, letter in enumerate(string.ascii_uppercase):
            if not mask >> i & 1:
                continue
            root = f"{letter}:\\"
            if k.GetDriveTypeW(root) != 2:       # DRIVE_REMOVABLE
                continue
            label = ctypes.create_unicode_buffer(261)
            if not k.GetVolumeInformationW(root, label, 261, None, None, None, None, 0):
                continue                          # a card reader with no card in it
            out.append(Drive(label=f"{label.value or 'USB drive'} ({letter}:)", path=root, device=f"{letter}:"))
        return out
    finally:
        k.SetErrorMode(old)


# ------------------------------------------------------------------- macOS
def _mac_drives() -> list[Drive]:
    out = []
    root_dev = os.stat("/").st_dev
    for name in sorted(os.listdir("/Volumes")):
        p = os.path.join("/Volumes", name)
        try:
            if os.path.ismount(p) and os.stat(p).st_dev != root_dev and os.access(p, os.W_OK):
                out.append(Drive(label=name, path=p))
        except OSError:
            continue
    return out
