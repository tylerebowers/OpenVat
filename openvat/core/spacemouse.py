"""3Dconnexion SpaceMouse: reading the device, and what its motion does.

Two ways in, tried in this order:

* **spacenavd**, the free SpaceMouse driver (package ``spacenavd`` on
  Linux, also for the BSDs and macOS): its socket (``$SPNAV_SOCKET``,
  ``$XDG_RUNTIME_DIR/spnav.sock`` or ``/var/run/spnav.sock``) sends every
  event as eight 32-bit integers - the type (0 motion, 1 button press,
  2 button release), then x y z rx ry rz and the period in ms for motion,
  the button number (0 = the first) for buttons;
* **the device itself** (Linux): its ``/dev/input/event*`` node, found by
  its six axes and its maker (3Dconnexion, or Logitech for the older
  ones).  Reading it needs permission: desktops usually give it to the
  user at the screen, otherwise a udev rule does - or spacenavd.

Both give motion in spacenavd's axes: x to the right, y up, z away from
the user; rx, ry, rz turn about those by the right-hand rule (rx > 0 tilts
the top of the cap toward the user, ry > 0 twists it counter-clockwise seen
from above).  The device's own axes (y toward the user, z down) are turned
into these the way spacenavd does it.

``navigation`` turns motion into camera moves the way 3Dconnexion's
"object mode" does: the model follows the cap - push it right and the
model goes right, lift it and the model goes up, pull it toward you and
the model comes closer, twist it and the model turns, tilt its top toward
you and you see more of the model's top.  ``SpaceMouseSettings`` keeps the
speed, which moves to reverse and what the first two buttons do.
"""

from __future__ import annotations

import errno
import glob
import math
import os
import socket
import struct
import sys
import time
from dataclasses import asdict, dataclass, fields

THREEDCONNEXION = 0x256F
LOGITECH = 0x046D                       # the older ones (SpaceNavigator, SpaceExplorer, SpacePilot ...);
LOGITECH_6DOF = {0xC603, 0xC605, 0xC606, 0xC621, 0xC623, 0xC625,      # not its gamepads, which have
                 0xC626, 0xC627, 0xC628, 0xC629, 0xC62B}              # six axes too
NAMES = ("3dconnexion", "spacemouse", "spacenavigator", "space navigator", "spacepilot", "spaceexplorer",
         "spaceball", "spacetraveler")
FULL_SCALE = 350.0                      # raw value at full deflection (spacenavd, most devices)
DEAD_ZONE = 0.04                        # of full deflection: a resting hand moves nothing
CURVE = 1.5                             # response: gentle near the middle, full speed at the end

# what a button can do (key -> label)
ACTIONS = {
    "home": "Home view",
    "turn90": "Turn the view 90°",
    "fit": "Fit to the scene",
    "top": "Top view",
    "front": "Front view",
    "rotate_model": "Turn the selected model 90°",
    "none": "Nothing",
}


@dataclass
class SpaceMouseSettings:
    enabled: bool = True
    speed: float = 1.0                  # times the speeds below
    invert_pan: bool = False
    invert_zoom: bool = False
    invert_tilt: bool = False
    invert_spin: bool = False
    button1: str = "home"
    button2: str = "turn90"

    PAN_PX = 700.0                      # like a mouse drag of this many px per second, at full deflection
    ZOOM_STEPS = 6.0                    # wheel notches per second
    TURN_DEG = 120.0                    # degrees per second

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d) -> "SpaceMouseSettings":
        out = cls()
        if not isinstance(d, dict):
            return out
        for f in fields(cls):
            if f.name not in d:
                continue
            v = d[f.name]
            kind = str(f.type)
            try:
                if kind == "bool":
                    setattr(out, f.name, bool(v))
                elif kind == "float":
                    setattr(out, f.name, min(max(float(v), 0.1), 5.0))
                elif str(v) in ACTIONS:
                    setattr(out, f.name, str(v))
            except (TypeError, ValueError):
                pass
        return out

    def action(self, button: int) -> str:
        """What button ``button`` (0 = the first) does."""
        return {0: self.button1, 1: self.button2}.get(button, "none")


@dataclass
class Motion:
    axes: tuple                         # x, y, z, rx, ry, rz in -1..1 (see the module notes)


@dataclass
class Button:
    index: int                          # 0 = the first button
    pressed: bool


def normalized(raw, scale=FULL_SCALE) -> tuple:
    """Raw axis values -> -1..1: the dead zone taken out, then the response
    curve.  ``scale`` is the full deflection (one for all, or one per axis)."""
    scales = scale if isinstance(scale, (list, tuple)) else [scale] * len(raw)
    out = []
    for v, s in zip(raw, scales):
        a = max(-1.0, min(1.0, float(v) / s))
        m = (abs(a) - DEAD_ZONE) / (1 - DEAD_ZONE)
        out.append(math.copysign(m ** CURVE, a) if m > 0 else 0.0)
    return tuple(out)


def navigation(axes, s: SpaceMouseSettings, dt: float) -> dict:
    """Camera moves for ``dt`` seconds of deflection ``axes`` (normalized):
    pan (px right, px down - like a mouse drag, see OrbitCamera.pan), zoom
    (wheel notches, + = in), yaw and pitch (degrees, as OrbitCamera.orbit_by
    takes them).  The twist about the line of sight (rz) is not used."""
    x, y, z, rx, ry, _rz = axes
    k = s.speed * dt
    pan = -1.0 if s.invert_pan else 1.0
    return {
        "pan": (pan * x * s.PAN_PX * k, -pan * y * s.PAN_PX * k),         # lifted: the model goes up
        "zoom": (1.0 if s.invert_zoom else -1.0) * z * s.ZOOM_STEPS * k,  # pulled toward you: closer
        "yaw": (1.0 if s.invert_spin else -1.0) * ry * s.TURN_DEG * k,    # twisted left: so is the model
        "pitch": (-1.0 if s.invert_tilt else 1.0) * rx * s.TURN_DEG * k,  # top toward you: see its top
    }


# --------------------------------------------------------------------------
# spacenavd

def spnav_socket_paths() -> list[str]:
    paths = []
    if os.environ.get("SPNAV_SOCKET"):
        paths.append(os.environ["SPNAV_SOCKET"])
    if os.environ.get("XDG_RUNTIME_DIR"):
        paths.append(os.path.join(os.environ["XDG_RUNTIME_DIR"], "spnav.sock"))
    paths += ["/var/run/spnav.sock", "/run/spnav.sock"]
    return list(dict.fromkeys(paths))


class SpnavSource:
    """Events from spacenavd's socket.  spacenavd sends motion when it
    changes, so the last motion holds until the next."""
    EVENT = struct.Struct("=8i")

    def __init__(self, sock: socket.socket, path: str):
        self.sock, self.path = sock, path
        self.sock.setblocking(False)
        self._buf = b""
        self.name = "SpaceMouse (spacenavd)"

    @classmethod
    def open(cls) -> "SpnavSource | None":
        if not hasattr(socket, "AF_UNIX"):
            return None
        for path in spnav_socket_paths():
            if not os.path.exists(path):
                continue
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                s.connect(path)
            except OSError:
                s.close()
                continue
            return cls(s, path)
        return None

    def fileno(self) -> int:
        return self.sock.fileno()

    def read(self) -> list:
        """The events waiting.  Raises EOFError when spacenavd went away."""
        while True:
            try:
                chunk = self.sock.recv(4096)
            except BlockingIOError:
                break
            except OSError as exc:
                raise EOFError(str(exc)) from None
            if not chunk:
                raise EOFError("spacenavd closed the connection")
            self._buf += chunk
        events = []
        size = self.EVENT.size
        while len(self._buf) >= size:
            data = self.EVENT.unpack_from(self._buf)
            self._buf = self._buf[size:]
            if data[0] == 0:
                events.append(Motion(normalized(data[1:7])))
            elif data[0] in (1, 2):
                events.append(Button(int(data[1]), data[0] == 1))
        return events

    def settle(self) -> None:
        return None

    def close(self) -> None:
        self.sock.close()


# --------------------------------------------------------------------------
# the device itself (Linux evdev)

EV_KEY, EV_REL, EV_ABS = 1, 2, 3
BTN_MISC = 0x100
REL_TIMEOUT = 0.15                      # s: the kernel drops relative values of 0, so an axis not
                                        # reported for this long is back at rest
DEVICE_AXES = (0, 2, 1, 3, 5, 4)        # spacenavd's axis i is the device's axis DEVICE_AXES[i] ...
DEVICE_SIGNS = (1, -1, -1, 1, -1, -1)   # ... times this (y up, z away instead of y toward you, z down)


@dataclass
class EvdevDevice:
    path: str                           # /dev/input/eventN
    name: str
    readable: bool


def _sys_read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def _has_six_axes(caps: str) -> bool:
    """A /sys .../capabilities/{rel,abs} bitmask (hex words, highest first)
    with the first six axes (x y z rx ry rz) set."""
    words = caps.split()
    if not words:
        return False
    try:
        low = int(words[-1], 16)
    except ValueError:
        return False
    return low & 0x3F == 0x3F


def find_evdev(root: str = "/dev/input", sys_root: str = "/sys/class/input") -> list[EvdevDevice]:
    """SpaceMice among the input devices: six axes, and 3Dconnexion's
    vendor id, one of its older Logitech product ids or a name that says so."""
    found = []
    for node in sorted(glob.glob(os.path.join(root, "event*"))):
        dev = os.path.join(sys_root, os.path.basename(node), "device")
        name = _sys_read(os.path.join(dev, "name"))
        try:
            vendor = int(_sys_read(os.path.join(dev, "id", "vendor")) or "0", 16)
            product = int(_sys_read(os.path.join(dev, "id", "product")) or "0", 16)
        except ValueError:
            vendor = product = 0
        six = _has_six_axes(_sys_read(os.path.join(dev, "capabilities", "rel"))) or \
            _has_six_axes(_sys_read(os.path.join(dev, "capabilities", "abs")))
        known = vendor == THREEDCONNEXION or (vendor == LOGITECH and product in LOGITECH_6DOF) \
            or any(n in name.lower() for n in NAMES)
        if six and known:
            found.append(EvdevDevice(node, name or os.path.basename(node), os.access(node, os.R_OK)))
    return found


def _ioc_read(nr: int, size: int) -> int:
    """_IOR('E', nr, size): an evdev ioctl that reads ``size`` bytes."""
    return (2 << 30) | (size << 16) | (ord("E") << 8) | nr


class EvdevSource:
    """Events from a SpaceMouse's /dev/input/eventN node."""
    EVENT = struct.Struct("llHHi")      # struct input_event: time (two longs), type, code, value

    def __init__(self, fd: int, device: EvdevDevice):
        self.fd, self.device = fd, device
        self.name = device.name
        self._buf = b""
        self.raw = [0.0] * 6                    # the device's own axes
        self.relative = [False] * 6
        self.seen = [0.0] * 6                   # when each relative axis was last reported
        self.scale = [FULL_SCALE] * 6
        self.button_base = BTN_MISC
        self._ask_device()

    @classmethod
    def open(cls, device: EvdevDevice) -> "EvdevSource":
        fd = os.open(device.path, os.O_RDONLY | os.O_NONBLOCK)
        return cls(fd, device)

    def _ask_device(self) -> None:
        """Full deflection of absolute axes (EVIOCGABS) and the first button
        (EVIOCGBIT for keys: spacenavd counts buttons from the lowest)."""
        try:
            import fcntl
        except ImportError:                     # pragma: no cover - not Linux
            return
        for axis in range(6):
            try:
                info = struct.unpack("6i", fcntl.ioctl(self.fd, _ioc_read(0x40 + axis, 24), bytes(24)))
            except OSError:
                continue
            reach = max(abs(info[1]), abs(info[2]))
            if reach > 0:
                self.scale[axis] = float(reach)
        try:
            bits = fcntl.ioctl(self.fd, _ioc_read(0x20 + EV_KEY, 96), bytes(96))
        except OSError:
            return
        for i, byte in enumerate(bits):
            if byte:
                self.button_base = i * 8 + ((byte & -byte).bit_length() - 1)
                return

    def fileno(self) -> int:
        return self.fd

    def read(self) -> list:
        """The events waiting.  Raises EOFError when the device is gone."""
        while True:
            try:
                chunk = os.read(self.fd, 64 * self.EVENT.size)
            except BlockingIOError:
                break
            except OSError as exc:
                if exc.errno in (errno.EAGAIN, errno.EINTR):
                    break
                raise EOFError(str(exc)) from None     # unplugged (ENODEV)
            if not chunk:
                break
            self._buf += chunk
        events, moved = [], False
        size = self.EVENT.size
        now = time.monotonic()
        while len(self._buf) >= size:
            _s, _us, kind, code, value = self.EVENT.unpack_from(self._buf)
            self._buf = self._buf[size:]
            if kind in (EV_REL, EV_ABS) and code < 6:
                self.raw[code] = float(value)
                self.relative[code] = kind == EV_REL
                self.seen[code] = now
                moved = True
            elif kind == EV_KEY and value in (0, 1) and code >= self.button_base:
                events.append(Button(code - self.button_base, value == 1))
        if moved:
            events.append(self.motion())
        return events

    def motion(self) -> Motion:
        """The device's axes now, in spacenavd's axes, normalized."""
        values = [self.raw[DEVICE_AXES[i]] * DEVICE_SIGNS[i] for i in range(6)]
        scales = [self.scale[DEVICE_AXES[i]] for i in range(6)]
        return Motion(normalized(values, scales))

    def settle(self) -> Motion | None:
        """Relative axes back at rest when not reported for a while: the
        motion then, or None when nothing changed."""
        now = time.monotonic()
        quiet = [rel and v != 0 and now - t > REL_TIMEOUT
                 for v, rel, t in zip(self.raw, self.relative, self.seen)]
        if not any(quiet):
            return None
        self.raw = [0.0 if q else v for v, q in zip(self.raw, quiet)]
        return self.motion()

    def close(self) -> None:
        try:
            os.close(self.fd)
        except OSError:
            pass


# --------------------------------------------------------------------------

def connect():
    """(source, status): spacenavd first, then the device itself.  source is
    None (and status says why) when there is no way to a SpaceMouse."""
    spnav = SpnavSource.open()
    if spnav is not None:
        return spnav, f"Connected through spacenavd ({spnav.path})"
    if sys.platform == "win32":
        return None, "SpaceMice work on Linux (and on macOS with spacenavd); Windows is not supported yet"
    if not sys.platform.startswith("linux"):
        return None, ("No SpaceMouse found. On this system OpenVat reads it through spacenavd "
                      "(3Dconnexion's own driver is not supported yet)")
    devices = find_evdev()
    for dev in devices:
        if dev.readable:
            try:
                return EvdevSource.open(dev), f"Connected to {dev.name} ({dev.path})"
            except OSError:
                continue
    if devices:
        dev = devices[0]
        return None, (f"{dev.name} is plugged in, but OpenVat may not read {dev.path}: install and "
                      "start spacenavd, or let your user read the device (a udev rule)")
    return None, "No SpaceMouse found (plug it in, or start spacenavd)"
