"""How the 3D views light the models (Edit → Preferences → Lighting).

Two lights and an ambient term: a key light (the main one, from the front
right and above by default), a weaker fill light from the other side so
the shadowed side is not flat, and a highlight (shine, gloss) from the key
light.  Directions are given as azimuth (degrees around the vertical, 0 =
+X, 90 = +Y, so 270 is the front of the plate) and elevation (degrees
above the plate).  With ``follow_view`` the lights turn with the camera, so
the side you look at stays lit while you orbit.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import numpy as np


@dataclass
class Lighting:
    ambient: float = 0.30            # light everywhere, also in the shadows
    key: float = 0.55                # key light brightness
    key_azimuth: float = 304.0
    key_elevation: float = 54.0
    fill: float = 0.25               # fill light brightness
    fill_azimuth: float = 140.0
    fill_elevation: float = 21.0
    shine: float = 0.15              # highlight strength
    gloss: float = 32.0              # highlight sharpness (higher: smaller, sharper)
    follow_view: bool = False        # the lights turn with the camera

    # ---------------------------------------------------------------- settings.json
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d) -> "Lighting":
        out = cls()
        if not isinstance(d, dict):
            return out
        for f in fields(cls):
            if f.name in d:
                try:
                    setattr(out, f.name, bool(d[f.name]) if f.type in (bool, "bool") else float(d[f.name]))
                except (TypeError, ValueError):
                    pass
        return out.clamped()

    def clamped(self) -> "Lighting":
        """Values inside the ranges the shader and the dialog use."""
        c = Lighting(**self.to_dict())
        for name in ("ambient", "key", "fill", "shine"):
            setattr(c, name, min(max(getattr(c, name), 0.0), 2.0))
        c.gloss = min(max(c.gloss, 1.0), 256.0)
        for name in ("key_azimuth", "fill_azimuth"):
            setattr(c, name, getattr(c, name) % 360.0)
        for name in ("key_elevation", "fill_elevation"):
            setattr(c, name, min(max(getattr(c, name), -90.0), 90.0))
        return c

    # ---------------------------------------------------------------- shader input
    def directions(self, camera_yaw: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Unit vectors toward the key and fill lights (world space).  The
        camera's yaw turns them when they follow the view."""
        turn = math.radians(camera_yaw) if self.follow_view else 0.0
        return (_direction(self.key_azimuth, self.key_elevation, turn),
                _direction(self.fill_azimuth, self.fill_elevation, turn))


def _direction(azimuth: float, elevation: float, turn: float = 0.0) -> np.ndarray:
    a, e = math.radians(azimuth) + turn, math.radians(elevation)
    return np.array([math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e)], dtype=np.float32)
