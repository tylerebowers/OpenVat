"""Orbit camera for the 3D viewport (pure numpy, no Qt)."""

from __future__ import annotations

import math

import numpy as np

# Named views: (yaw degrees, pitch degrees).  Yaw 0 looks along +Y (front).
VIEWS = {
    "iso":    (-35.0, 30.0),
    "front":  (0.0, 0.0),
    "back":   (180.0, 0.0),
    "left":   (-90.0, 0.0),
    "right":  (90.0, 0.0),
    "top":    (0.0, 89.9),
    "bottom": (0.0, -89.9),
}


class OrbitCamera:
    def __init__(self) -> None:
        self.target = np.array([0.0, 0.0, 20.0])
        self.distance = 300.0
        self.yaw = VIEWS["iso"][0]
        self.pitch = VIEWS["iso"][1]
        self.fov = 40.0
        self.aspect = 1.0

    # ------------------------------------------------------------ vectors
    def position(self) -> np.ndarray:
        yaw, pitch = math.radians(self.yaw), math.radians(self.pitch)
        d = np.array([math.sin(yaw) * math.cos(pitch),
                      -math.cos(yaw) * math.cos(pitch),
                      math.sin(pitch)])
        return self.target + d * self.distance

    def forward(self) -> np.ndarray:
        f = self.target - self.position()
        return f / np.linalg.norm(f)

    def right(self) -> np.ndarray:
        r = np.cross(self.forward(), [0, 0, 1])
        n = np.linalg.norm(r)
        return r / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])

    def up(self) -> np.ndarray:
        return np.cross(self.right(), self.forward())

    # ------------------------------------------------------------ matrices
    def view_matrix(self) -> np.ndarray:
        eye = self.position()
        f, r, u = self.forward(), self.right(), self.up()
        m = np.eye(4)
        m[0, :3], m[1, :3], m[2, :3] = r, u, -f
        m[0, 3], m[1, 3], m[2, 3] = -r @ eye, -u @ eye, f @ eye
        return m

    def projection_matrix(self, near: float = 1.0, far: float = 5000.0) -> np.ndarray:
        f = 1.0 / math.tan(math.radians(self.fov) / 2)
        m = np.zeros((4, 4))
        m[0, 0] = f / self.aspect
        m[1, 1] = f
        m[2, 2] = (far + near) / (near - far)
        m[2, 3] = 2 * far * near / (near - far)
        m[3, 2] = -1.0
        return m

    # ------------------------------------------------------------ control
    def orbit(self, dx: float, dy: float) -> None:
        self.yaw = (self.yaw + dx * 0.4) % 360.0
        self.pitch = float(np.clip(self.pitch + dy * 0.4, -89.9, 89.9))

    def pan(self, dx: float, dy: float) -> None:
        scale = self.distance * 0.0015
        self.target = self.target - self.right() * dx * scale + self.up() * dy * scale

    def zoom(self, steps: float) -> None:
        self.distance = float(np.clip(self.distance * (0.9 ** steps), 5.0, 3000.0))

    def set_view(self, name: str) -> None:
        self.yaw, self.pitch = VIEWS[name]

    def look_from(self, direction) -> None:
        """Place the camera on the given (unit) direction from the target."""
        d = np.asarray(direction, dtype=float)
        d /= np.linalg.norm(d)
        self.pitch = float(np.clip(math.degrees(math.asin(d[2])), -89.9, 89.9))
        if abs(d[2]) < 0.999:
            self.yaw = math.degrees(math.atan2(d[0], -d[1])) % 360.0

    def orbit_by(self, dyaw: float, dpitch: float) -> None:
        self.yaw = (self.yaw + dyaw) % 360.0
        self.pitch = float(np.clip(self.pitch + dpitch, -89.9, 89.9))

    def fit(self, bounds: np.ndarray | None, plate_size: tuple[float, float]) -> None:
        """Frame the given bounds (or the build plate when None)."""
        if bounds is None:
            lo = np.array([-plate_size[0] / 2, -plate_size[1] / 2, 0.0])
            hi = np.array([plate_size[0] / 2, plate_size[1] / 2, 0.0])
        else:
            lo, hi = bounds
        self.target = (lo + hi) / 2
        radius = max(np.linalg.norm(hi - lo) / 2, 10.0)
        self.distance = radius / math.sin(math.radians(self.fov) / 2) * 1.1

    def ray(self, px: float, py: float, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        """World-space ray (origin, direction) through pixel (px, py)."""
        x = 2.0 * px / width - 1.0
        y = 1.0 - 2.0 * py / height
        inv = np.linalg.inv(self.projection_matrix() @ self.view_matrix())
        near = inv @ np.array([x, y, -1.0, 1.0])
        far = inv @ np.array([x, y, 1.0, 1.0])
        near, far = near[:3] / near[3], far[:3] / far[3]
        d = far - near
        return near, d / np.linalg.norm(d)


def ray_plane_z(origin: np.ndarray, direction: np.ndarray, z: float) -> np.ndarray | None:
    """Intersection of a ray with the horizontal plane at height z."""
    if abs(direction[2]) < 1e-9:
        return None
    t = (z - origin[2]) / direction[2]
    if t < 0:
        return None
    return origin + direction * t
