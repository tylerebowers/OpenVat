"""Running totals for the home page ("stats" in settings.json): how many
models were opened, projects saved, slices and layers made, supports
placed, print files written (and how many of them to a USB drive), and the
resin and print time in the files written."""

from __future__ import annotations

# key -> label, in the order the home page shows them
STATS = {
    "models": "Models opened",
    "projects": "Projects saved",
    "slices": "Slices",
    "layers": "Layers sliced",
    "supports": "Supports placed",
    "exports": "Print files exported",
    "resin_ml": "Resin in exported files",
    "print_s": "Print time exported",
}


def totals(settings: dict) -> dict[str, float]:
    """Every total (0 for those never counted); bad values count as 0."""
    saved = settings.get("stats")
    saved = saved if isinstance(saved, dict) else {}
    out = {}
    for key in (*STATS, "usb"):
        try:
            out[key] = max(float(saved.get(key, 0) or 0), 0.0)
        except (TypeError, ValueError):
            out[key] = 0.0
    return out


def add(settings: dict, **amounts) -> None:
    """Add to totals in ``settings`` (the caller saves it)."""
    t = totals(settings)
    for key, amount in amounts.items():
        t[key] = t.get(key, 0.0) + float(amount)
    settings["stats"] = {k: (round(v, 3) if k in ("resin_ml", "print_s") else int(round(v))) for k, v in t.items()}


def shown(key: str, value: float) -> str:
    """How a total reads: '1,234', '86.4 mL', '1.25 L', '45 min', '12.5 h', '312 h'."""
    if key == "resin_ml":
        return f"{value / 1000:.2f} L" if value >= 1000 else f"{value:.1f} mL" if value < 100 else f"{value:.0f} mL"
    if key == "print_s":
        hours = value / 3600
        if hours < 1:
            return f"{value / 60:.0f} min"
        return f"{hours:.1f} h" if hours < 100 else f"{hours:,.0f} h"
    return f"{int(value):,}"
