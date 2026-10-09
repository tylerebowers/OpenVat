"""The 3D views' colors the user can change (Edit -> Preferences), as
#rrggbb - one set for the light and one for the dark theme; the grid is
derived from the plate.  Kept apart from the views so the preferences can be
read before OpenGL is set up (see glplatform)."""

DEFAULT_THEME_COLORS = {
    "light": {
        "background": "#e6e9ee",
        "plate": "#a9b2bf",
        "model": "#3a86c8",
        "selected": "#ef9a12",
        "support": "#2fa36a",
        "support_selected": "#86c51c",
    },
    "dark": {
        "background": "#1f2126",
        "plate": "#3a404b",
        "model": "#549ec7",
        "selected": "#fab840",
        "support": "#59cc8c",
        "support_selected": "#bff266",
    },
}
DEFAULT_COLORS = DEFAULT_THEME_COLORS["dark"]
