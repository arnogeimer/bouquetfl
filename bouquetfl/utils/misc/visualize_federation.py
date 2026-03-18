"""
visualize_federation.py — world-map visualisation of a BouquetFL federation.

Shows clients (blue) and server (red) as dots at random positions within their
country's borders. Black arcs from each client to the server are annotated with
the upload/download speed of that client's location. Hardware config and — when
available — real measured timings are shown next to each client dot.

Called automatically from server_app.py after the federation finishes.

Optional dependency (not in main pyproject.toml):
    cartopy >= 0.23  —  install with: uv add cartopy
    matplotlib       —  install with: uv add matplotlib
    shapely          —  install with: uv add shapely
"""

import random
import tomllib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom
import shapely.ops


# ---------------------------------------------------------------------------
# Country geometry cache — loaded once per process
# ---------------------------------------------------------------------------

_country_geom_cache: dict[str, sgeom.base.BaseGeometry] = {}


def _get_country_geom(name: str) -> sgeom.base.BaseGeometry | None:
    """Return the shapely geometry for a Natural Earth country name, or None."""
    if name in _country_geom_cache:
        return _country_geom_cache[name]

    shpfile = shpreader.natural_earth(
        resolution="50m", category="cultural", name="admin_0_countries"
    )
    reader = shpreader.Reader(shpfile)
    for record in reader.records():
        country_name = record.attributes.get("NAME", "")
        geom = record.geometry
        # Unify multi-polygons into a single geometry
        if hasattr(geom, "geoms"):
            geom = shapely.ops.unary_union(geom.geoms)
        _country_geom_cache[country_name] = geom

    return _country_geom_cache.get(name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_point(location: str) -> tuple[float, float]:
    """Return a random (lon, lat) point inside the named country's borders."""
    geom = _get_country_geom(location)
    if geom is None:
        # Fallback: uniform random point on the globe
        return random.uniform(-180, 180), random.uniform(-60, 80)

    minx, miny, maxx, maxy = geom.bounds
    for _ in range(1000):
        p = sgeom.Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if geom.contains(p):
            return p.x, p.y

    # Fallback: centroid if rejection sampling fails (e.g. tiny island nation)
    c = geom.centroid
    return c.x, c.y


def _great_circle_path(lon1: float, lat1: float, lon2: float, lat2: float,
                        n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Return (lons, lats) arrays for a smooth great-circle arc using SLERP."""
    lon1r, lat1r = np.radians(lon1), np.radians(lat1)
    lon2r, lat2r = np.radians(lon2), np.radians(lat2)

    # Unit vectors on the sphere
    v1 = np.array([np.cos(lat1r) * np.cos(lon1r),
                   np.cos(lat1r) * np.sin(lon1r),
                   np.sin(lat1r)])
    v2 = np.array([np.cos(lat2r) * np.cos(lon2r),
                   np.cos(lat2r) * np.sin(lon2r),
                   np.sin(lat2r)])

    omega = np.arccos(np.clip(v1 @ v2, -1.0, 1.0))
    if omega < 1e-10:
        return np.array([lon1, lon2]), np.array([lat1, lat2])

    t  = np.linspace(0, 1, n)
    vs = (np.sin((1 - t) * omega) / np.sin(omega))[:, None] * v1 \
       + (np.sin(       t * omega) / np.sin(omega))[:, None] * v2

    lats = np.degrees(np.arcsin(np.clip(vs[:, 2], -1.0, 1.0)))
    lons = np.degrees(np.arctan2(vs[:, 1], vs[:, 0]))
    return lons, lats


_LOCATIONS_SCHEMA_VERSION = 1


def _load_location_speeds(location: str) -> dict:
    with open("networkconf/locations.toml", "rb") as f:
        data = tomllib.load(f)
    v = data.get("schema_version", 0)
    if v != _LOCATIONS_SCHEMA_VERSION:
        raise RuntimeError(f"networkconf/locations.toml schema_version={v}, expected {_LOCATIONS_SCHEMA_VERSION}")
    for loc in data["locations"]:
            if loc["name"] == location:
                return loc
    return {"upload_mbps": 0.0, "download_mbps": 0.0}


# ---------------------------------------------------------------------------
# Public entry point — called from server_app.py
# ---------------------------------------------------------------------------

def visualize(
    hardware_config: dict,
    server_location: str = "Germany",
    output_path: str = "federation_map.pdf",
) -> None:
    """Render a world map of the federation and save it to output_path.

    Parameters
    ----------
    hardware_config : dict
        In-memory hardware config dict keyed by "client_0", "client_1", …
    server_location : str
        Location name for the server (must match a Natural Earth country name).
    output_path : str
        Path where the PDF image is saved.
    """
    fig = plt.figure(figsize=(20, 11))
    ax  = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    ax.set_global()
    ax.add_feature(cfeature.LAND,      facecolor="#e8e8e8")
    ax.add_feature(cfeature.OCEAN,     facecolor="#cce5f5")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#888888")
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3, edgecolor="#aaaaaa")
    ax.gridlines(draw_labels=False, linewidth=0.3, color="#cccccc", linestyle="--")

    # Server
    server_lon, server_lat = _random_point(server_location)
    ax.plot(server_lon, server_lat, "r*", markersize=18,
            transform=ccrs.PlateCarree(), zorder=5)
    ax.text(server_lon + 1.5, server_lat + 1.5, f"SERVER\n{server_location}",
            fontsize=7, color="darkred", fontweight="bold",
            transform=ccrs.PlateCarree(), zorder=6)

    # Clients
    for client_key, profile in hardware_config.items():
        location = profile.get("location", "Germany")
        speeds       = _load_location_speeds(location)
        c_lon, c_lat = _random_point(location)

        ax.plot(c_lon, c_lat, "o", color="#2255cc", markersize=9,
                transform=ccrs.PlateCarree(), zorder=5)

        # Great-circle arc to server (SLERP-interpolated for smooth rendering)
        arc_lons, arc_lats = _great_circle_path(c_lon, c_lat, server_lon, server_lat)
        ax.plot(arc_lons, arc_lats,
                color="black", linewidth=1.0, alpha=0.6,
                transform=ccrs.PlateCarree(), zorder=3)

        # Hardware + timing label
        hw_lines = (
            f"{client_key}\n"
            f"GPU: {profile['gpu']}\n"
            f"CPU: {profile['cpu']}\n"
            f"RAM: {profile['ram_gb']} GB\n"
            f"Loc: {location}\n"
            f"↑{speeds['upload_mbps']:.0f} ↓{speeds['download_mbps']:.0f} Mbps"
        )
        ax.text(c_lon + 1.5, c_lat - 1.0, hw_lines,
                fontsize=5.5, color="#112266", verticalalignment="top",
                transform=ccrs.PlateCarree(), zorder=6,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#2255cc", alpha=0.85, linewidth=0.6))

    ax.legend(handles=[
        mpatches.Patch(color="#2255cc", label="Client"),
        mpatches.Patch(color="red",     label="Server"),
    ], loc="lower left", fontsize=9)

    ax.set_title("BouquetFL — Federation Hardware Map", fontsize=14, pad=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[visualize] saved to {output_path}")
    plt.close(fig)
