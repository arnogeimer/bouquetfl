"""
visualize_gif.py — animated GIF showing per-client progress through a
federated-learning round.

Each client goes through four sequential phases:
  1. download  — arc from server fills in blue
  2. load      — small progress bar below client label
  3. train     — small progress bar below client label
  4. upload    — arc from client fills in black

Clients progress independently: one may still be downloading while another
is already training.

Optional dependency (not in main pyproject.toml):
    cartopy >= 0.23, matplotlib, shapely, Pillow (for GIF export)
"""

from __future__ import annotations

import random
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from bouquetfl.utils.misc.visualize_federation import (
    _random_point,
    _great_circle_path,
    _load_location_speeds,
)


# ---------------------------------------------------------------------------
# Per-client state
# ---------------------------------------------------------------------------

@dataclass
class VisualClient:
    """Tracks a single client's position, timings, and current phase."""

    key: str
    profile: dict
    lon: float
    lat: float
    speeds: dict

    # Phase durations (seconds) — set before animation starts
    download_time: float = 0.0
    load_time: float = 0.0
    train_time: float = 0.0
    upload_time: float = 0.0

    # Precomputed arc path (lon/lat arrays) from client → server
    arc_lons: np.ndarray = field(default_factory=lambda: np.array([]))
    arc_lats: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def total_time(self) -> float:
        return self.download_time + self.load_time + self.train_time + self.upload_time

    def phase_at(self, t: float) -> tuple[str, float]:
        """Return (phase_name, fraction_within_phase) for a given global time t.

        Returns ("done", 1.0) when all phases are complete.
        """
        remaining = t

        if self.download_time > 0:
            if remaining < self.download_time:
                return "download", remaining / self.download_time
            remaining -= self.download_time

        if self.load_time > 0:
            if remaining < self.load_time:
                return "load", remaining / self.load_time
            remaining -= self.load_time

        if self.train_time > 0:
            if remaining < self.train_time:
                return "train", remaining / self.train_time
            remaining -= self.train_time

        if self.upload_time > 0:
            if remaining < self.upload_time:
                return "upload", remaining / self.upload_time
            remaining -= self.upload_time

        return "done", 1.0

    def completed_before(self, t: float, phase: str) -> bool:
        """True if the given phase has fully completed by time t."""
        boundaries = {
            "download": self.download_time,
            "load":     self.download_time + self.load_time,
            "train":    self.download_time + self.load_time + self.train_time,
            "upload":   self.total_time,
        }
        return t >= boundaries.get(phase, float("inf"))


# ---------------------------------------------------------------------------
# Frame drawing
# ---------------------------------------------------------------------------

_PHASE_COLORS = {
    "download": "#3377ee",
    "load":     "#ee8833",
    "train":    "#33aa55",
    "upload":   "#222222",
}

_PHASE_LABELS = {
    "download": "downloading",
    "load":     "loading",
    "train":    "training",
    "upload":   "uploading",
}


def _draw_base_map(fig, ax) -> None:
    """Draw the static map background."""
    ax.set_global()
    ax.add_feature(cfeature.LAND,      facecolor="#e8e8e8")
    ax.add_feature(cfeature.OCEAN,     facecolor="#cce5f5")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#888888")
    ax.add_feature(cfeature.BORDERS,   linewidth=0.3, edgecolor="#aaaaaa")
    ax.gridlines(draw_labels=False, linewidth=0.3, color="#cccccc", linestyle="--")


def _draw_arc_partial(ax, arc_lons, arc_lats, fraction: float,
                      color: str, linewidth: float = 2.0) -> None:
    """Draw a fraction of a precomputed great-circle arc."""
    n = len(arc_lons)
    end = max(2, int(n * fraction))
    ax.plot(arc_lons[:end], arc_lats[:end],
            color=color, linewidth=linewidth, alpha=0.8,
            transform=ccrs.PlateCarree(), zorder=4)


def _draw_progress_bar(ax, lon: float, lat: float, fraction: float,
                       color: str, label: str, y_offset: float = -4.0) -> None:
    """Draw a small progress bar in data coordinates below a client dot."""
    bar_w, bar_h = 14.0, 1.8
    x0 = lon - bar_w / 2
    y0 = lat + y_offset

    # Background
    bg = FancyBboxPatch((x0, y0), bar_w, bar_h,
                        boxstyle="round,pad=0.3",
                        facecolor="#dddddd", edgecolor="#999999", linewidth=0.5,
                        transform=ccrs.PlateCarree(), zorder=6)
    ax.add_patch(bg)

    # Fill
    if fraction > 0:
        fill_w = max(0.1, bar_w * fraction)
        fill = FancyBboxPatch((x0, y0), fill_w, bar_h,
                              boxstyle="round,pad=0.3",
                              facecolor=color, edgecolor="none",
                              transform=ccrs.PlateCarree(), zorder=7)
        ax.add_patch(fill)

    # Label
    ax.text(x0 + bar_w / 2, y0 + bar_h / 2, f"{label}  {fraction*100:.0f}%",
            fontsize=5, color="white", fontweight="bold",
            ha="center", va="center",
            transform=ccrs.PlateCarree(), zorder=8)


def _render_frame(clients: list[VisualClient],
                  server_lon: float, server_lat: float,
                  server_location: str,
                  t: float, t_max: float) -> plt.Figure:
    """Render a single animation frame at time t."""
    fig = plt.figure(figsize=(20, 11))
    ax  = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    _draw_base_map(fig, ax)

    # Server
    ax.plot(server_lon, server_lat, "r*", markersize=18,
            transform=ccrs.PlateCarree(), zorder=5)
    ax.text(server_lon + 1.5, server_lat + 1.5, f"SERVER\n{server_location}",
            fontsize=7, color="darkred", fontweight="bold",
            transform=ccrs.PlateCarree(), zorder=6)

    for vc in clients:
        phase, frac = vc.phase_at(t)

        # Client dot
        dot_color = _PHASE_COLORS.get(phase, "#2255cc")
        ax.plot(vc.lon, vc.lat, "o", color=dot_color, markersize=9,
                transform=ccrs.PlateCarree(), zorder=5)

        # Static dim arc (background track)
        ax.plot(vc.arc_lons, vc.arc_lats,
                color="#cccccc", linewidth=1.0, alpha=0.4,
                transform=ccrs.PlateCarree(), zorder=3)

        # Phase-specific visuals
        if phase == "download":
            # Arc fills blue from server → client
            _draw_arc_partial(ax, vc.arc_lons[::-1], vc.arc_lats[::-1],
                              frac, _PHASE_COLORS["download"])

        elif phase == "load":
            # Download arc stays fully blue
            _draw_arc_partial(ax, vc.arc_lons[::-1], vc.arc_lats[::-1],
                              1.0, _PHASE_COLORS["download"])
            _draw_progress_bar(ax, vc.lon, vc.lat, frac,
                               _PHASE_COLORS["load"], _PHASE_LABELS["load"])

        elif phase == "train":
            # Download arc stays
            _draw_arc_partial(ax, vc.arc_lons[::-1], vc.arc_lats[::-1],
                              1.0, _PHASE_COLORS["download"])
            _draw_progress_bar(ax, vc.lon, vc.lat, frac,
                               _PHASE_COLORS["train"], _PHASE_LABELS["train"])

        elif phase == "upload":
            # Upload arc fills black from client → server
            _draw_arc_partial(ax, vc.arc_lons, vc.arc_lats,
                              frac, _PHASE_COLORS["upload"])

        elif phase == "done":
            # Full black arc = upload complete
            _draw_arc_partial(ax, vc.arc_lons, vc.arc_lats,
                              1.0, _PHASE_COLORS["upload"], linewidth=1.5)
            ax.text(vc.lon + 1.5, vc.lat + 1.5, "✓",
                    fontsize=10, color="green", fontweight="bold",
                    transform=ccrs.PlateCarree(), zorder=8)

        # Client label (always shown)
        location = vc.profile.get("location", "")
        hw_lines = (
            f"{vc.key}\n"
            f"GPU: {vc.profile['gpu']}\n"
            f"↑{vc.speeds['upload_mbps']:.0f} ↓{vc.speeds['download_mbps']:.0f} Mbps"
        )
        ax.text(vc.lon + 1.5, vc.lat - 1.0, hw_lines,
                fontsize=5.5, color="#112266", verticalalignment="top",
                transform=ccrs.PlateCarree(), zorder=6,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#2255cc", alpha=0.85, linewidth=0.6))

    # Title with time indicator
    ax.set_title(f"BouquetFL — Round progress  t={t:.1f}s / {t_max:.1f}s",
                 fontsize=14, pad=10)

    ax.legend(handles=[
        mpatches.Patch(color=_PHASE_COLORS["download"], label="Download"),
        mpatches.Patch(color=_PHASE_COLORS["load"],     label="Load"),
        mpatches.Patch(color=_PHASE_COLORS["train"],    label="Train"),
        mpatches.Patch(color=_PHASE_COLORS["upload"],   label="Upload"),
        mpatches.Patch(color="red",                     label="Server"),
    ], loc="lower left", fontsize=8)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def visualize_gif(
    hardware_config: dict,
    timings: dict | None = None,
    server_location: str = "Germany",
    output_path: str = "federation_round.gif",
    num_frames: int = 60,
    frame_duration_ms: int = 200,
) -> None:
    """Render an animated GIF of one federation round.

    Parameters
    ----------
    hardware_config : dict
        Hardware config dict keyed by "client_0", "client_1", …
    timings : dict | None
        Per-client timing dict keyed by client key, each containing
        "download_time", "data_load_time", "train_time", "upload_time".
        If None, placeholder values are generated from network speeds.
    server_location : str
        Country name for the server.
    output_path : str
        Where to save the GIF.
    num_frames : int
        Number of animation frames.
    frame_duration_ms : int
        Delay between frames in milliseconds.
    """
    from PIL import Image

    server_lon, server_lat = _random_point(server_location)

    # Build VisualClient objects
    clients: list[VisualClient] = []
    for client_key, profile in hardware_config.items():
        location = profile.get("location", "Germany")
        speeds = _load_location_speeds(location)
        c_lon, c_lat = _random_point(location)

        arc_lons, arc_lats = _great_circle_path(c_lon, c_lat, server_lon, server_lat)

        vc = VisualClient(
            key=client_key,
            profile=profile,
            lon=c_lon, lat=c_lat,
            speeds=speeds,
            arc_lons=arc_lons, arc_lats=arc_lats,
        )

        # Set phase durations from real timings or estimate from speeds
        if timings and client_key in timings:
            ct = timings[client_key]
            vc.download_time = ct.get("download_time", 1.0)
            vc.load_time     = ct.get("data_load_time", 1.0)
            vc.train_time    = ct.get("train_time", 5.0)
            vc.upload_time   = ct.get("upload_time", 1.0)
        else:
            # Rough placeholders based on a ~50 MB model
            model_mb = 50.0
            vc.download_time = (model_mb * 8) / speeds["download_mbps"]
            vc.load_time     = 1.0
            vc.train_time    = random.uniform(3.0, 10.0)
            vc.upload_time   = (model_mb * 8) / speeds["upload_mbps"]

        clients.append(vc)

    t_max = max(vc.total_time for vc in clients)
    time_steps = np.linspace(0, t_max, num_frames)

    # Render frames
    frames: list[Image.Image] = []
    for i, t in enumerate(time_steps):
        fig = _render_frame(clients, server_lon, server_lat,
                            server_location, t, t_max)

        # Rasterize figure to PIL Image
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        w, h = fig.canvas.get_width_height()
        img = Image.frombytes("RGBA", (w, h), buf).convert("RGB")
        frames.append(img)
        plt.close(fig)

        if (i + 1) % 10 == 0:
            print(f"[visualize_gif] rendered frame {i + 1}/{num_frames}")

    # Save GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
    )
    print(f"[visualize_gif] saved to {output_path}")


# ---------------------------------------------------------------------------
# Flower aggregation hook — returns a closure for train_metrics_aggr_fn
# ---------------------------------------------------------------------------

def make_train_metrics_aggr_fn(
    hardware_config: dict,
    server_location: str = "Germany",
    num_frames: int = 60,
    frame_duration_ms: int = 200,
):
    """Return a train_metrics_aggr_fn compatible with FedAvg.

    The returned function:
      1. Extracts per-client timings from the RecordDicts.
      2. Generates a GIF for the round.
      3. Performs the standard weighted-average aggregation and returns
         the aggregated MetricRecord.

    Parameters
    ----------
    hardware_config : dict
        Hardware config dict keyed by "client_0", "client_1", …
    server_location : str
        Country name for the server.
    """
    from flwr.common import MetricRecord as FlowerMetricRecord

    # Keep track of round number across calls
    round_counter = [0]

    # Sort client keys so index matches Flower's ordering
    client_keys = sorted(hardware_config.keys(),
                         key=lambda k: int(k.split("_")[1]))

    def _aggr_fn(records: list, weighting_metric_name: str):
        round_counter[0] += 1
        server_round = round_counter[0]

        # --- Extract per-client timings ---
        timings: dict[str, dict] = {}
        for i, record in enumerate(records):
            client_key = client_keys[i] if i < len(client_keys) else f"client_{i}"
            mr = next(iter(record.metric_records.values()))
            timings[client_key] = {
                "download_time":  float(mr.get("download_time",  -1.0)),
                "data_load_time": float(mr.get("data_load_time", -1.0)),
                "train_time":     float(mr.get("train_time",     -1.0)),
                "upload_time":    float(mr.get("upload_time",    -1.0)),
            }

        # --- Generate round GIF ---
        try:
            from pathlib import Path
            Path("plots").mkdir(exist_ok=True)
            output_path = f"plots/round_{server_round}.gif"
            visualize_gif(
                hardware_config,
                timings=timings,
                server_location=server_location,
                output_path=output_path,
                num_frames=num_frames,
                frame_duration_ms=frame_duration_ms,
            )
        except Exception as e:
            print(f"[visualize_gif] round {server_round} GIF failed: {e}")

        # --- Standard weighted-average aggregation (replicate default) ---
        weights: list[float] = []
        for record in records:
            mr = next(iter(record.metric_records.values()))
            weights.append(float(mr[weighting_metric_name]))

        total_weight = sum(weights)
        if total_weight == 0:
            total_weight = 1.0
        weight_factors = [w / total_weight for w in weights]

        aggregated = FlowerMetricRecord()
        for record, wf in zip(records, weight_factors):
            mr = next(iter(record.metric_records.values()))
            for key, value in mr.items():
                if key == weighting_metric_name:
                    continue
                if key not in aggregated:
                    aggregated[key] = value * wf
                else:
                    aggregated[key] = float(aggregated[key]) + value * wf

        return aggregated

    return _aggr_fn
