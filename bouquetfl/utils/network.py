"""
network.py — placeholder network emulation utilities.

Provides upload/download speed lookup, ping lookup, and model size estimation
for simulating realistic communication overhead in federated learning.

All values are placeholders — replace with real measurements or a
calibrated model as needed.
"""

import tomllib

_LOCATIONS_SCHEMA_VERSION  = 1
_PING_GRAPH_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Location speeds
# ---------------------------------------------------------------------------

def get_location_speeds(location: str) -> dict:
    """Return upload/download speeds (Mbit/s) for a named location.

    Parameters
    ----------
    location : str
        Location name matching an entry in datasets/locations.toml.

    Returns
    -------
    dict with keys "download_mbps" and "upload_mbps".
    """
    with open("datasets/locations.toml", "rb") as f:
        data = tomllib.load(f)
    v = data.get("schema_version", 0)
    if v != _LOCATIONS_SCHEMA_VERSION:
        raise RuntimeError(f"datasets/locations.toml schema_version={v}, expected {_LOCATIONS_SCHEMA_VERSION}")
    for entry in data["locations"]:
        if entry["name"] == location:
            return {
                "download_mbps": entry["download_mbps"],
                "upload_mbps":   entry["upload_mbps"],
            }
    raise ValueError(f"Location '{location}' not found in datasets/locations.toml.")


# ---------------------------------------------------------------------------
# Ping
# ---------------------------------------------------------------------------

def get_ping(location_a: str, location_b: str) -> float:
    """Return round-trip ping (ms) between two locations.

    Looks up the undirected edge (A, B) or (B, A) in datasets/ping_graph.toml.

    Parameters
    ----------
    location_a, location_b : str
        Location names matching entries in datasets/ping_graph.toml.

    Returns
    -------
    float — round-trip latency in milliseconds.
    """
    with open("datasets/ping_graph.toml", "rb") as f:
        data = tomllib.load(f)
    v = data.get("schema_version", 0)
    if v != _PING_GRAPH_SCHEMA_VERSION:
        raise RuntimeError(f"datasets/ping_graph.toml schema_version={v}, expected {_PING_GRAPH_SCHEMA_VERSION}")
    for edge in data["edges"]:
        if (edge["from"] == location_a and edge["to"] == location_b) or \
           (edge["from"] == location_b and edge["to"] == location_a):
            return float(edge["ping_ms"])
    raise ValueError(
        f"No ping entry found for ({location_a}, {location_b}) "
        "in datasets/ping_graph.toml."
    )


# ---------------------------------------------------------------------------
# Model size
# ---------------------------------------------------------------------------

def estimate_model_size_mb(state_dict: dict) -> float:
    """Estimate the size of a model state_dict in megabytes.

    Counts the total number of bytes across all tensors.

    Parameters
    ----------
    state_dict : dict
        PyTorch state_dict (mapping of str → Tensor).

    Returns
    -------
    float — size in MB.
    """
    import torch
    total_bytes = sum(
        t.numel() * t.element_size()
        for t in state_dict.values()
        if isinstance(t, torch.Tensor)
    )
    return total_bytes / (1024 ** 2)


# ---------------------------------------------------------------------------
# Transfer time
# ---------------------------------------------------------------------------

def estimate_transfer_time(size_mb: float, speed_mbps: float, ping_ms: float = 0.0) -> float:
    """Estimate transfer time in seconds given model size, link speed, and ping.

    Parameters
    ----------
    size_mb   : float — model size in MB.
    speed_mbps: float — link speed in Mbit/s.
    ping_ms   : float — one-way latency in ms (default 0).

    Returns
    -------
    float — estimated transfer time in seconds.
    """
    transfer_s = (size_mb * 8) / speed_mbps   # MB → Mbit, divide by Mbit/s
    latency_s  = ping_ms / 1000.0
    return transfer_s + latency_s
