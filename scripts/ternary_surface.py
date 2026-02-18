#!/usr/bin/env python3
"""Generate a static Three.js ternary surface from ggen database data."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Add parent to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

if TYPE_CHECKING:
    from ggen import SystemExplorer


SQRT3_2 = float(np.sqrt(3.0) / 2.0)


@dataclass
class InterpolationResult:
    energy: float
    sigma: Optional[float] = None


def ternary_to_xy(a: float, b: float, c: float) -> Tuple[float, float]:
    """Convert ternary fractions to 2D cartesian coordinates."""
    x = b + 0.5 * c
    y = c * SQRT3_2
    return x, y


def energy_to_rgb(energy: float, e_min: float, e_max: float) -> Tuple[float, float, float]:
    """Match the frontend color ramp used in experiments/ternary-surface.jsx."""
    if e_max <= e_min:
        return 0.2, 0.7, 1.0

    t = max(0.0, min(1.0, (energy - e_min) / (e_max - e_min)))
    if t < 0.25:
        s = t / 0.25
        r = 0.08
        g = 0.15 + 0.45 * s
        b = 0.7 + 0.25 * s
    elif t < 0.5:
        s = (t - 0.25) / 0.25
        r = 0.08 + 0.12 * s
        g = 0.6 + 0.35 * s
        b = 0.95 - 0.6 * s
    elif t < 0.75:
        s = (t - 0.5) / 0.25
        r = 0.2 + 0.7 * s
        g = 0.95 - 0.1 * s
        b = 0.35 - 0.2 * s
    else:
        s = (t - 0.75) / 0.25
        r = 0.9 + 0.1 * s
        g = 0.85 - 0.55 * s
        b = 0.15 - 0.05 * s
    return r, g, b


def gather_phase_points(
    explorer: "SystemExplorer",
    chemical_system: str,
    tested_only: bool,
    all_polymorphs: bool,
    exclude_space_groups: Optional[Sequence[str]],
    max_e_above_hull_ev: Optional[float],
) -> Tuple[List[Dict[str, float]], Dict[str, int]]:
    """
    Gather ternary composition points and e_above_hull values.

    Returns a list of points in the form:
      {"a": ..., "b": ..., "c": ..., "energy": ...}
    """
    from pymatgen.core import Composition

    pd, e_above_hull_map, filter_counts = explorer.get_phase_diagram(
        chemical_system,
        update_database=True,
        exclude_space_groups=list(exclude_space_groups) if exclude_space_groups else None,
        tested_only=tested_only,
        all_polymorphs=all_polymorphs,
    )
    if pd is None:
        return [], filter_counts

    chemsys = explorer.db.normalize_chemsys(chemical_system)
    elements = chemsys.split("-")
    if len(elements) != 3:
        raise ValueError(
            f"Ternary surface requires exactly 3 elements, got {len(elements)} in {chemsys}"
        )

    if all_polymorphs:
        structures = explorer.db.get_structures_for_subsystem(chemsys, valid_only=True)
    else:
        structures = list(explorer.db.get_best_structures_for_subsystem(chemsys).values())

    exclude_set = set(exclude_space_groups or [])
    if exclude_set:
        structures = [s for s in structures if s.space_group_symbol not in exclude_set]
    if tested_only:
        structures = [s for s in structures if s.is_dynamically_stable is not None]

    # Keep the lowest-energy entry per exact composition triplet.
    by_comp: Dict[Tuple[float, float, float], Dict[str, object]] = {}
    excluded_by_hull = 0
    for s in structures:
        e_hull = e_above_hull_map.get(s.id)
        if e_hull is None:
            continue
        if max_e_above_hull_ev is not None and e_hull > max_e_above_hull_ev:
            excluded_by_hull += 1
            continue

        comp = Composition(s.formula)
        frac = comp.fractional_composition
        a = float(frac.get(elements[0], 0.0))
        b = float(frac.get(elements[1], 0.0))
        c = float(frac.get(elements[2], 0.0))

        key = (round(a, 6), round(b, 6), round(c, 6))
        prev = by_comp.get(key)
        if prev is None or e_hull < float(prev["energy"]):
            by_comp[key] = {"energy": float(e_hull), "formula": s.formula}

    points = [
        {
            "a": a,
            "b": b,
            "c": c,
            "energy": float(meta["energy"]),
            "formula": str(meta["formula"]),
        }
        for (a, b, c), meta in by_comp.items()
    ]
    if excluded_by_hull:
        filter_counts["excluded_above_hull_cutoff"] = excluded_by_hull
    return points, filter_counts


def idw_energy(
    target_xyz: np.ndarray,
    known_xyz: np.ndarray,
    known_energy: np.ndarray,
    power: float = 2.0,
    k: int = 8,
) -> float:
    """Inverse-distance interpolation in ternary composition space."""
    distances = np.linalg.norm(known_xyz - target_xyz, axis=1)
    exact = np.where(distances < 1e-10)[0]
    if exact.size > 0:
        return float(known_energy[exact[0]])

    if k > 0 and len(distances) > k:
        nearest = np.argpartition(distances, k)[:k]
        distances = distances[nearest]
        energies = known_energy[nearest]
    else:
        energies = known_energy

    weights = 1.0 / np.power(distances + 1e-12, power)
    return float(np.sum(weights * energies) / np.sum(weights))


class IDWInterpolator:
    """Inverse-distance weighted interpolator in ternary composition space."""

    def __init__(self, known_xyz: np.ndarray, known_energy: np.ndarray, power: float, k: int):
        self.known_xyz = known_xyz
        self.known_energy = known_energy
        self.power = power
        self.k = k

    def predict(self, target_xyz: np.ndarray) -> InterpolationResult:
        return InterpolationResult(
            energy=idw_energy(
                target_xyz=target_xyz,
                known_xyz=self.known_xyz,
                known_energy=self.known_energy,
                power=self.power,
                k=self.k,
            )
        )


class GPInterpolator:
    """Gaussian Process interpolator on reduced barycentric coordinates (a, b)."""

    def __init__(
        self,
        known_xyz: np.ndarray,
        known_energy: np.ndarray,
        restarts: int,
        noise_floor: float,
    ):
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        except ImportError as exc:
            raise RuntimeError(
                "GP interpolation requires scikit-learn. Install with: pip install scikit-learn"
            ) from exc

        x_train = known_xyz[:, :2]
        y_train = known_energy
        kernel = ConstantKernel(1.0, (1e-4, 1e4)) * Matern(
            length_scale=0.2, length_scale_bounds=(1e-3, 10.0), nu=2.5
        ) + WhiteKernel(noise_level=noise_floor, noise_level_bounds=(1e-10, 1e-1))
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=max(noise_floor, 1e-12),
            normalize_y=True,
            n_restarts_optimizer=max(0, restarts),
            random_state=0,
        )
        self.gp.fit(x_train, y_train)

    def predict(self, target_xyz: np.ndarray) -> InterpolationResult:
        x = np.array([[target_xyz[0], target_xyz[1]]], dtype=float)
        mean, std = self.gp.predict(x, return_std=True)
        return InterpolationResult(energy=float(mean[0]), sigma=float(std[0]))


def build_convex_hull_classifier(known_xyz: np.ndarray):
    """Return a callable(point_xyz) -> bool for inside/edge of convex hull in composition space."""
    from scipy.spatial import ConvexHull, Delaunay, QhullError

    xy = known_xyz[:, :2]
    unique_xy = np.unique(np.round(xy, 10), axis=0)
    if unique_xy.shape[0] < 3:
        return None
    try:
        hull = ConvexHull(unique_xy)
        hull_xy = unique_xy[hull.vertices]
        tri = Delaunay(hull_xy)
    except QhullError:
        return None

    def is_inside(target_xyz: np.ndarray) -> bool:
        query = np.array([[target_xyz[0], target_xyz[1]]], dtype=float)
        return bool(tri.find_simplex(query)[0] >= 0)

    return is_inside


def create_interpolator(
    method: str,
    known_xyz: np.ndarray,
    known_energy: np.ndarray,
    idw_power: float,
    idw_k: int,
    gp_restarts: int,
    gp_noise_floor: float,
):
    if method == "idw":
        return IDWInterpolator(
            known_xyz=known_xyz,
            known_energy=known_energy,
            power=idw_power,
            k=idw_k,
        )
    if method == "gp":
        return GPInterpolator(
            known_xyz=known_xyz,
            known_energy=known_energy,
            restarts=gp_restarts,
            noise_floor=gp_noise_floor,
        )
    raise ValueError(f"Unknown interpolation method: {method}")


def build_mesh(
    points: List[Dict[str, float]],
    resolution: int,
    interpolator,
    mask_to_convex_hull: bool,
) -> Dict[str, object]:
    """Build a regular ternary grid mesh interpolated from phase points."""
    if not points:
        raise ValueError("No points available to build surface")

    vertices: List[float] = []
    indices: List[int] = []
    colors: List[float] = []
    ternary_coords: List[List[float]] = []
    energies: List[float] = []
    uncertainty: List[Optional[float]] = []
    xy_coords: List[Tuple[float, float]] = []
    index_map: Dict[Tuple[int, int], int] = {}
    known_xyz = np.array([[p["a"], p["b"], p["c"]] for p in points], dtype=float)
    in_hull = build_convex_hull_classifier(known_xyz) if mask_to_convex_hull else None

    step = 1.0 / float(resolution)
    idx = 0
    for i in range(resolution + 1):
        for j in range(resolution - i + 1):
            a = i * step
            b = j * step
            c = 1.0 - a - b
            if c < -1e-8:
                continue
            txyz = np.array([a, b, c], dtype=float)
            if in_hull is not None and not in_hull(txyz):
                continue

            pred = interpolator.predict(txyz)
            e = pred.energy
            x, y = ternary_to_xy(a, b, c)
            xy_coords.append((x, y))
            ternary_coords.append([a, b, c])
            energies.append(e)
            uncertainty.append(pred.sigma)
            index_map[(i, j)] = idx
            idx += 1

    if not energies:
        raise RuntimeError("Interpolation generated no mesh vertices after filtering")

    e_min = float(min(energies))
    e_max = float(max(energies))
    e_center = 0.5 * (e_min + e_max)
    e_span = max(e_max - e_min, 1e-12)
    display_y = [(e - e_center) / e_span for e in energies]
    y_min = float(min(display_y))
    y_max = float(max(display_y))

    for (x, y), y_disp in zip(xy_coords, display_y):
        vertices.extend([x - 0.5, y_disp, y - SQRT3_2 / 2.0])

    for e in energies:
        r, g, b = energy_to_rgb(e, e_min, e_max)
        colors.extend([r, g, b])

    for i in range(resolution):
        for j in range(resolution - i):
            v0 = index_map.get((i, j))
            v1 = index_map.get((i, j + 1))
            v2 = index_map.get((i + 1, j))
            if v0 is not None and v1 is not None and v2 is not None:
                indices.extend([v0, v1, v2])

            if j < resolution - i - 1:
                v3 = index_map.get((i + 1, j + 1))
                if v1 is not None and v2 is not None and v3 is not None:
                    indices.extend([v1, v3, v2])

    sigma_vals = [s for s in uncertainty if s is not None]
    return {
        "vertices": vertices,
        "indices": indices,
        "colors": colors,
        "ternaryCoords": ternary_coords,
        "energies": energies,
        "uncertainty": uncertainty,
        "eMin": e_min,
        "eMax": e_max,
        "sigmaMin": float(min(sigma_vals)) if sigma_vals else None,
        "sigmaMax": float(max(sigma_vals)) if sigma_vals else None,
        "yMin": y_min,
        "yMax": y_max,
    }


def render_html_plotly(payload: Dict[str, object]) -> str:
    """Render a fully self-contained Plotly HTML (no CDN dependency)."""
    import plotly.graph_objects as go

    data = payload["data"]
    mesh = data["mesh"]
    elements = data["elements"]
    src_points = data["sourcePoints"]

    vertices = mesh["vertices"]
    x = vertices[0::3]
    z = vertices[1::3]  # normalized display height
    y = vertices[2::3]
    energies = mesh["energies"]
    idx = mesh["indices"]
    i = idx[0::3]
    j = idx[1::3]
    k = idx[2::3]

    e_min = float(mesh["eMin"])
    e_max = float(mesh["eMax"])
    e_center = 0.5 * (e_min + e_max)
    e_span = max(e_max - e_min, 1e-12)
    z_raw = energies
    z_norm = z

    p_xyz = [ternary_to_xy(p["a"], p["b"], p["c"]) for p in src_points]
    p_x = [xy[0] - 0.5 for xy in p_xyz]
    p_y = [xy[1] - SQRT3_2 / 2.0 for xy in p_xyz]
    p_z_raw = [p["energy"] for p in src_points]
    p_z_norm = [((e - e_center) / e_span) for e in p_z_raw]
    p_hover = [
        f"{elements[0]}={p['a']:.3f}<br>{elements[1]}={p['b']:.3f}<br>{elements[2]}={p['c']:.3f}<br>E_hull={p['energy']:.4f} eV/atom"
        for p in src_points
    ]

    colorscale = [
        [0.0, "rgb(20,38,178)"],
        [0.25, "rgb(20,153,242)"],
        [0.5, "rgb(51,242,89)"],
        [0.75, "rgb(230,217,51)"],
        [1.0, "rgb(230,51,32)"],
    ]

    trace_surface_norm = go.Mesh3d(
        x=x,
        y=y,
        z=z_norm,
        i=i,
        j=j,
        k=k,
        intensity=energies,
        colorscale=colorscale,
        cmin=e_min,
        cmax=e_max,
        opacity=0.96,
        name="surface (normalized)",
        showscale=True,
        colorbar=dict(title="E_hull (eV/atom)"),
        hovertemplate=f"{elements[0]}=%{{customdata[0]:.3f}}<br>"
        + f"{elements[1]}=%{{customdata[1]:.3f}}<br>"
        + f"{elements[2]}=%{{customdata[2]:.3f}}<br>"
        + "E_hull=%{intensity:.4f} eV/atom<extra></extra>",
        customdata=mesh["ternaryCoords"],
        visible=True,
    )
    trace_surface_raw = go.Mesh3d(
        x=x,
        y=y,
        z=z_raw,
        i=i,
        j=j,
        k=k,
        intensity=energies,
        colorscale=colorscale,
        cmin=e_min,
        cmax=e_max,
        opacity=0.96,
        name="surface (raw)",
        showscale=False,
        hovertemplate=trace_surface_norm.hovertemplate,
        customdata=mesh["ternaryCoords"],
        visible=False,
    )
    trace_points_norm = go.Scatter3d(
        x=p_x,
        y=p_y,
        z=p_z_norm,
        mode="markers",
        marker=dict(size=2.8, color=p_z_raw, colorscale=colorscale, cmin=e_min, cmax=e_max),
        name="source points",
        text=p_hover,
        hovertemplate="%{text}<extra></extra>",
        visible=True,
    )
    trace_points_raw = go.Scatter3d(
        x=p_x,
        y=p_y,
        z=p_z_raw,
        mode="markers",
        marker=dict(size=2.8, color=p_z_raw, colorscale=colorscale, cmin=e_min, cmax=e_max),
        name="source points (raw)",
        text=p_hover,
        hovertemplate="%{text}<extra></extra>",
        visible=False,
    )

    fig = go.Figure(
        data=[trace_surface_norm, trace_surface_raw, trace_points_norm, trace_points_raw]
    )
    fig.update_layout(
        template="plotly_dark",
        title={
            "text": f"{payload['title']} ternary manifold<br><sup>{payload['meta']}</sup>",
            "x": 0.5,
        },
        paper_bgcolor="#14141e",
        plot_bgcolor="#14141e",
        scene=dict(
            xaxis=dict(
                title="",
                showticklabels=False,
                showbackground=False,
                zeroline=False,
                visible=False,
            ),
            yaxis=dict(
                title="",
                showticklabels=False,
                showbackground=False,
                zeroline=False,
                visible=False,
            ),
            zaxis=dict(
                title="Normalized height",
                backgroundcolor="#14141e",
                gridcolor="#2b3245",
                zerolinecolor="#2b3245",
            ),
            aspectmode="data",
            camera=dict(eye=dict(x=1.45, y=1.35, z=0.95)),
            annotations=[
                dict(x=-0.5, y=-SQRT3_2 / 2.0, z=-0.58, text=elements[0], showarrow=False),
                dict(x=0.5, y=-SQRT3_2 / 2.0, z=-0.58, text=elements[1], showarrow=False),
                dict(x=0.0, y=SQRT3_2 / 2.0, z=-0.58, text=elements[2], showarrow=False),
            ],
        ),
        margin=dict(l=8, r=8, b=8, t=56),
        legend=dict(y=0.99, x=0.01),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.01,
                y=1.12,
                buttons=[
                    dict(
                        label="Normalized + Points",
                        method="update",
                        args=[
                            {"visible": [True, False, True, False]},
                            {"scene.zaxis.title.text": "Normalized height"},
                        ],
                    ),
                    dict(
                        label="Normalized (No Points)",
                        method="update",
                        args=[
                            {"visible": [True, False, False, False]},
                            {"scene.zaxis.title.text": "Normalized height"},
                        ],
                    ),
                    dict(
                        label="Raw + Points",
                        method="update",
                        args=[
                            {"visible": [False, True, False, True]},
                            {"scene.zaxis.title.text": "E_hull (eV/atom)"},
                        ],
                    ),
                    dict(
                        label="Raw (No Points)",
                        method="update",
                        args=[
                            {"visible": [False, True, False, False]},
                            {"scene.zaxis.title.text": "E_hull (eV/atom)"},
                        ],
                    ),
                ],
            )
        ],
    )

    return fig.to_html(full_html=True, include_plotlyjs=True, config={"responsive": True})


def render_html_three(payload: Dict[str, object]) -> str:
    """Render a Three.js HTML with height/point toggles."""
    template = Template(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Ternary Surface - $title</title>
  <style>
    html, body { margin: 0; height: 100%; background: #14141e; color: #b0b8cc; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    #app { height: 100%; display: flex; flex-direction: column; }
    #header { flex: 0 0 auto; display: flex; justify-content: space-between; align-items: center; padding: 12px 16px; border-bottom: 1px solid #1a1a2e; gap: 12px; }
    #title { font-size: 14px; font-weight: 700; color: #d7deef; }
    #subtitle { font-size: 11px; color: #63708d; }
    #canvas-wrap { position: relative; flex: 1 1 auto; min-height: 0; }
    #view { width: 100%; height: 100%; display: block; }
    .panel { position: absolute; background: rgba(18,18,30,.9); border: 1px solid #1a1a2e; border-radius: 6px; padding: 10px 12px; }
    #controls { left: 16px; bottom: 16px; min-width: 250px; }
    #info { right: 16px; top: 16px; max-width: 340px; font-size: 11px; color: #93a0bd; line-height: 1.35; }
    .row { margin-top: 8px; display: flex; align-items: center; gap: 8px; }
    .lbl { width: 74px; color: #7d8caf; }
    button { font: inherit; font-size: 10px; background: transparent; color: #b4c1de; border: 1px solid #2b3b56; border-radius: 4px; padding: 3px 8px; cursor: pointer; }
    button.active { background: #1a2a4a; color: #6aafff; border-color: #2a4a7a; }
    #status { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #f1b06a; background: rgba(20,20,30,.92); border: 1px solid #5a3a30; border-radius: 6px; padding: 10px 12px; display: none; max-width: 60vw; }
    #hover { position: absolute; pointer-events: none; display: none; z-index: 9; max-width: 260px; color: #d8e2ff; background: rgba(12,14,22,.94); border: 1px solid #2b3b56; border-radius: 6px; padding: 8px 9px; font-size: 10px; line-height: 1.35; white-space: normal; }
  </style>
</head>
<body>
  <div id="app">
    <div id="header">
      <div>
        <div id="title">TERNARY ENERGY SURFACE (THREE.JS)</div>
        <div id="subtitle">$subtitle</div>
      </div>
      <div style="font-size:10px;color:#63708d;">$meta</div>
    </div>
    <div id="canvas-wrap">
      <canvas id="view"></canvas>
      <div id="controls" class="panel">
        <div style="font-size:10px;color:#6e7d9f;">Drag: orbit | Right-drag: pan | Scroll: zoom</div>
        <div class="row"><span class="lbl">Height</span><button id="modeNorm">Normalized</button><button id="modeRaw" class="active">Raw</button></div>
        <div class="row"><span class="lbl">Points</span><button id="ptsToggle" class="active">On</button></div>
        <div class="row"><span class="lbl">Wireframe</span><button id="wireToggle">Off</button></div>
        <div class="row"><span class="lbl">Z Scale</span><input id="zscale" type="range" min="0.2" max="3" step="0.1" value="1.0" /><span id="zval">1.0x</span></div>
      </div>
      <div id="info" class="panel">
        <div style="font-size:13px;font-weight:700;color:#d0d4e4;">$title</div>
        <div style="margin-top:6px;">Surface = $interp_desc interpolation of E_hull from filtered ggen entries.</div>
        <div style="margin-top:8px;">Points: <span id="npts"></span> (<span id="nhull"></span> hull)<br/>Energy range: <span id="erange"></span> eV/atom</div>
      </div>
      <div id="status"></div>
      <div id="hover"></div>
    </div>
  </div>
  <script type="module">
    const DATA = $data_json;
    const statusEl = document.getElementById("status");
    function fail(msg) { statusEl.style.display = "block"; statusEl.textContent = msg; }
    try {
      async function loadThree() {
        const urls = [
          "https://cdn.jsdelivr.net/npm/three@0.162.0/build/three.module.js",
          "https://unpkg.com/three@0.162.0/build/three.module.js",
          "https://esm.sh/three@0.162.0"
        ];
        let lastErr = null;
        for (const url of urls) {
          try {
            const mod = await import(url);
            if (mod) return mod;
          } catch (err) {
            lastErr = err;
          }
        }
        throw lastErr || new Error("Three.js module load failed from all CDNs.");
      }
      const THREE = await loadThree();
      const ELS = DATA.elements;
      document.getElementById("npts").textContent = String(DATA.sourcePoints.length);
      document.getElementById("erange").textContent = DATA.mesh.eMin.toFixed(4) + " to " + DATA.mesh.eMax.toFixed(4);
      const hullMask = DATA.sourcePoints.map((p) => p.energy <= 1e-6);
      const hullCount = hullMask.reduce((acc, isHull) => acc + (isHull ? 1 : 0), 0);
      document.getElementById("nhull").textContent = String(hullCount);
      const hoverEl = document.getElementById("hover");

      const canvas = document.getElementById("view");
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x14141e);
      const camera = new THREE.PerspectiveCamera(45, 2, 0.01, 100);
      const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

      scene.add(new THREE.AmbientLight(0x8090b0, 1.0));
      const d1 = new THREE.DirectionalLight(0xffffff, 1.1); d1.position.set(2, 3, 1); scene.add(d1);
      const d2 = new THREE.DirectionalLight(0x88aaff, 0.45); d2.position.set(-2, 2, -1); scene.add(d2);

      const group = new THREE.Group(); scene.add(group);
      const meshData = DATA.mesh;
      const baseVerts = new Float32Array(meshData.vertices);
      const energies = meshData.energies;
      const eCenter = 0.5 * (meshData.eMin + meshData.eMax);
      const eSpan = Math.max(meshData.eMax - meshData.eMin, 1e-12);
      const positions = new Float32Array(baseVerts.length);
      const colors = new Float32Array(meshData.colors);
      positions.set(baseVerts);

      let heightMode = "raw";
      let showPoints = true;
      let wireframe = false;
      let zScale = 1.0;
      let hoveredSourceIdx = -1;

      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      geo.setIndex(meshData.indices);
      geo.computeVertexNormals();
      const mat = new THREE.MeshPhongMaterial({
        vertexColors: true,
        side: THREE.DoubleSide,
        shininess: 60,
        wireframe: false,
        transparent: true,
        opacity: 0.96,
        polygonOffset: true,
        polygonOffsetFactor: 1,
        polygonOffsetUnits: 1
      });
      const mesh = new THREE.Mesh(geo, mat);
      group.add(mesh);
      let axisGroup = null;

      const pPos = new Float32Array(DATA.sourcePoints.length * 3);
      const pCol = new Float32Array(DATA.sourcePoints.length * 3);
      const hullPos = new Float32Array(Math.max(hullCount, 1) * 3);
      const hullCol = new Float32Array(Math.max(hullCount, 1) * 3);
      const hullSourceIdx = [];
      let hIdx = 0;
      for (let i = 0; i < DATA.sourcePoints.length; i++) {
        const p = DATA.sourcePoints[i];
        const x = p.b + 0.5 * p.c;
        const y = p.c * 0.8660254037844386;
        pPos[3*i] = x - 0.5;
        pPos[3*i+1] = ((p.energy - eCenter) / eSpan);
        pPos[3*i+2] = y - 0.4330127018922193;
        const idx = i * 3;
        if (hullMask[i]) {
          // Slightly bright non-hull overlay still includes these.
          pCol[idx] = 0.98; pCol[idx+1] = 0.92; pCol[idx+2] = 0.62;
          hullPos[3*hIdx] = pPos[3*i];
          hullPos[3*hIdx+1] = pPos[3*i+1];
          hullPos[3*hIdx+2] = pPos[3*i+2];
          hullCol[3*hIdx] = 1.0; hullCol[3*hIdx+1] = 0.82; hullCol[3*hIdx+2] = 0.22;
          hullSourceIdx.push(i);
          hIdx += 1;
        } else {
          pCol[idx] = 0.80; pCol[idx+1] = 0.85; pCol[idx+2] = 0.95;
        }
      }
      const pGeo = new THREE.BufferGeometry();
      pGeo.setAttribute("position", new THREE.BufferAttribute(pPos, 3));
      pGeo.setAttribute("color", new THREE.BufferAttribute(pCol, 3));
      const pMat = new THREE.PointsMaterial({
        size: 0.009,
        vertexColors: true,
        transparent: true,
        opacity: 0.9,
        depthTest: true,
        depthWrite: false
      });
      const points = new THREE.Points(pGeo, pMat);
      group.add(points);
      const hullGeo = new THREE.BufferGeometry();
      hullGeo.setAttribute("position", new THREE.BufferAttribute(hullPos, 3));
      hullGeo.setAttribute("color", new THREE.BufferAttribute(hullCol, 3));
      hullGeo.setDrawRange(0, hullCount);
      const hullMat = new THREE.PointsMaterial({
        size: 0.024,
        vertexColors: true,
        transparent: true,
        opacity: 0.98,
        sizeAttenuation: true,
        depthTest: true,
        depthWrite: false
      });
      const hullPoints = new THREE.Points(hullGeo, hullMat);
      group.add(hullPoints);
      const hoverMarker = new THREE.Mesh(
        new THREE.SphereGeometry(0.0045, 12, 12),
        new THREE.MeshBasicMaterial({
          color: 0x8fe7ff,
          transparent: true,
          opacity: 0.95,
          depthTest: false,
          depthWrite: false
        })
      );
      hoverMarker.renderOrder = 12;
      hoverMarker.visible = false;
      group.add(hoverMarker);

      const raycaster = new THREE.Raycaster();
      raycaster.params.Points.threshold = 0.02;
      const mouse = new THREE.Vector2(2, 2);

      function ternaryText(a, b, c) {
        return ELS[0] + "=" + (a * 100).toFixed(1) + "%, "
          + ELS[1] + "=" + (b * 100).toFixed(1) + "%, "
          + ELS[2] + "=" + (c * 100).toFixed(1) + "%";
      }

      function makeLabelSprite(text, color = "#a9bbdf", size = 30) {
        const c = document.createElement("canvas");
        c.width = 256; c.height = 96;
        const ctx = c.getContext("2d");
        ctx.clearRect(0, 0, c.width, c.height);
        ctx.font = "bold " + size + "px monospace";
        ctx.fillStyle = color;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(text, c.width / 2, c.height / 2);
        const tex = new THREE.CanvasTexture(c);
        return new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true }));
      }

      function toXY(a, b, c) {
        return { x: b + 0.5 * c - 0.5, z: c * 0.8660254037844386 - 0.4330127018922193 };
      }

      function addSeg(arr, p1, p2) {
        arr.push(p1.x, p1.y, p1.z, p2.x, p2.y, p2.z);
      }
      function showHoverMarker(sourceIdx) {
        if (sourceIdx < 0 || sourceIdx >= DATA.sourcePoints.length) return;
        hoveredSourceIdx = sourceIdx;
        hoverMarker.visible = true;
        hoverMarker.position.set(pPos[3*sourceIdx], pPos[3*sourceIdx+1], pPos[3*sourceIdx+2]);
        hoverMarker.material.color.setHex(hullMask[sourceIdx] ? 0xffcf4a : 0x8fe7ff);
      }
      function clearHoverMarker() {
        hoveredSourceIdx = -1;
        hoverMarker.visible = false;
      }

      let target = new THREE.Vector3(0, 0, 0);
      let spherical = { theta: 0.6, phi: 1.0, radius: 1.7 };
      function updateCamera() {
        const t = spherical.theta, p = spherical.phi, r = spherical.radius;
        camera.position.set(target.x + r * Math.sin(p) * Math.cos(t), target.y + r * Math.cos(p), target.z + r * Math.sin(p) * Math.sin(t));
        camera.lookAt(target);
      }
      function fitCamera() {
        geo.computeBoundingBox();
        const bb = geo.boundingBox;
        const cx = 0.5 * (bb.min.x + bb.max.x);
        const cy = 0.5 * (bb.min.y + bb.max.y);
        const cz = 0.5 * (bb.min.z + bb.max.z);
        const sx = bb.max.x - bb.min.x, sy = bb.max.y - bb.min.y, sz = bb.max.z - bb.min.z;
        target.set(cx, cy, cz);
        spherical.radius = Math.max(1.4, Math.sqrt(sx*sx + sy*sy + sz*sz) * 1.6);
        updateCamera();
      }

      function applyHeightMode(refitCamera = false) {
        for (let vi = 0; vi < energies.length; vi++) {
          const yNorm = baseVerts[3 * vi + 1];
          const yRaw = energies[vi];
          positions[3 * vi + 1] = (heightMode === "raw" ? yRaw : yNorm) * zScale;
        }
        geo.attributes.position.needsUpdate = true;
        geo.computeVertexNormals();
        for (let i = 0; i < DATA.sourcePoints.length; i++) {
          const p = DATA.sourcePoints[i];
          const yNorm = (p.energy - eCenter) / eSpan;
          pPos[3*i+1] = (heightMode === "raw" ? p.energy : yNorm) * zScale;
        }
        pGeo.attributes.position.needsUpdate = true;
        if (hullCount > 0) {
          let hi = 0;
          for (let i = 0; i < DATA.sourcePoints.length; i++) {
            if (!hullMask[i]) continue;
            const p = DATA.sourcePoints[i];
            const yNorm = (p.energy - eCenter) / eSpan;
            hullPos[3*hi+1] = (heightMode === "raw" ? p.energy : yNorm) * zScale;
            hi += 1;
          }
          hullGeo.attributes.position.needsUpdate = true;
        }
        if (hoveredSourceIdx >= 0) {
          hoverMarker.position.set(
            pPos[3*hoveredSourceIdx],
            pPos[3*hoveredSourceIdx+1],
            pPos[3*hoveredSourceIdx+2]
          );
        }
        points.visible = showPoints;
        hullPoints.visible = showPoints;
        hoverMarker.visible = showPoints && hoveredSourceIdx >= 0;
        mat.wireframe = wireframe;

        if (axisGroup) group.remove(axisGroup);
        axisGroup = new THREE.Group();
        const minY = (() => {
          let v = Infinity;
          for (let i = 1; i < positions.length; i += 3) v = Math.min(v, positions[i]);
          return v;
        })();
        const floorY = minY - 0.03;

        // Base triangle edges
        const c0 = toXY(1, 0, 0);
        const c1 = toXY(0, 1, 0);
        const c2 = toXY(0, 0, 1);
        const edgeLines = [];
        addSeg(edgeLines, { x: c0.x, y: floorY, z: c0.z }, { x: c1.x, y: floorY, z: c1.z });
        addSeg(edgeLines, { x: c1.x, y: floorY, z: c1.z }, { x: c2.x, y: floorY, z: c2.z });
        addSeg(edgeLines, { x: c2.x, y: floorY, z: c2.z }, { x: c0.x, y: floorY, z: c0.z });
        const edgeGeo = new THREE.BufferGeometry();
        edgeGeo.setAttribute("position", new THREE.Float32BufferAttribute(edgeLines, 3));
        axisGroup.add(new THREE.LineSegments(edgeGeo, new THREE.LineBasicMaterial({ color: 0x6f86ad, transparent: true, opacity: 0.9 })));

        // Ternary guide lines at 20/40/60/80%
        const gridLines = [];
        for (const t of [0.2, 0.4, 0.6, 0.8]) {
          // A = t
          let p1 = toXY(t, 1 - t, 0), p2 = toXY(t, 0, 1 - t);
          addSeg(gridLines, { x: p1.x, y: floorY, z: p1.z }, { x: p2.x, y: floorY, z: p2.z });
          // B = t
          p1 = toXY(1 - t, t, 0); p2 = toXY(0, t, 1 - t);
          addSeg(gridLines, { x: p1.x, y: floorY, z: p1.z }, { x: p2.x, y: floorY, z: p2.z });
          // C = t
          p1 = toXY(1 - t, 0, t); p2 = toXY(0, 1 - t, t);
          addSeg(gridLines, { x: p1.x, y: floorY, z: p1.z }, { x: p2.x, y: floorY, z: p2.z });
        }
        const gridGeo = new THREE.BufferGeometry();
        gridGeo.setAttribute("position", new THREE.Float32BufferAttribute(gridLines, 3));
        axisGroup.add(new THREE.LineSegments(gridGeo, new THREE.LineBasicMaterial({ color: 0x33425f, transparent: true, opacity: 0.55 })));

        // Corner labels
        const labA = makeLabelSprite(ELS[0]);
        const labB = makeLabelSprite(ELS[1]);
        const labC = makeLabelSprite(ELS[2]);
        labA.position.set(c0.x - 0.03, floorY - 0.03, c0.z - 0.04);
        labB.position.set(c1.x + 0.04, floorY - 0.03, c1.z - 0.01);
        labC.position.set(c2.x, floorY - 0.03, c2.z + 0.05);
        labA.scale.set(0.2, 0.075, 1);
        labB.scale.set(0.2, 0.075, 1);
        labC.scale.set(0.2, 0.075, 1);
        axisGroup.add(labA); axisGroup.add(labB); axisGroup.add(labC);
        group.add(axisGroup);

        if (refitCamera) {
          fitCamera();
        }
      }

      function resize() {
        const w = canvas.clientWidth, h = canvas.clientHeight;
        renderer.setSize(w, h, false);
        camera.aspect = Math.max(w / Math.max(h, 1), 0.1);
        camera.updateProjectionMatrix();
      }
      window.addEventListener("resize", resize);
      resize();

      let drag = false, pan = false, prev = { x: 0, y: 0 };
      function camDir() { const d = new THREE.Vector3(); camera.getWorldDirection(d); return d; }
      function camRight() { return new THREE.Vector3().crossVectors(camDir(), camera.up).normalize(); }
      function camUp() { const d = camDir(); const r = camRight(); return new THREE.Vector3().crossVectors(r, d).normalize(); }
      canvas.addEventListener("mousedown", (e) => { drag = !(e.button === 2 || e.shiftKey); pan = !drag; prev = { x: e.clientX, y: e.clientY }; });
      canvas.addEventListener("mousemove", (e) => {
        const dx = e.clientX - prev.x, dy = e.clientY - prev.y; prev = { x: e.clientX, y: e.clientY };
        if (drag) { spherical.theta += dx * 0.004; spherical.phi = Math.max(0.15, Math.min(Math.PI - 0.15, spherical.phi - dy * 0.004)); updateCamera(); }
        else if (pan) { const speed = 0.0015 * spherical.radius; target.add(camRight().multiplyScalar(-dx * speed).add(camUp().multiplyScalar(dy * speed))); updateCamera(); }
      });
      window.addEventListener("mouseup", () => { drag = false; pan = false; });
      canvas.addEventListener("contextmenu", (e) => e.preventDefault());
      canvas.addEventListener("wheel", (e) => { e.preventDefault(); const factor = 1 + e.deltaY * 0.001; spherical.radius = Math.max(0.3, Math.min(8, spherical.radius * factor)); updateCamera(); }, { passive: false });
      canvas.addEventListener("mousemove", (e) => {
        const rect = canvas.getBoundingClientRect();
        mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
        // Keep picking radius roughly screen-consistent across zoom.
        raycaster.params.Points.threshold = Math.max(0.004, 0.006 * spherical.radius);
        raycaster.setFromCamera(mouse, camera);
        const meshHits = raycaster.intersectObject(mesh, false);
        const meshDist = meshHits.length > 0 ? meshHits[0].distance : Infinity;
        const depthSlack = Math.max(0.002, 0.003 * spherical.radius);

        // Prefer exact source points for hover, then fallback to surface
        const pHit = showPoints ? raycaster.intersectObject(points, false) : [];
        const hHit = showPoints ? raycaster.intersectObject(hullPoints, false) : [];
        const hVisible = hHit.length > 0 ? hHit.find((hit) => hit.distance <= meshDist + depthSlack) : null;
        if (hVisible) {
          // Map hull point index back to source point
          const hLocal = hVisible.index;
          const srcIdx = hullSourceIdx[hLocal];
          if (srcIdx >= 0) {
            const p = DATA.sourcePoints[srcIdx];
            showHoverMarker(srcIdx);
            hoverEl.style.display = "block";
            hoverEl.style.left = (e.clientX - rect.left + 12) + "px";
            hoverEl.style.top = (e.clientY - rect.top + 12) + "px";
            hoverEl.innerHTML = "<b>Hull Point (On-Hull)</b><br>Formula: " + (p.formula || "?")
              + "<br>" + ternaryText(p.a, p.b, p.c)
              + "<br>E_hull: " + p.energy.toFixed(6) + " eV/atom";
            return;
          }
        }
        const pVisible = pHit.length > 0 ? pHit.find((hit) => hit.distance <= meshDist + depthSlack) : null;
        if (pVisible) {
          const idx = pVisible.index;
          const p = DATA.sourcePoints[idx];
          showHoverMarker(idx);
          hoverEl.style.display = "block";
          hoverEl.style.left = (e.clientX - rect.left + 12) + "px";
          hoverEl.style.top = (e.clientY - rect.top + 12) + "px";
          hoverEl.innerHTML = "<b>Source Point</b><br>Formula: " + (p.formula || "?")
            + "<br>" + ternaryText(p.a, p.b, p.c)
            + "<br>E_hull: " + p.energy.toFixed(4) + " eV/atom";
          return;
        }

        const mHit = meshHits;
        if (mHit.length > 0 && mHit[0].face) {
          clearHoverMarker();
          const face = mHit[0].face;
          const pnt = mHit[0].point;
          const ids = [face.a, face.b, face.c];
          let best = ids[0], bestD = Infinity;
          for (const vid of ids) {
            const dx = positions[3 * vid] - pnt.x;
            const dy = positions[3 * vid + 1] - pnt.y;
            const dz = positions[3 * vid + 2] - pnt.z;
            const d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < bestD) { bestD = d2; best = vid; }
          }
          const tri = meshData.ternaryCoords[best];
          const eHull = meshData.energies[best];
          hoverEl.style.display = "block";
          hoverEl.style.left = (e.clientX - rect.left + 12) + "px";
          hoverEl.style.top = (e.clientY - rect.top + 12) + "px";
          hoverEl.innerHTML = "<b>Interpolated Surface</b><br>" + ternaryText(tri[0], tri[1], tri[2])
            + "<br>E_hull: " + eHull.toFixed(4) + " eV/atom";
          return;
        }
        clearHoverMarker();
        hoverEl.style.display = "none";
      });
      canvas.addEventListener("mouseleave", () => { hoverEl.style.display = "none"; clearHoverMarker(); });

      const modeNorm = document.getElementById("modeNorm");
      const modeRaw = document.getElementById("modeRaw");
      const ptsBtn = document.getElementById("ptsToggle");
      const wireBtn = document.getElementById("wireToggle");
      modeNorm.onclick = () => { heightMode = "normalized"; modeNorm.classList.add("active"); modeRaw.classList.remove("active"); applyHeightMode(); };
      modeRaw.onclick = () => { heightMode = "raw"; modeRaw.classList.add("active"); modeNorm.classList.remove("active"); applyHeightMode(); };
      ptsBtn.onclick = () => { showPoints = !showPoints; ptsBtn.textContent = showPoints ? "On" : "Off"; ptsBtn.classList.toggle("active", showPoints); applyHeightMode(); };
      wireBtn.onclick = () => { wireframe = !wireframe; wireBtn.textContent = wireframe ? "On" : "Off"; wireBtn.classList.toggle("active", wireframe); applyHeightMode(); };
      document.getElementById("zscale").addEventListener("input", (e) => { zScale = Number(e.target.value); document.getElementById("zval").textContent = zScale.toFixed(1) + "x"; applyHeightMode(); });

      applyHeightMode(true);
      (function animate(){ requestAnimationFrame(animate); renderer.render(scene, camera); })();
    } catch (err) {
      fail("Renderer error: " + (err && err.message ? err.message : String(err)));
      console.error(err);
    }
  </script>
</body>
</html>
"""
    )
    return template.substitute(
        title=payload["title"],
        subtitle=payload["subtitle"],
        meta=payload["meta"],
        interp_desc=payload["interp_desc"],
        data_json=json.dumps(payload["data"]),
    )


def render_html(
    payload: Dict[str, object], renderer: str
) -> str:
    if renderer == "plotly":
        return render_html_plotly(payload)
    return render_html_three(payload)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a static ternary energy surface HTML from ggen data."
    )
    parser.add_argument("system", help="Chemical system, e.g. Co-Fe-Mn")
    parser.add_argument(
        "--db",
        "-d",
        default="./ggen.db",
        help="Path to ggen database (default: ./ggen.db)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output HTML path (default: <chemsys>_ternary_surface.html)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=60,
        help="Ternary grid resolution for the interpolated surface (default: 60)",
    )
    parser.add_argument(
        "--interpolation",
        choices=["idw", "gp"],
        default="idw",
        help="Interpolation algorithm for surface generation (default: idw)",
    )
    parser.add_argument(
        "--mask-convex-hull",
        action="store_true",
        help="Mask mesh to convex hull of available composition points",
    )
    parser.add_argument(
        "--idw-power",
        type=float,
        default=2.0,
        help="IDW distance power (default: 2.0)",
    )
    parser.add_argument(
        "--idw-k",
        type=int,
        default=8,
        help="IDW nearest-neighbor count (0 = all points, default: 8)",
    )
    parser.add_argument(
        "--gp-restarts",
        type=int,
        default=4,
        help="GP optimizer restarts for kernel hyperparameters (default: 4)",
    )
    parser.add_argument(
        "--gp-noise-floor",
        type=float,
        default=1e-6,
        help="GP minimum noise regularization (default: 1e-6)",
    )
    parser.add_argument(
        "--tested-only",
        action="store_true",
        help="Only include structures with phonon stability labels",
    )
    parser.add_argument(
        "--all-polymorphs",
        action="store_true",
        help="Include all polymorphs instead of lowest-energy structure per formula",
    )
    parser.add_argument(
        "--exclude-p1",
        action="store_true",
        help="Exclude P1 structures (same behavior as scripts/report.py)",
    )
    parser.add_argument(
        "--max-ehull-mev",
        type=float,
        default=150.0,
        help="Exclude structures above this E_hull threshold in meV/atom (default: 150)",
    )
    parser.add_argument(
        "--renderer",
        choices=["three", "plotly"],
        default="three",
        help="HTML renderer backend (default: three)",
    )
    args = parser.parse_args()

    from ggen.report import SystemExplorer

    if args.resolution < 10 or args.resolution > 200:
        raise ValueError("--resolution must be between 10 and 200")
    if args.max_ehull_mev < 0:
        raise ValueError("--max-ehull-mev must be >= 0")
    if args.idw_power <= 0:
        raise ValueError("--idw-power must be > 0")
    if args.idw_k < 0:
        raise ValueError("--idw-k must be >= 0")
    if args.gp_restarts < 0:
        raise ValueError("--gp-restarts must be >= 0")
    if args.gp_noise_floor < 0:
        raise ValueError("--gp-noise-floor must be >= 0")

    db_path = Path(args.db)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    with SystemExplorer(str(db_path)) as explorer:
        chemsys = explorer.db.normalize_chemsys(args.system)
        elements = chemsys.split("-")
        if len(elements) != 3:
            raise ValueError(f"Expected ternary system (3 elements), got: {chemsys}")

        exclude_sg = ["P1"] if args.exclude_p1 else None
        max_e_above_hull_ev = args.max_ehull_mev / 1000.0
        points, filter_counts = gather_phase_points(
            explorer=explorer,
            chemical_system=chemsys,
            tested_only=args.tested_only,
            all_polymorphs=args.all_polymorphs,
            exclude_space_groups=exclude_sg,
            max_e_above_hull_ev=max_e_above_hull_ev,
        )

    if not points:
        raise RuntimeError(
            f"No usable points found for {chemsys}. Check filters or run exploration first."
        )

    known_xyz = np.array([[p["a"], p["b"], p["c"]] for p in points], dtype=float)
    known_energy = np.array([p["energy"] for p in points], dtype=float)
    interpolator = create_interpolator(
        method=args.interpolation,
        known_xyz=known_xyz,
        known_energy=known_energy,
        idw_power=args.idw_power,
        idw_k=args.idw_k,
        gp_restarts=args.gp_restarts,
        gp_noise_floor=args.gp_noise_floor,
    )
    mesh = build_mesh(
        points,
        resolution=args.resolution,
        interpolator=interpolator,
        mask_to_convex_hull=args.mask_convex_hull,
    )
    output_path = (
        Path(args.output)
        if args.output
        else Path(f"{chemsys}_ternary_surface.html")
    )

    meta_bits = [
        f"{len(points)} source points",
        f"resolution {args.resolution}",
        f"E_hull <= {args.max_ehull_mev:.1f} meV",
        f"interp {args.interpolation}",
    ]
    if args.mask_convex_hull:
        meta_bits.append("masked to convex hull")
    if args.interpolation == "idw":
        meta_bits.append(f"idw(p={args.idw_power:.2f}, k={args.idw_k})")
    else:
        meta_bits.append(
            f"gp(restarts={args.gp_restarts}, noise={args.gp_noise_floor:.1e})"
        )
    if args.exclude_p1:
        meta_bits.append("exclude P1")
    if args.tested_only:
        meta_bits.append("tested only")
    if args.all_polymorphs:
        meta_bits.append("all polymorphs")
    for key, val in sorted(filter_counts.items()):
        meta_bits.append(f"{key}: {val}")

    payload = {
        "title": chemsys,
        "subtitle": f"{elements[0]} - {elements[1]} - {elements[2]}",
        "meta": " | ".join(meta_bits),
        "interp_desc": args.interpolation.upper(),
        "data": {
            "system": chemsys,
            "elements": elements,
            "sourcePoints": points,
            "mesh": mesh,
        },
    }
    output_path.write_text(
        render_html(payload, renderer=args.renderer),
        encoding="utf-8",
    )
    print(f"Saved ternary surface: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
