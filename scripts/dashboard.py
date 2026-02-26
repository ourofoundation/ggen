#!/usr/bin/env python3
"""
Interactive Plotly dashboard for the ggen database.

Generates a standalone HTML dashboard with:
  - Database overview stats (total structures, sources, systems)
  - Pie chart of structures by source (ggen / mp / alexandria)
  - Space group distribution
  - Energy distribution histograms
  - Discoveries analysis (if available)
  - Per-system breakdown

Usage:
    python scripts/dashboard.py                     # Default db, all stats
    python scripts/dashboard.py --db path/to/ggen.db
    python scripts/dashboard.py --discoveries       # Include discoveries panel
    python scripts/dashboard.py --near-hull 0.050   # Near-hull cutoff for discoveries
    python scripts/dashboard.py -o my_dashboard.html
"""

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import plotly.graph_objects as go

from ggen import Colors, StructureDatabase

SOURCE_COLORS = {
    "ggen": "#00b4d8",
    "mp": "#e63946",
    "alexandria": "#2a9d8f",
}

CATEGORY_COLORS = {
    "new_composition": "#06d6a0",
    "beat_mp": "#ffd166",
    "elemental": "#8d99ae",
}

PHONON_COLORS = {
    "stable": "#06d6a0",
    "unstable": "#ef476f",
    "untested": "#8d99ae",
}

PLOTLY_TEMPLATE = "plotly_dark"
BG_COLOR = "#0e1117"
CARD_COLOR = "#1a1d23"
GRID_COLOR = "#2a2d35"
TEXT_COLOR = "#e0e0e0"
ACCENT = "#00b4d8"


# ── Database queries ────────────────────────────────────────────────


def query_source_counts(conn: sqlite3.Connection) -> Dict[str, int]:
    rows = conn.execute(
        "SELECT source, COUNT(*) FROM structures WHERE is_valid = 1 GROUP BY source"
    ).fetchall()
    return {row[0]: row[1] for row in rows}


def query_total_structures(conn: sqlite3.Connection) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM structures WHERE is_valid = 1"
    ).fetchone()[0]


def query_total_formulas(conn: sqlite3.Connection) -> int:
    return conn.execute(
        "SELECT COUNT(DISTINCT formula) FROM structures WHERE is_valid = 1"
    ).fetchone()[0]


def query_explored_systems(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute(
        "SELECT DISTINCT chemical_system FROM runs ORDER BY chemical_system"
    ).fetchall()
    return [row[0] for row in rows]


def query_space_group_distribution(conn: sqlite3.Connection) -> Dict[str, int]:
    rows = conn.execute(
        """SELECT COALESCE(space_group_symbol, 'Unknown') as sg, COUNT(*)
           FROM structures WHERE is_valid = 1
           GROUP BY sg ORDER BY COUNT(*) DESC"""
    ).fetchall()
    return {row[0]: row[1] for row in rows}


def query_energy_distribution(conn: sqlite3.Connection) -> List[Tuple[float, str]]:
    rows = conn.execute(
        """SELECT energy_per_atom, source FROM structures
           WHERE is_valid = 1 AND energy_per_atom IS NOT NULL"""
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def query_phonon_status(conn: sqlite3.Connection) -> Dict[str, int]:
    rows = conn.execute(
        """SELECT
             SUM(CASE WHEN is_dynamically_stable = 1 THEN 1 ELSE 0 END) as stable,
             SUM(CASE WHEN is_dynamically_stable = 0 THEN 1 ELSE 0 END) as unstable,
             SUM(CASE WHEN is_dynamically_stable IS NULL THEN 1 ELSE 0 END) as untested
           FROM structures WHERE is_valid = 1"""
    ).fetchone()
    return {"stable": rows[0] or 0, "unstable": rows[1] or 0, "untested": rows[2] or 0}


def query_structures_per_system(conn: sqlite3.Connection) -> Dict[str, Dict[str, int]]:
    """Count structures per chemsys, broken down by source."""
    rows = conn.execute(
        """SELECT chemsys, source, COUNT(*) FROM structures
           WHERE is_valid = 1 GROUP BY chemsys, source
           ORDER BY chemsys"""
    ).fetchall()
    result: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for chemsys, source, count in rows:
        result[chemsys][source] = count
    return dict(result)


def query_num_atoms_distribution(conn: sqlite3.Connection) -> List[Tuple[int, str]]:
    rows = conn.execute(
        """SELECT num_atoms, source FROM structures
           WHERE is_valid = 1 AND num_atoms IS NOT NULL"""
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def query_crystal_system_distribution(conn: sqlite3.Connection) -> Dict[str, int]:
    """Map space group numbers to crystal systems and count."""
    rows = conn.execute(
        """SELECT space_group_number, COUNT(*) FROM structures
           WHERE is_valid = 1 AND space_group_number IS NOT NULL
           GROUP BY space_group_number"""
    ).fetchall()
    counts: Dict[str, int] = Counter()
    for sg_num, count in rows:
        counts[_sg_to_crystal_system(sg_num)] += count
    return dict(counts)


def _sg_to_crystal_system(sg: int) -> str:
    if sg <= 2:
        return "Triclinic"
    elif sg <= 15:
        return "Monoclinic"
    elif sg <= 74:
        return "Orthorhombic"
    elif sg <= 142:
        return "Tetragonal"
    elif sg <= 167:
        return "Trigonal"
    elif sg <= 194:
        return "Hexagonal"
    else:
        return "Cubic"


def query_runs_timeline(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """SELECT chemical_system, started_at, completed_at,
                  num_candidates, num_successful
           FROM runs WHERE started_at IS NOT NULL
           ORDER BY started_at"""
    ).fetchall()
    return [
        {
            "system": row[0],
            "started": row[1],
            "completed": row[2],
            "candidates": row[3] or 0,
            "successful": row[4] or 0,
        }
        for row in rows
    ]


# ── Discoveries (reuses logic from discoveries.py) ─────────────────


def run_discoveries_analysis(
    db: StructureDatabase, near_hull_cutoff: float = 0.0
) -> Optional[Dict[str, Any]]:
    """Run the discoveries analysis and return the report dict."""
    try:
        from scripts.discoveries import (
            analyze_system,
            deduplicate_discoveries,
            build_report,
        )
    except ImportError:
        from discoveries import analyze_system, deduplicate_discoveries, build_report

    systems = db.list_explored_systems()
    if not systems:
        return None

    all_disc = []
    for chemsys in systems:
        disc = analyze_system(db, chemsys, near_hull_cutoff=near_hull_cutoff)
        all_disc.append(disc)

    unique = deduplicate_discoveries(all_disc)
    return build_report(all_disc, unique, near_hull_cutoff)


# ── Figure builders ─────────────────────────────────────────────────


def _base_layout(**overrides) -> Dict[str, Any]:
    layout = dict(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR, family="Inter, system-ui, sans-serif"),
        margin=dict(l=50, r=30, t=50, b=50),
    )
    layout.update(overrides)
    return layout


def fig_source_pie(source_counts: Dict[str, int]) -> go.Figure:
    labels = list(source_counts.keys())
    values = list(source_counts.values())
    colors = [SOURCE_COLORS.get(s, "#666") for s in labels]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors, line=dict(color=BG_COLOR, width=2)),
            textinfo="label+value+percent",
            textfont=dict(size=13),
            hovertemplate="<b>%{label}</b><br>%{value:,} structures<br>%{percent}<extra></extra>",
            hole=0.45,
        )
    )
    fig.update_layout(
        **_base_layout(
            title=dict(text="Structures by Source", font=dict(size=16)),
            showlegend=False,
            height=370,
        )
    )
    return fig


def fig_crystal_system_bar(crystal_counts: Dict[str, int]) -> go.Figure:
    order = [
        "Triclinic", "Monoclinic", "Orthorhombic",
        "Tetragonal", "Trigonal", "Hexagonal", "Cubic",
    ]
    labels = [c for c in order if c in crystal_counts]
    values = [crystal_counts[c] for c in labels]
    cs_colors = [
        "#ef476f", "#ffd166", "#06d6a0", "#118ab2",
        "#073b4c", "#8338ec", "#ff006e",
    ]
    colors = cs_colors[: len(labels)]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker=dict(color=colors, line=dict(color=BG_COLOR, width=1)),
            hovertemplate="<b>%{x}</b><br>%{y:,} structures<extra></extra>",
        )
    )
    fig.update_layout(
        **_base_layout(
            title=dict(text="Crystal System Distribution", font=dict(size=16)),
            xaxis=dict(gridcolor=GRID_COLOR),
            yaxis=dict(gridcolor=GRID_COLOR, title="Count"),
            height=370,
        )
    )
    return fig


def fig_space_group_top(sg_counts: Dict[str, int], top_n: int = 20) -> go.Figure:
    sorted_sg = sorted(sg_counts.items(), key=lambda x: -x[1])[:top_n]
    labels = [s[0] for s in sorted_sg][::-1]
    values = [s[1] for s in sorted_sg][::-1]

    fig = go.Figure(
        go.Bar(
            y=labels,
            x=values,
            orientation="h",
            marker=dict(color=ACCENT, line=dict(color=BG_COLOR, width=1)),
            hovertemplate="<b>%{y}</b><br>%{x:,} structures<extra></extra>",
        )
    )
    fig.update_layout(
        **_base_layout(
            title=dict(text=f"Top {top_n} Space Groups", font=dict(size=16)),
            xaxis=dict(gridcolor=GRID_COLOR, title="Count"),
            yaxis=dict(gridcolor=GRID_COLOR, dtick=1),
            height=max(370, top_n * 22),
        )
    )
    return fig


def fig_energy_histogram(energy_data: List[Tuple[float, str]]) -> go.Figure:
    by_source: Dict[str, List[float]] = defaultdict(list)
    for e, src in energy_data:
        by_source[src].append(e)

    fig = go.Figure()
    for src in ["ggen", "mp", "alexandria"]:
        if src not in by_source:
            continue
        fig.add_trace(
            go.Histogram(
                x=by_source[src],
                name=src,
                marker_color=SOURCE_COLORS.get(src, "#666"),
                opacity=0.75,
                hovertemplate=f"<b>{src}</b><br>Energy: %{{x:.3f}} eV/atom<br>Count: %{{y}}<extra></extra>",
            )
        )

    fig.update_layout(
        **_base_layout(
            title=dict(text="Energy per Atom Distribution", font=dict(size=16)),
            xaxis=dict(gridcolor=GRID_COLOR, title="Energy per atom (eV)"),
            yaxis=dict(gridcolor=GRID_COLOR, title="Count"),
            barmode="overlay",
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0)"),
            height=370,
        )
    )
    return fig


def fig_atoms_histogram(atoms_data: List[Tuple[int, str]]) -> go.Figure:
    by_source: Dict[str, List[int]] = defaultdict(list)
    for n, src in atoms_data:
        by_source[src].append(n)

    fig = go.Figure()
    for src in ["ggen", "mp", "alexandria"]:
        if src not in by_source:
            continue
        fig.add_trace(
            go.Histogram(
                x=by_source[src],
                name=src,
                marker_color=SOURCE_COLORS.get(src, "#666"),
                opacity=0.75,
                hovertemplate=f"<b>{src}</b><br>Atoms: %{{x}}<br>Count: %{{y}}<extra></extra>",
            )
        )

    fig.update_layout(
        **_base_layout(
            title=dict(text="Atoms per Structure", font=dict(size=16)),
            xaxis=dict(gridcolor=GRID_COLOR, title="Number of atoms"),
            yaxis=dict(gridcolor=GRID_COLOR, title="Count"),
            barmode="overlay",
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0)"),
            height=370,
        )
    )
    return fig


def fig_phonon_pie(phonon_counts: Dict[str, int]) -> go.Figure:
    labels = list(phonon_counts.keys())
    values = list(phonon_counts.values())
    colors = [PHONON_COLORS.get(l, "#666") for l in labels]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors, line=dict(color=BG_COLOR, width=2)),
            textinfo="label+value+percent",
            textfont=dict(size=13),
            hovertemplate="<b>%{label}</b><br>%{value:,} structures<br>%{percent}<extra></extra>",
            hole=0.45,
        )
    )
    fig.update_layout(
        **_base_layout(
            title=dict(text="Phonon Stability Status", font=dict(size=16)),
            showlegend=False,
            height=370,
        )
    )
    return fig


def fig_system_bar(sys_counts: Dict[str, Dict[str, int]], top_n: int = 25) -> go.Figure:
    totals = {k: sum(v.values()) for k, v in sys_counts.items()}
    top_systems = sorted(totals, key=lambda k: -totals[k])[:top_n]

    fig = go.Figure()
    for src in ["ggen", "mp", "alexandria"]:
        vals = [sys_counts[s].get(src, 0) for s in top_systems]
        if not any(vals):
            continue
        fig.add_trace(
            go.Bar(
                x=top_systems,
                y=vals,
                name=src,
                marker_color=SOURCE_COLORS.get(src, "#666"),
                hovertemplate=f"<b>{src}</b><br>%{{x}}<br>%{{y:,}} structures<extra></extra>",
            )
        )

    fig.update_layout(
        **_base_layout(
            title=dict(text=f"Structures per System (top {min(top_n, len(top_systems))})", font=dict(size=16)),
            xaxis=dict(gridcolor=GRID_COLOR, tickangle=-45),
            yaxis=dict(gridcolor=GRID_COLOR, title="Count"),
            barmode="stack",
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0)"),
            height=400,
        )
    )
    return fig


# ── Discoveries figures ─────────────────────────────────────────────


def fig_discovery_categories(report: Dict[str, Any]) -> go.Figure:
    summary = report["summary"]
    labels = ["New Compositions", "Beat MP"]
    values = [summary["new_compositions"], summary["beat_mp"]]
    colors = [CATEGORY_COLORS["new_composition"], CATEGORY_COLORS["beat_mp"]]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors, line=dict(color=BG_COLOR, width=2)),
            textinfo="label+value+percent",
            textfont=dict(size=13),
            hovertemplate="<b>%{label}</b><br>%{value} discoveries<br>%{percent}<extra></extra>",
            hole=0.45,
        )
    )
    fig.update_layout(
        **_base_layout(
            title=dict(text="Discovery Categories", font=dict(size=16)),
            showlegend=False,
            height=370,
        )
    )
    return fig


def fig_discovery_phonon(report: Dict[str, Any]) -> go.Figure:
    summary = report["summary"]
    labels = ["Phonon Stable", "Phonon Unstable", "Untested"]
    values = [summary["phonon_stable"], summary["phonon_unstable"], summary["phonon_untested"]]
    colors = [PHONON_COLORS["stable"], PHONON_COLORS["unstable"], PHONON_COLORS["untested"]]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors, line=dict(color=BG_COLOR, width=2)),
            textinfo="label+value+percent",
            textfont=dict(size=13),
            hovertemplate="<b>%{label}</b><br>%{value} discoveries<br>%{percent}<extra></extra>",
            hole=0.45,
        )
    )
    fig.update_layout(
        **_base_layout(
            title=dict(text="Discovery Phonon Status", font=dict(size=16)),
            showlegend=False,
            height=370,
        )
    )
    return fig


def fig_discovery_energy_gains(report: Dict[str, Any]) -> go.Figure:
    beat_entries = [
        d for d in report["unique_discoveries"] if d["category"] == "beat_mp" and d["energy_gain_eV"] is not None
    ]
    if not beat_entries:
        fig = go.Figure()
        fig.update_layout(**_base_layout(
            title=dict(text="Energy Gain over MP (meV/atom)", font=dict(size=16)),
            height=370,
            annotations=[dict(text="No beat-MP discoveries", showarrow=False, font=dict(size=14, color=TEXT_COLOR))],
        ))
        return fig

    beat_entries.sort(key=lambda d: -d["energy_gain_eV"])
    formulas = [d["formula"] for d in beat_entries]
    gains = [d["energy_gain_eV"] * 1000 for d in beat_entries]

    fig = go.Figure(
        go.Bar(
            x=formulas,
            y=gains,
            marker=dict(
                color=gains,
                colorscale=[[0, "#ffd166"], [1, "#ef476f"]],
                line=dict(color=BG_COLOR, width=1),
            ),
            hovertemplate="<b>%{x}</b><br>%{y:.1f} meV/atom improvement<extra></extra>",
        )
    )
    fig.update_layout(
        **_base_layout(
            title=dict(text="Energy Gain over MP", font=dict(size=16)),
            xaxis=dict(gridcolor=GRID_COLOR, tickangle=-45),
            yaxis=dict(gridcolor=GRID_COLOR, title="Improvement (meV/atom)"),
            height=400,
        )
    )
    return fig


def fig_discovery_table(report: Dict[str, Any]) -> go.Figure:
    discoveries = report["unique_discoveries"]
    if not discoveries:
        fig = go.Figure()
        fig.update_layout(**_base_layout(height=200))
        return fig

    formulas = [d["formula"] for d in discoveries]
    space_groups = [d["space_group"] or "?" for d in discoveries]
    energies = [f"{d['energy_per_atom']:.4f}" for d in discoveries]
    categories = [d["category"].replace("_", " ").title() for d in discoveries]
    gains = [
        f"{d['energy_gain_eV'] * 1000:.1f}" if d["energy_gain_eV"] is not None else "—"
        for d in discoveries
    ]
    phonon = []
    for d in discoveries:
        if d["is_dynamically_stable"] is True:
            phonon.append("Stable")
        elif d["is_dynamically_stable"] is False:
            phonon.append("Unstable")
        else:
            phonon.append("—")
    systems = [", ".join(d["systems"][:2]) + ("..." if len(d["systems"]) > 2 else "") for d in discoveries]

    cat_colors = []
    for d in discoveries:
        if d["category"] == "new_composition":
            cat_colors.append(CATEGORY_COLORS["new_composition"])
        elif d["category"] == "beat_mp":
            cat_colors.append(CATEGORY_COLORS["beat_mp"])
        else:
            cat_colors.append("#8d99ae")

    phonon_colors = []
    for d in discoveries:
        if d["is_dynamically_stable"] is True:
            phonon_colors.append(PHONON_COLORS["stable"])
        elif d["is_dynamically_stable"] is False:
            phonon_colors.append(PHONON_COLORS["unstable"])
        else:
            phonon_colors.append("#555")

    fig = go.Figure(
        go.Table(
            header=dict(
                values=["Formula", "Space Group", "E (eV/atom)", "Category", "Gain (meV)", "Phonon", "System"],
                fill_color="#1e2130",
                font=dict(color=TEXT_COLOR, size=12),
                align="left",
                line=dict(color=GRID_COLOR, width=1),
                height=32,
            ),
            cells=dict(
                values=[formulas, space_groups, energies, categories, gains, phonon, systems],
                fill_color=[
                    [CARD_COLOR] * len(formulas),
                    [CARD_COLOR] * len(formulas),
                    [CARD_COLOR] * len(formulas),
                    [[f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.2)" for c in cat_colors]],
                    [CARD_COLOR] * len(formulas),
                    [[f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.2)" for c in phonon_colors]],
                    [CARD_COLOR] * len(formulas),
                ],
                font=dict(color=TEXT_COLOR, size=11),
                align="left",
                line=dict(color=GRID_COLOR, width=1),
                height=28,
            ),
        )
    )
    fig.update_layout(
        **_base_layout(
            title=dict(text="Unique Discoveries", font=dict(size=16)),
            height=max(300, 60 + len(discoveries) * 30),
        )
    )
    return fig


def fig_per_system_discoveries(report: Dict[str, Any]) -> go.Figure:
    per_sys = report["per_system"]
    systems = sorted(per_sys.keys())
    new_comp = [per_sys[s]["new_compositions"] for s in systems]
    beat_mp = [per_sys[s]["beat_mp"] for s in systems]

    has_data = any(n > 0 for n in new_comp) or any(n > 0 for n in beat_mp)
    if not has_data:
        fig = go.Figure()
        fig.update_layout(**_base_layout(
            title=dict(text="Discoveries per System", font=dict(size=16)),
            height=370,
            annotations=[dict(text="No discoveries", showarrow=False, font=dict(size=14, color=TEXT_COLOR))],
        ))
        return fig

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=systems, y=new_comp, name="New Composition",
        marker_color=CATEGORY_COLORS["new_composition"],
        hovertemplate="<b>%{x}</b><br>%{y} new compositions<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=systems, y=beat_mp, name="Beat MP",
        marker_color=CATEGORY_COLORS["beat_mp"],
        hovertemplate="<b>%{x}</b><br>%{y} beat MP<extra></extra>",
    ))
    fig.update_layout(
        **_base_layout(
            title=dict(text="Discoveries per System", font=dict(size=16)),
            xaxis=dict(gridcolor=GRID_COLOR, tickangle=-45),
            yaxis=dict(gridcolor=GRID_COLOR, title="Count"),
            barmode="stack",
            legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0)"),
            height=400,
        )
    )
    return fig


# ── HTML assembly ───────────────────────────────────────────────────


def _stat_card(label: str, value: str, color: str = ACCENT) -> str:
    return f"""
    <div style="background:{CARD_COLOR}; border-radius:10px; padding:20px 24px;
                flex:1; min-width:160px; border-left:3px solid {color};">
      <div style="font-size:13px; color:#8d99ae; text-transform:uppercase;
                  letter-spacing:1px; margin-bottom:6px;">{label}</div>
      <div style="font-size:28px; font-weight:700; color:{color};">{value}</div>
    </div>"""


def build_dashboard_html(
    conn: sqlite3.Connection,
    db: StructureDatabase,
    include_discoveries: bool = False,
    near_hull_cutoff: float = 0.0,
) -> str:
    """Build the full dashboard HTML string."""

    # Gather data
    source_counts = query_source_counts(conn)
    total_structures = query_total_structures(conn)
    total_formulas = query_total_formulas(conn)
    explored_systems = query_explored_systems(conn)
    sg_dist = query_space_group_distribution(conn)
    energy_data = query_energy_distribution(conn)
    atoms_data = query_num_atoms_distribution(conn)
    phonon_counts = query_phonon_status(conn)
    crystal_counts = query_crystal_system_distribution(conn)
    sys_counts = query_structures_per_system(conn)

    # Stat cards
    stat_cards = "".join([
        _stat_card("Total Structures", f"{total_structures:,}"),
        _stat_card("Unique Formulas", f"{total_formulas:,}", "#06d6a0"),
        _stat_card("Systems Explored", str(len(explored_systems)), "#ffd166"),
        _stat_card("Sources", str(len(source_counts)), "#e63946"),
        _stat_card("Phonon Tested", f"{phonon_counts['stable'] + phonon_counts['unstable']:,}", "#8338ec"),
    ])

    # Build figures
    figures: List[Tuple[str, go.Figure]] = [
        ("source_pie", fig_source_pie(source_counts)),
        ("crystal_bar", fig_crystal_system_bar(crystal_counts)),
        ("energy_hist", fig_energy_histogram(energy_data)),
        ("atoms_hist", fig_atoms_histogram(atoms_data)),
        ("phonon_pie", fig_phonon_pie(phonon_counts)),
        ("sg_top", fig_space_group_top(sg_dist)),
        ("system_bar", fig_system_bar(sys_counts)),
    ]

    # Discoveries
    disc_stat_cards = ""
    if include_discoveries:
        report = run_discoveries_analysis(db, near_hull_cutoff)
        if report and report["summary"]["unique_discoveries"] > 0:
            s = report["summary"]
            disc_stat_cards = "".join([
                _stat_card("Unique Discoveries", str(s["unique_discoveries"]), "#06d6a0"),
                _stat_card("New Compositions", str(s["new_compositions"]), CATEGORY_COLORS["new_composition"]),
                _stat_card("Beat MP", str(s["beat_mp"]), CATEGORY_COLORS["beat_mp"]),
                _stat_card("Phonon Stable", str(s["phonon_stable"]), PHONON_COLORS["stable"]),
                _stat_card("Higher Symmetry", str(s["higher_symmetry"]), "#8338ec"),
            ])
            figures.extend([
                ("disc_categories", fig_discovery_categories(report)),
                ("disc_phonon", fig_discovery_phonon(report)),
                ("disc_energy_gains", fig_discovery_energy_gains(report)),
                ("disc_per_system", fig_per_system_discoveries(report)),
                ("disc_table", fig_discovery_table(report)),
            ])

    # Assemble HTML
    return _assemble_html(figures, stat_cards, disc_stat_cards)


def _assemble_html(
    figures: List[Tuple[str, go.Figure]],
    stat_cards_html: str,
    disc_stat_cards_html: str = "",
) -> str:
    plotly_js = "https://cdn.plot.ly/plotly-2.35.2.min.js"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Split figures into db overview and discoveries
    db_figs = [(name, fig) for name, fig in figures if not name.startswith("disc_")]
    disc_figs = [(name, fig) for name, fig in figures if name.startswith("disc_")]

    def render_figures_safe(figs: List[Tuple[str, go.Figure]], prefix: str) -> Tuple[str, str]:
        divs = []
        scripts = []
        for i, (name, fig) in enumerate(figs):
            div_id = f"{prefix}-{i}"
            fig_dict = fig.to_plotly_json()
            data_json = json.dumps(fig_dict.get("data", []))
            layout_json = json.dumps(fig_dict.get("layout", {}))
            wide = "chart-card-wide" if name in ("system_bar", "disc_table", "disc_energy_gains", "disc_per_system") else ""
            divs.append(f'<div class="chart-card {wide}" id="{div_id}"></div>')
            scripts.append(
                f"Plotly.newPlot('{div_id}', {data_json}, {layout_json}, "
                f"{{responsive: true, displayModeBar: false}});"
            )
        return "\n    ".join(divs), "\n    ".join(scripts)

    db_divs, db_scripts = render_figures_safe(db_figs, "db")
    disc_divs, disc_scripts = render_figures_safe(disc_figs, "disc")

    disc_section = ""
    if disc_figs:
        disc_section = f"""
  <div class="section-header">Discoveries</div>
  <div class="stat-row">{disc_stat_cards_html}</div>
  <div class="chart-grid">
    {disc_divs}
  </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>ggen Dashboard</title>
  <script src="{plotly_js}"></script>
  <style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{
      background: {BG_COLOR};
      color: {TEXT_COLOR};
      font-family: Inter, system-ui, -apple-system, sans-serif;
      padding: 24px 32px;
      min-height: 100vh;
    }}
    h1 {{
      font-size: 26px;
      font-weight: 700;
      margin-bottom: 4px;
      color: #fff;
    }}
    .subtitle {{
      color: #8d99ae;
      font-size: 13px;
      margin-bottom: 24px;
    }}
    .stat-row {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      margin-bottom: 24px;
    }}
    .section-header {{
      font-size: 18px;
      font-weight: 600;
      color: #fff;
      margin: 32px 0 16px 0;
      padding-bottom: 8px;
      border-bottom: 1px solid {GRID_COLOR};
    }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
      gap: 16px;
      margin-bottom: 16px;
    }}
    .chart-card {{
      background: {CARD_COLOR};
      border-radius: 10px;
      padding: 12px;
      overflow: hidden;
    }}
    .chart-card-wide {{
      grid-column: 1 / -1;
    }}
  </style>
</head>
<body>
  <h1>ggen Database Dashboard</h1>
  <div class="subtitle">Generated {now}</div>

  <div class="stat-row">{stat_cards_html}</div>

  <div class="section-header">Database Overview</div>
  <div class="chart-grid">
    {db_divs}
  </div>

  {disc_section}

  <script>
    {db_scripts}
    {disc_scripts}
  </script>
</body>
</html>"""


# ── CLI ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate an interactive Plotly dashboard for the ggen database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Basic DB stats dashboard
  %(prog)s --discoveries                # Include discoveries panel
  %(prog)s --near-hull 0.050            # Near-hull cutoff for discoveries
  %(prog)s -o my_dashboard.html         # Custom output path
        """,
    )

    parser.add_argument(
        "--db", "-d",
        default="./ggen.db",
        help="Path to ggen database (default: ./ggen.db)",
    )

    parser.add_argument(
        "--discoveries",
        action="store_true",
        help="Include discoveries analysis panel",
    )

    parser.add_argument(
        "--near-hull",
        type=float,
        default=0.0,
        metavar="EV",
        help="Near-hull cutoff for discoveries (eV/atom)",
    )

    parser.add_argument(
        "-o", "--output",
        default="dashboard.html",
        metavar="FILE",
        help="Output HTML file (default: dashboard.html)",
    )

    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored terminal output",
    )

    args = parser.parse_args()

    if args.no_color or not sys.stdout.isatty():
        Colors.disable()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"{Colors.RED}Error:{Colors.RESET} Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    C = Colors
    print(f"{C.BOLD}Generating dashboard...{C.RESET}")

    db = StructureDatabase(str(db_path))
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        html = build_dashboard_html(
            conn=conn,
            db=db,
            include_discoveries=args.discoveries,
            near_hull_cutoff=args.near_hull,
        )

        out_path = Path(args.output)
        out_path.write_text(html)

        total = query_total_structures(conn)
        sources = query_source_counts(conn)
        source_str = ", ".join(f"{s}: {c:,}" for s, c in sorted(sources.items()))

        print(f"\n  {C.CYAN}Structures:{C.RESET} {total:,} ({source_str})")
        print(f"  {C.CYAN}Systems:{C.RESET}    {len(query_explored_systems(conn))}")
        if args.discoveries:
            print(f"  {C.CYAN}Discoveries:{C.RESET} included")
        print(f"\n{C.GREEN}Dashboard written to {out_path}{C.RESET}")

    finally:
        conn.close()
        db.close()


if __name__ == "__main__":
    main()
