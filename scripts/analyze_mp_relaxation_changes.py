#!/usr/bin/env python3
"""
Analyze how much MP structures change (space group and shape) after mLIP/ORB relaxation.

Uses the original CIF and space group stored in generation_metadata by relax.py
(--source mp --unrelaxed). Structures relaxed before that metadata was added are
excluded (no mp_original_* in metadata).

Usage (use pyenv ggen or your ggen env so ggen/orb_models are available):
    pyenv shell ggen
    python scripts/analyze_mp_relaxation_changes.py --database ggen.db
    python scripts/analyze_mp_relaxation_changes.py --database ggen.db --system Fe-O
    python scripts/analyze_mp_relaxation_changes.py --database ggen.db --csv out.csv
    python scripts/analyze_mp_relaxation_changes.py --database ggen.db --no-geometry
"""

import argparse
import csv
import heapq
import json
import math
import re
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from ggen import Colors, StructureDatabase


def _crystal_system_from_sg_number(sg_number: Optional[int]) -> str:
    """Map space group number (1-230) to crystal system. Returns 'Unknown' if invalid."""
    if sg_number is None or sg_number < 1 or sg_number > 230:
        return "Unknown"
    return _crystal_system_from_sg_number_cached(int(sg_number))


@lru_cache(maxsize=256)
def _crystal_system_from_sg_number_cached(sg_number: int) -> str:
    """Cached SG number -> crystal system mapping."""
    try:
        from pymatgen.symmetry.groups import SpaceGroup

        sg = SpaceGroup.from_int_number(sg_number)
        cs = getattr(sg, "crystal_system", None)
        return cs.capitalize() if cs else _crystal_system_fallback(sg_number)
    except Exception:
        return _crystal_system_fallback(sg_number)


def _crystal_system_fallback(sg_number: int) -> str:
    """IT number ranges when pymatgen crystal_system is unavailable."""
    if sg_number <= 2:
        return "Triclinic"
    if sg_number <= 15:
        return "Monoclinic"
    if sg_number <= 74:
        return "Orthorhombic"
    if sg_number <= 142:
        return "Tetragonal"
    if sg_number <= 167:
        return "Trigonal"
    if sg_number <= 194:
        return "Hexagonal"
    return "Cubic"


@lru_cache(maxsize=256)
def _n_elements_label(chemsys: str) -> str:
    """e.g. 'Fe-O' -> 'Binary', 'Fe-Mn-O' -> 'Ternary'."""
    n = len(chemsys.split("-"))
    if n == 1:
        return "Unary"
    if n == 2:
        return "Binary"
    if n == 3:
        return "Ternary"
    if n == 4:
        return "Quaternary"
    return f"{n}-ary"


_CIF_VOL = re.compile(r"_cell_volume\s+([0-9\.]+)")
_CIF_A = re.compile(r"_cell_length_a\s+([0-9\.]+)")
_CIF_B = re.compile(r"_cell_length_b\s+([0-9\.]+)")
_CIF_C = re.compile(r"_cell_length_c\s+([0-9\.]+)")
_CIF_ALPHA = re.compile(r"_cell_angle_alpha\s+([0-9\.]+)")
_CIF_BETA = re.compile(r"_cell_angle_beta\s+([0-9\.]+)")
_CIF_GAMMA = re.compile(r"_cell_angle_gamma\s+([0-9\.]+)")


@lru_cache(maxsize=4096)
def _lattice_metrics_from_cif(
    cif_content: str,
) -> Optional[Tuple[float, float, float, float, float, float, float]]:
    """Parse CIF once and return lattice metrics needed for deltas."""
    try:
        # Fast path using regex for massive speedup
        ma = _CIF_A.search(cif_content)
        mb = _CIF_B.search(cif_content)
        mc = _CIF_C.search(cif_content)
        malpha = _CIF_ALPHA.search(cif_content)
        mbeta = _CIF_BETA.search(cif_content)
        mgamma = _CIF_GAMMA.search(cif_content)

        if ma and mb and mc and malpha and mbeta and mgamma:
            a = float(ma.group(1))
            b = float(mb.group(1))
            c = float(mc.group(1))
            alpha = float(malpha.group(1))
            beta = float(mbeta.group(1))
            gamma = float(mgamma.group(1))

            mvol = _CIF_VOL.search(cif_content)
            if mvol:
                vol = float(mvol.group(1))
            else:
                al = math.radians(alpha)
                be = math.radians(beta)
                ga = math.radians(gamma)
                
                # Formula for volume of a parallelepiped
                vol = a * b * c * math.sqrt(
                    1 - math.cos(al)**2 - math.cos(be)**2 - math.cos(ga)**2 
                    + 2 * math.cos(al) * math.cos(be) * math.cos(ga)
                )
            return (vol, a, b, c, alpha, beta, gamma)
    except Exception:
        pass

    try:
        from pymatgen.core import Structure

        structure = Structure.from_str(cif_content, fmt="cif")
        lat = structure.lattice
        return (lat.volume, lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma)
    except Exception:
        return None


def _compute_geometry(
    original_cif: str,
    relaxed_cif: str,
) -> Optional[Dict[str, Any]]:
    """
    Parse both CIFs and compute volume/lattice deltas. Returns None on parse failure.
    """
    orig_metrics = _lattice_metrics_from_cif(original_cif)
    relax_metrics = _lattice_metrics_from_cif(relaxed_cif)
    if orig_metrics is None or relax_metrics is None:
        return None

    vol_orig, a_orig, b_orig, c_orig, alpha_orig, beta_orig, gamma_orig = orig_metrics
    vol_relax, a_relax, b_relax, c_relax, alpha_relax, beta_relax, gamma_relax = (
        relax_metrics
    )
    if vol_orig <= 0:
        return None
    volume_change_pct = 100.0 * (vol_relax - vol_orig) / vol_orig

    return {
        "volume_change_pct": volume_change_pct,
        "a_change_pct": (100.0 * (a_relax - a_orig) / a_orig if a_orig else None),
        "b_change_pct": (100.0 * (b_relax - b_orig) / b_orig if b_orig else None),
        "c_change_pct": (100.0 * (c_relax - c_orig) / c_orig if c_orig else None),
        "alpha_change": alpha_relax - alpha_orig if alpha_orig else None,
        "beta_change": beta_relax - beta_orig if beta_orig else None,
        "gamma_change": gamma_relax - gamma_orig if gamma_orig else None,
    }


def load_mp_relaxed_with_metadata(
    db: StructureDatabase,
    chemical_system: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load MP structures needed for analysis with a lightweight projection.
    """
    _ensure_analysis_indexes(db)

    query = """
        SELECT
            id, formula, chemsys, space_group_number, space_group_symbol,
            cif_content,
            CAST(json_extract(generation_metadata, '$.mp_original_space_group_number') AS INTEGER) as orig_sg_num,
            json_extract(generation_metadata, '$.mp_original_space_group_symbol') as orig_sg_sym,
            json_extract(generation_metadata, '$.mp_original_cif') as orig_cif
        FROM structures
        WHERE source = ? AND energy_per_atom IS NOT NULL AND is_valid = 1
          AND generation_metadata IS NOT NULL
          AND (
            json_extract(generation_metadata, '$.mp_original_space_group_number') IS NOT NULL
            OR json_extract(generation_metadata, '$.mp_original_space_group_symbol') IS NOT NULL
          )
    """
    params: List[Any] = ["mp"]
    if chemical_system:
        chemsys = db.normalize_chemsys(chemical_system)
        query = """
            SELECT
                id, formula, chemsys, space_group_number, space_group_symbol,
                cif_content,
                CAST(json_extract(generation_metadata, '$.mp_original_space_group_number') AS INTEGER) as orig_sg_num,
                json_extract(generation_metadata, '$.mp_original_space_group_symbol') as orig_sg_sym,
                json_extract(generation_metadata, '$.mp_original_cif') as orig_cif
            FROM structures
            WHERE source = ? AND energy_per_atom IS NOT NULL AND is_valid = 1
              AND generation_metadata IS NOT NULL
              AND (
                json_extract(generation_metadata, '$.mp_original_space_group_number') IS NOT NULL
                OR json_extract(generation_metadata, '$.mp_original_space_group_symbol') IS NOT NULL
              )
              AND chemsys = ?
        """
        params.append(chemsys)

    # Use row_factory = None for performance if possible, or extract correctly
    old_row_factory = db.conn.row_factory
    db.conn.row_factory = None
    try:
        structures: List[Dict[str, Any]] = []
        for row in db.conn.execute(query, params):
            orig_sg_num = row[6] # orig_sg_num
            orig_sg_sym = row[7] # orig_sg_sym
            
            # Only include if we actually have the metadata we need
            if orig_sg_num is not None or orig_sg_sym is not None:
                structures.append(
                    {
                        "id": row[0],
                        "formula": row[1],
                        "chemsys": row[2],
                        "space_group_number": row[3],
                        "space_group_symbol": row[4],
                        "cif_content": row[5],
                        "orig_sg_num": orig_sg_num,
                        "orig_sg_sym": orig_sg_sym,
                        "orig_cif": row[8], # orig_cif
                    }
                )
        return structures
    finally:
        db.conn.row_factory = old_row_factory


def _ensure_analysis_indexes(db: StructureDatabase) -> None:
    """Create optional indexes that accelerate this analysis query pattern."""
    try:
        db.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_structures_mp_relax_analysis
            ON structures(source, is_valid, chemsys, id)
            WHERE source = 'mp'
              AND is_valid = 1
              AND energy_per_atom IS NOT NULL
              AND generation_metadata IS NOT NULL
            """
        )
    except Exception:
        # Non-fatal optimization: analysis should still run without index support.
        pass


def analyze(
    db: StructureDatabase,
    chemical_system: Optional[str] = None,
    include_geometry: bool = True,
    collect_rows: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], int]:
    """
    Run analysis. Returns (rows for CSV, summary stats, cif_parse_failures).
    """
    structures = load_mp_relaxed_with_metadata(db, chemical_system=chemical_system)
    if not structures:
        return [], {}, 0

    rows: List[Dict[str, Any]] = []
    sg_changed_count = 0
    sg_symbol_only_changed = 0
    volume_changes: List[float] = []
    by_crystal_system: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"changed": 0, "unchanged": 0}
    )
    by_n_elements: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"changed": 0, "unchanged": 0}
    )
    by_original_sg: Dict[Tuple[int, str], Dict[str, int]] = defaultdict(
        lambda: {"changed": 0, "unchanged": 0}
    )
    formula_sg_changed: Dict[str, int] = defaultdict(int)
    formula_total: Dict[str, int] = defaultdict(int)
    cif_failures = 0
    top_abs_volume_changes: List[Tuple[float, str, Any, float]] = []

    # Pre-cache commonly used functions inside the loop
    for s in structures:
        orig_sg_num = s["orig_sg_num"]
        orig_sg_sym = s["orig_sg_sym"] or ""
        curr_sg_num = s["space_group_number"]
        curr_sg_sym = (s["space_group_symbol"] or "").strip()

        # Normalize to comparable types
        if orig_sg_num is not None:
            orig_sg_num = int(orig_sg_num)
        sg_num_changed = (
            orig_sg_num is not None
            and curr_sg_num is not None
            and orig_sg_num != curr_sg_num
        )
        sg_sym_changed = orig_sg_sym != curr_sg_sym
        sg_sym_only_this = sg_sym_changed and not sg_num_changed
        if sg_num_changed:
            sg_changed_count += 1
        elif sg_sym_only_this:
            sg_symbol_only_changed += 1

        crystal_sys = _crystal_system_from_sg_number(
            orig_sg_num if orig_sg_num is not None else curr_sg_num
        )
        
        # Micro-optimization for dictionary lookups
        cs_dict = by_crystal_system[crystal_sys]
        el_dict = by_n_elements[_n_elements_label(s["chemsys"])]
        sg_dict = by_original_sg[(orig_sg_num or 0, orig_sg_sym or "?")]
        
        if sg_num_changed:
            cs_dict["changed"] += 1
            el_dict["changed"] += 1
            sg_dict["changed"] += 1
            formula_sg_changed[s["formula"]] += 1
        else:
            cs_dict["unchanged"] += 1
            el_dict["unchanged"] += 1
            sg_dict["unchanged"] += 1

        formula_total[s["formula"]] += 1

        geom = None
        if include_geometry and s["orig_cif"] and s["cif_content"]:
            geom = _compute_geometry(s["orig_cif"], s["cif_content"])
            if geom is None:
                cif_failures += 1
            else:
                pct = geom["volume_change_pct"]
                volume_changes.append(pct)
                # Track top-N by absolute volume change without sorting all rows.
                item = (abs(pct), s["formula"], s["id"], pct)
                if len(top_abs_volume_changes) < 10:
                    heapq.heappush(top_abs_volume_changes, item)
                else:
                    heapq.heappushpop(top_abs_volume_changes, item)

        if collect_rows:
            row = {
                "id": s["id"],
                "formula": s["formula"],
                "chemsys": s["chemsys"],
                "original_sg_number": orig_sg_num,
                "original_sg_symbol": orig_sg_sym,
                "relaxed_sg_number": curr_sg_num,
                "relaxed_sg_symbol": curr_sg_sym,
                "sg_number_changed": sg_num_changed,
                "sg_symbol_only_changed": sg_sym_only_this,
                "crystal_system": crystal_sys,
                "n_elements_label": _n_elements_label(s["chemsys"]),
            }
            if geom:
                row["volume_change_pct"] = geom["volume_change_pct"]
                row["a_change_pct"] = geom.get("a_change_pct")
                row["b_change_pct"] = geom.get("b_change_pct")
                row["c_change_pct"] = geom.get("c_change_pct")
            else:
                row["volume_change_pct"] = None
                row["a_change_pct"] = row["b_change_pct"] = row["c_change_pct"] = None
            rows.append(row)

    total = len(structures)
    summary = {
        "total": total,
        "sg_number_changed": sg_changed_count,
        "sg_symbol_only_changed": sg_symbol_only_changed,
        "by_crystal_system": dict(by_crystal_system),
        "by_n_elements": dict(by_n_elements),
        "by_original_sg": dict(by_original_sg),
        "formula_sg_changed": dict(formula_sg_changed),
        "formula_total": dict(formula_total),
        "volume_changes": volume_changes,
        "top_abs_volume_changes": [
            (formula, sid, pct)
            for _, formula, sid, pct in sorted(
                top_abs_volume_changes, key=lambda x: x[0], reverse=True
            )
        ],
        "cif_parse_failures": cif_failures,
    }
    return rows, summary, cif_failures


def print_report(summary: Dict[str, Any], C: Any) -> None:
    """Print summary and tables to stdout."""
    total = summary["total"]
    sg_changed = summary["sg_number_changed"]
    sg_sym_only = summary["sg_symbol_only_changed"]
    pct = 100.0 * sg_changed / total if total else 0

    print(f"\n{C.BOLD}MP relaxation change analysis{C.RESET}")
    print(f"{C.DIM}{'=' * 50}{C.RESET}\n")
    print(f"  Relaxed MP structures with original metadata: {C.CYAN}{total}{C.RESET}")
    print(
        f"  Space group {C.BOLD}number{C.RESET} changed: {C.CYAN}{sg_changed}{C.RESET} ({pct:.1f}%)"
    )
    if sg_sym_only:
        print(
            f"  Space group symbol only (same number):     {C.DIM}{sg_sym_only}{C.RESET}"
        )

    by_cs = summary.get("by_crystal_system") or {}
    if by_cs:
        print(f"\n{C.BOLD}By original crystal system{C.RESET}")
        for name in [
            "Triclinic",
            "Monoclinic",
            "Orthorhombic",
            "Tetragonal",
            "Trigonal",
            "Hexagonal",
            "Cubic",
            "Unknown",
        ]:
            if name not in by_cs:
                continue
            d = by_cs[name]
            ch, un = d["changed"], d["unchanged"]
            t = ch + un
            p = 100.0 * ch / t if t else 0
            print(f"  {name:14s}  {ch:5d} changed / {t:5d} total  ({p:5.1f}% changed)")
        print()

    by_el = summary.get("by_n_elements") or {}
    if by_el:
        print(f"{C.BOLD}By number of elements{C.RESET}")
        for label in ["Unary", "Binary", "Ternary", "Quaternary"]:
            if label not in by_el:
                continue
            d = by_el[label]
            ch, un = d["changed"], d["unchanged"]
            t = ch + un
            p = 100.0 * ch / t if t else 0
            print(f"  {label:12s}  {ch:5d} changed / {t:5d} total  ({p:5.1f}% changed)")
        for k, d in sorted(by_el.items()):
            if k in ("Unary", "Binary", "Ternary", "Quaternary"):
                continue
            ch, un = d["changed"], d["unchanged"]
            t = ch + un
            p = 100.0 * ch / t if t else 0
            print(f"  {k:12s}  {ch:5d} changed / {t:5d} total  ({p:5.1f}% changed)")
        print()

    by_sg = summary.get("by_original_sg") or {}
    if by_sg:
        # Top original SGs by count of changes
        sg_list = [
            (k, v["changed"], v["changed"] + v["unchanged"]) for k, v in by_sg.items()
        ]
        sg_list.sort(key=lambda x: -x[1])
        print(f"{C.BOLD}Top original space groups by SG change count{C.RESET}")
        for (sg_num, sg_sym), ch, t in sg_list[:20]:
            if ch == 0:
                continue
            p = 100.0 * ch / t if t else 0
            print(
                f"  SG {sg_num:3d} {sg_sym:10s}  {ch:4d} changed / {t:4d}  ({p:5.1f}%)"
            )
        print()

    formula_changed = summary.get("formula_sg_changed") or {}
    formula_total_d = summary.get("formula_total") or {}
    if formula_changed:
        top_formulas = sorted(
            [
                (f, formula_changed[f], formula_total_d.get(f, 0))
                for f in formula_changed
            ],
            key=lambda x: -x[1],
        )[:20]
        print(f"{C.BOLD}Top formulas by SG change count{C.RESET}")
        for f, ch, t in top_formulas:
            p = 100.0 * ch / t if t else 0
            print(f"  {f:20s}  {ch:4d} changed / {t:4d}  ({p:5.1f}%)")
        print()

    vol = summary.get("volume_changes") or []
    if vol:
        import statistics

        abs_vol = [abs(v) for v in vol]
        print(f"{C.BOLD}Volume change (relaxed vs original){C.RESET}")
        print(f"  N with geometry:    {len(vol)}")
        print(f"  Mean volume change:  {statistics.mean(vol):+.2f}%")
        print(f"  Mean |volume change|: {statistics.mean(abs_vol):.2f}%")
        if len(vol) >= 2:
            print(f"  Stdev:               {statistics.stdev(vol):.2f}%")
        largest = summary.get("top_abs_volume_changes") or []
        if largest:
            print("  Largest |volume change| (formula, id, %):")
            for f, sid, pct in largest:
                print(f"    {f:16s} {sid}  {pct:+.2f}%")
        print()

    fail = summary.get("cif_parse_failures", 0)
    if fail:
        print(
            f"{C.DIM}CIF parse failures (excluded from volume stats): {fail}{C.RESET}\n"
        )


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    """Write rows to CSV (flat keys)."""
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze MP structure changes after mLIP/ORB relaxation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--database",
        type=str,
        default="ggen.db",
        help="Path to database file (default: ggen.db)",
    )
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="Chemical system filter (e.g. Fe-O). Default: all.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Write per-structure results to this CSV path",
    )
    parser.add_argument(
        "--no-geometry",
        action="store_true",
        help="Skip CIF parsing and volume/lattice metrics (faster on large DBs)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )
    args = parser.parse_args()

    if args.no_color or not sys.stdout.isatty():
        Colors.disable()
    C = Colors

    db_path = Path(args.database)
    if not db_path.exists():
        print(f"{C.RED}Error:{C.RESET} Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    db = StructureDatabase(str(db_path))
    try:
        rows, summary, _ = analyze(
            db,
            chemical_system=args.system,
            include_geometry=not args.no_geometry,
            collect_rows=bool(args.csv),
        )
        if summary.get("total", 0) == 0:
            print(
                f"{C.YELLOW}No relaxed MP structures with mp_original_* metadata found.{C.RESET}"
            )
            print(
                f"{C.DIM}Structures relaxed before metadata backfill was added are excluded.{C.RESET}"
            )
            return
        print_report(summary, C)
        if args.csv:
            write_csv(rows, args.csv)
            print(f"{C.GREEN}Wrote CSV: {args.csv}{C.RESET}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
