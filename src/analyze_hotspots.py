"""
Variant hotspot analysis across HV regions.

For each region (HV1F, HV1R, HV2F, HV3R):
  1. Find rCRS position range across all files
  2. Compute per-position variant frequency in noisy vs clean files
  3. Detect hotspots: positions where noisy_rate >> clean_rate
  4. Detect SNV clusters: windows of consecutive variant-dense positions
  5. Visualize and save to plots/hotspots_<region>.png
"""

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from scipy.ndimage import uniform_filter1d

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data"
LABELS_CSV = ROOT / "labels_v2.csv"
PLOTS_DIR = ROOT / "plots"
REGIONS = ["HV1F", "HV1R", "HV2F", "HV3R"]


def resolve_path(rel_path: str) -> Path:
    return DATA_ROOT / rel_path


def load_variants(json_path: Path) -> dict:
    """Extract alignment range and variants from Tracy JSON."""
    with open(json_path) as f:
        d = json.load(f)

    ref_align = d.get("ref1align", "")
    ref_bases = ref_align.replace("-", "")
    start = d.get("ref1pos", 0)
    end = start + len(ref_bases) - 1

    v = d.get("variants", {})
    rows = v.get("rows", [])
    cols = v.get("columns", [])
    col_idx = {c: i for i, c in enumerate(cols)}

    variants = []
    for row in rows:
        vtype = row[col_idx["type"]]
        pos = row[col_idx["pos"]]
        ref = row[col_idx["ref"]]
        alt = row[col_idx["alt"]]
        filt = row[col_idx["filter"]]
        qual = row[col_idx["qual"]]
        variants.append({
            "pos": pos,
            "ref": ref,
            "alt": alt,
            "type": vtype,
            "filter": filt,
            "qual": qual,
        })

    return {"start": start, "end": end, "variants": variants}


def analyze_region(region: str, df_labels: pd.DataFrame) -> dict:
    """Analyze one HV region. Returns stats dict."""
    mask = df_labels["FilePath"].str.contains(region)
    sub = df_labels[mask].copy()
    if sub.empty:
        print(f"[{region}] No files found, skip.")
        return {}

    noisy = sub[sub["Label"] == 1]
    clean = sub[sub["Label"] == 0]
    n_noisy = len(noisy)
    n_clean = len(clean)
    print(f"[{region}] total={len(sub)}, noisy={n_noisy}, clean={n_clean}")

    all_starts, all_ends = [], []
    # pos -> count of files having variant there
    noisy_counts: dict[int, int] = defaultdict(int)
    clean_counts: dict[int, int] = defaultdict(int)
    # per-file SNV positions for cluster analysis
    noisy_snv_sets: list[set] = []
    clean_snv_sets: list[set] = []

    def process_files(rows, counts_dict, snv_sets):
        for _, row in rows.iterrows():
            p = resolve_path(row["FilePath"])
            if not p.exists():
                continue
            try:
                info = load_variants(p)
            except Exception as e:
                print(f"  [warn] {p.name}: {e}")
                continue
            all_starts.append(info["start"])
            all_ends.append(info["end"])
            seen_pos = set()
            snv_pos = set()
            for v in info["variants"]:
                pos = v["pos"]
                if pos not in seen_pos:
                    counts_dict[pos] += 1
                    seen_pos.add(pos)
                if v["type"] == "SNV":
                    snv_pos.add(pos)
            snv_sets.append(snv_pos)

    process_files(noisy, noisy_counts, noisy_snv_sets)
    process_files(clean, clean_counts, clean_snv_sets)

    if not all_starts:
        print(f"[{region}] No readable files, skip.")
        return {}

    # Use percentile-clipped range to exclude outlier alignments
    pos_min = int(np.percentile(all_starts, 5))
    pos_max = int(np.percentile(all_ends, 95))
    raw_min, raw_max = min(all_starts), max(all_ends)
    if raw_min != pos_min or raw_max != pos_max:
        print(f"[{region}] rCRS range: {pos_min}–{pos_max} (clipped from {raw_min}–{raw_max})")
    else:
        print(f"[{region}] rCRS range: {pos_min}–{pos_max}")

    positions = np.arange(pos_min, pos_max + 1)
    N = len(positions)
    pos_to_idx = {p: i for i, p in enumerate(positions)}

    noisy_rate = np.zeros(N)
    clean_rate = np.zeros(N)

    for pos, cnt in noisy_counts.items():
        if pos in pos_to_idx:
            noisy_rate[pos_to_idx[pos]] = cnt / n_noisy

    for pos, cnt in clean_counts.items():
        if pos in pos_to_idx:
            clean_rate[pos_to_idx[pos]] = cnt / n_clean

    # Fisher's exact test per position (with at least 1 variant in noisy)
    # Returns positions where noisy enrichment is significant
    hotspot_mask = np.zeros(N, dtype=bool)
    hotspot_pval = np.ones(N)
    for i, pos in enumerate(positions):
        nc = noisy_counts.get(pos, 0)
        cc = clean_counts.get(pos, 0)
        if nc == 0:
            continue
        table = [[nc, n_noisy - nc], [cc, n_clean - cc]]
        _, pval = fisher_exact(table, alternative="greater")
        hotspot_pval[i] = pval
        if pval < 0.05 and noisy_rate[i] > clean_rate[i] + 0.05:
            hotspot_mask[i] = True

    # SNV cluster density: sliding window of 20 bp over per-file SNV sets
    # Aggregate SNV positions (any file, noisy only) → density array
    cluster_window = 20
    noisy_snv_density = np.zeros(N)
    for snv_set in noisy_snv_sets:
        for pos in snv_set:
            if pos in pos_to_idx:
                noisy_snv_density[pos_to_idx[pos]] += 1
    noisy_snv_density /= max(n_noisy, 1)
    # Smooth with rolling window
    snv_cluster_density = uniform_filter1d(noisy_snv_density, size=cluster_window)

    # High-density cluster threshold: mean + 2*std of non-zero regions
    nonzero = snv_cluster_density[snv_cluster_density > 0]
    if len(nonzero) > 0:
        cluster_thresh = nonzero.mean() + 2 * nonzero.std()
    else:
        cluster_thresh = 1.0
    cluster_mask = snv_cluster_density > cluster_thresh

    return {
        "region": region,
        "pos_min": pos_min,
        "pos_max": pos_max,
        "positions": positions,
        "noisy_rate": noisy_rate,
        "clean_rate": clean_rate,
        "hotspot_mask": hotspot_mask,
        "hotspot_pval": hotspot_pval,
        "snv_cluster_density": snv_cluster_density,
        "cluster_mask": cluster_mask,
        "cluster_thresh": cluster_thresh,
        "n_noisy": n_noisy,
        "n_clean": n_clean,
    }


def visualize_region(stats: dict, out_dir: Path):
    if not stats:
        return

    region = stats["region"]
    positions = stats["positions"]
    noisy_rate = stats["noisy_rate"]
    clean_rate = stats["clean_rate"]
    hotspot_mask = stats["hotspot_mask"]
    snv_cluster_density = stats["snv_cluster_density"]
    cluster_mask = stats["cluster_mask"]
    cluster_thresh = stats["cluster_thresh"]

    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharex=True,
                              gridspec_kw={"height_ratios": [3, 2, 1]})
    fig.suptitle(
        f"{region} — Variant Hotspot Analysis\n"
        f"(noisy n={stats['n_noisy']}, clean n={stats['n_clean']}, "
        f"rCRS {stats['pos_min']}–{stats['pos_max']})",
        fontsize=13,
    )

    ax0, ax1, ax2 = axes

    # Panel 0: variant rate noisy vs clean
    ax0.fill_between(positions, noisy_rate, alpha=0.3, color="tomato", label=f"noisy (n={stats['n_noisy']})")
    ax0.fill_between(positions, clean_rate, alpha=0.3, color="steelblue", label=f"clean (n={stats['n_clean']})")
    ax0.plot(positions, noisy_rate, color="tomato", lw=0.8)
    ax0.plot(positions, clean_rate, color="steelblue", lw=0.8)
    # Shade hotspot regions
    in_hotspot = False
    hs_start = None
    for i, (pos, is_hot) in enumerate(zip(positions, hotspot_mask)):
        if is_hot and not in_hotspot:
            hs_start = pos
            in_hotspot = True
        elif not is_hot and in_hotspot:
            ax0.axvspan(hs_start, pos, alpha=0.15, color="red", zorder=0)
            in_hotspot = False
    if in_hotspot:
        ax0.axvspan(hs_start, positions[-1], alpha=0.15, color="red", zorder=0)
    ax0.set_ylabel("Variant rate")
    ax0.legend(loc="upper right", fontsize=9)
    ax0.set_ylim(0, min(1.0, max(noisy_rate.max(), clean_rate.max()) * 1.2 + 0.05))
    red_patch = mpatches.Patch(color="red", alpha=0.3, label="hotspot (noisy enriched, p<0.05)")
    ax0.legend(handles=ax0.lines[:2] + [red_patch],
               labels=[f"noisy (n={stats['n_noisy']})", f"clean (n={stats['n_clean']})", "hotspot"],
               loc="upper right", fontsize=9)

    # Panel 1: SNV cluster density
    ax1.plot(positions, snv_cluster_density, color="darkorange", lw=1.0, label="SNV cluster density (20bp window)")
    ax1.axhline(cluster_thresh, color="red", lw=0.8, ls="--", label=f"threshold={cluster_thresh:.3f}")
    ax1.fill_between(positions, snv_cluster_density, where=cluster_mask,
                     alpha=0.4, color="darkorange", label="high-density cluster")
    ax1.set_ylabel("SNV density\n(noisy files)")
    ax1.legend(loc="upper right", fontsize=9)

    # Panel 2: difference (noisy - clean)
    diff = noisy_rate - clean_rate
    ax2.bar(positions, diff, color=np.where(diff > 0, "tomato", "steelblue"),
            width=1.0, alpha=0.7)
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_ylabel("Δ rate\n(noisy−clean)")
    ax2.set_xlabel("rCRS position")

    plt.tight_layout()
    out_path = out_dir / f"hotspots_{region}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[{region}] Saved → {out_path}")


def print_hotspot_summary(stats: dict):
    if not stats:
        return
    region = stats["region"]
    positions = stats["positions"]
    hotspot_mask = stats["hotspot_mask"]
    cluster_mask = stats["cluster_mask"]
    noisy_rate = stats["noisy_rate"]
    clean_rate = stats["clean_rate"]

    print(f"\n=== {region} HOTSPOT POSITIONS (noisy-enriched, p<0.05) ===")
    hot_pos = positions[hotspot_mask]
    if len(hot_pos) == 0:
        print("  None found.")
    else:
        print(f"  {len(hot_pos)} positions:")
        for pos in hot_pos:
            idx = pos - stats["pos_min"]
            nr = noisy_rate[idx]
            cr = clean_rate[idx]
            print(f"    rCRS {pos:6d}: noisy={nr:.3f}  clean={cr:.3f}  Δ={nr-cr:.3f}")

    print(f"\n=== {region} SNV CLUSTER REGIONS ===")
    # Find contiguous clusters
    in_clust = False
    clust_start = None
    clusters = []
    for pos, is_clust in zip(positions, cluster_mask):
        if is_clust and not in_clust:
            clust_start = pos
            in_clust = True
        elif not is_clust and in_clust:
            clusters.append((clust_start, pos - 1))
            in_clust = False
    if in_clust:
        clusters.append((clust_start, positions[-1]))

    if not clusters:
        print("  None found.")
    else:
        for s, e in clusters:
            print(f"  rCRS {s}–{e}  (len={e-s+1})")


# Known homopolymer / sequencing-artifact positions in rCRS (0-based ranges are inclusive)
# These are real sequencing artifacts, not degraded-DNA noise signals.
HOMOPOLYMER_REGIONS = [
    (303, 315, "HV2 poly-C"),
    (513, 524, "HV2 poly-AC"),
    (568, 573, "HV3 poly-C"),
    (16183, 16194, "HV1 poly-C"),
    (16184, 16193, "HV1 poly-CA"),
]

# Threshold: positions with clean_rate above this are likely population polymorphisms
POPULATION_VARIANT_CLEAN_RATE = 0.30


def _get_hotspot_set(stats: dict, pos_range: range | None = None) -> set[int]:
    """Return set of rCRS positions flagged as hotspots in stats, optionally restricted to pos_range."""
    positions = stats["positions"]
    mask = stats["hotspot_mask"]
    result = set()
    for pos, is_hot in zip(positions, mask):
        if is_hot:
            if pos_range is None or pos in pos_range:
                result.add(pos)
    return result


def _is_homopolymer(pos: int) -> str | None:
    """Return homopolymer annotation if pos falls in known artifact region, else None."""
    for start, end, label in HOMOPOLYMER_REGIONS:
        if start <= pos <= end:
            return label
    return None


def _get_rate_at(stats: dict, pos: int) -> tuple[float, float]:
    """Return (noisy_rate, clean_rate) at rCRS pos, or (0,0) if out of range."""
    positions = stats["positions"]
    idx = pos - stats["pos_min"]
    if idx < 0 or idx >= len(positions):
        return 0.0, 0.0
    return float(stats["noisy_rate"][idx]), float(stats["clean_rate"][idx])


def analyze_paired(
    pair_name: str,
    stats_a: dict,
    stats_b: dict,
    out_dir: Path,
):
    """
    Cross-validate hotspots between two primers covering the same rCRS region.

    Biological filters applied in order:
    1. Pair intersection: position must be flagged in BOTH primers (direction-consistent)
    2. Population polymorphism: clean_rate >= 0.30 → real haplogroup SNP, not noise
    3. Known homopolymer: poly-C/poly-A runs → sequencing chemistry artifact

    Final output: "confirmed noise hotspots" that survive all three filters.
    """
    if not stats_a or not stats_b:
        print(f"[{pair_name}] Missing stats for one or both regions, skip.")
        return

    region_a = stats_a["region"]
    region_b = stats_b["region"]

    # Shared rCRS range (intersection of both coverages)
    shared_min = max(stats_a["pos_min"], stats_b["pos_min"])
    shared_max = min(stats_a["pos_max"], stats_b["pos_max"])
    if shared_min >= shared_max:
        print(f"[{pair_name}] No overlapping rCRS range, skip.")
        return

    shared_range = range(shared_min, shared_max + 1)
    print(f"\n[{pair_name}] Shared rCRS range: {shared_min}–{shared_max}")

    # Layer 1: hotspot sets per region, restricted to shared range
    hot_a = _get_hotspot_set(stats_a, shared_range)
    hot_b = _get_hotspot_set(stats_b, shared_range)

    intersection = hot_a & hot_b          # both primers agree
    only_a = hot_a - hot_b                # only forward/first primer
    only_b = hot_b - hot_a                # only reverse/second primer

    print(f"  {region_a} hotspots in shared range: {len(hot_a)}")
    print(f"  {region_b} hotspots in shared range: {len(hot_b)}")
    print(f"  Intersection (both primers): {len(intersection)}")
    print(f"  Only {region_a}: {len(only_a)} (likely {region_a}-direction artifact)")
    print(f"  Only {region_b}: {len(only_b)} (likely {region_b}-direction artifact)")

    # Layer 2 + 3: apply biological filters to the intersection
    confirmed = set()
    filtered_pop = set()
    filtered_homo = set()

    for pos in sorted(intersection):
        nr_a, cr_a = _get_rate_at(stats_a, pos)
        nr_b, cr_b = _get_rate_at(stats_b, pos)
        clean_rate_max = max(cr_a, cr_b)  # conservative: use higher of the two

        homo = _is_homopolymer(pos)
        if homo:
            filtered_homo.add(pos)
        elif clean_rate_max >= POPULATION_VARIANT_CLEAN_RATE:
            filtered_pop.add(pos)
        else:
            confirmed.add(pos)

    print(f"\n  [Filter 2] Population polymorphisms removed: {len(filtered_pop)}")
    print(f"  [Filter 3] Homopolymer artifacts removed:    {len(filtered_homo)}")
    print(f"  ✓ Confirmed noise hotspots: {len(confirmed)}")

    # Print confirmed hotspots
    if confirmed:
        print(f"\n=== {pair_name} CONFIRMED NOISE HOTSPOTS ===")
        for pos in sorted(confirmed):
            nr_a, cr_a = _get_rate_at(stats_a, pos)
            nr_b, cr_b = _get_rate_at(stats_b, pos)
            print(
                f"  rCRS {pos:6d}:  "
                f"{region_a} noisy={nr_a:.3f}/clean={cr_a:.3f}  "
                f"{region_b} noisy={nr_b:.3f}/clean={cr_b:.3f}"
            )

    # Visualization
    _visualize_paired(
        pair_name, region_a, region_b, stats_a, stats_b,
        shared_min, shared_max, intersection, only_a, only_b,
        confirmed, filtered_pop, filtered_homo, out_dir,
    )


def _visualize_paired(
    pair_name, region_a, region_b, stats_a, stats_b,
    shared_min, shared_max, intersection, only_a, only_b,
    confirmed, filtered_pop, filtered_homo, out_dir: Path,
):
    positions = np.arange(shared_min, shared_max + 1)
    N = len(positions)

    def rate_array(stats, label):
        arr = np.zeros(N)
        for i, pos in enumerate(positions):
            nr, cr = _get_rate_at(stats, pos)
            arr[i] = nr if label == "noisy" else cr
        return arr

    noisy_a = rate_array(stats_a, "noisy")
    noisy_b = rate_array(stats_b, "noisy")
    clean_a = rate_array(stats_a, "clean")
    clean_b = rate_array(stats_b, "clean")

    # Build category mask for each position (for bottom panel)
    cat = np.zeros(N, dtype=int)  # 0=none,1=only_a,2=only_b,3=intersection,4=confirmed
    for i, pos in enumerate(positions):
        if pos in confirmed:
            cat[i] = 4
        elif pos in filtered_pop or pos in filtered_homo:
            cat[i] = 3  # intersection but filtered out
        elif pos in only_a:
            cat[i] = 1
        elif pos in only_b:
            cat[i] = 2

    fig, axes = plt.subplots(3, 1, figsize=(18, 11), sharex=True,
                              gridspec_kw={"height_ratios": [3, 3, 2]})
    fig.suptitle(
        f"{pair_name} — Paired Biological Filter\n"
        f"Shared rCRS {shared_min}–{shared_max}  |  "
        f"confirmed={len(confirmed)}  pop_filtered={len(filtered_pop)}  "
        f"homo_filtered={len(filtered_homo)}",
        fontsize=13,
    )

    # Panel 0: region_a rates
    ax0 = axes[0]
    ax0.fill_between(positions, noisy_a, alpha=0.3, color="tomato")
    ax0.fill_between(positions, clean_a, alpha=0.3, color="steelblue")
    ax0.plot(positions, noisy_a, color="tomato", lw=0.8, label=f"{region_a} noisy")
    ax0.plot(positions, clean_a, color="steelblue", lw=0.8, label=f"{region_a} clean")
    ax0.set_ylabel(f"{region_a}\nVariant rate")
    ax0.legend(loc="upper right", fontsize=8)

    # Panel 1: region_b rates
    ax1 = axes[1]
    ax1.fill_between(positions, noisy_b, alpha=0.3, color="darkorange")
    ax1.fill_between(positions, clean_b, alpha=0.3, color="teal")
    ax1.plot(positions, noisy_b, color="darkorange", lw=0.8, label=f"{region_b} noisy")
    ax1.plot(positions, clean_b, color="teal", lw=0.8, label=f"{region_b} clean")
    ax1.set_ylabel(f"{region_b}\nVariant rate")
    ax1.legend(loc="upper right", fontsize=8)

    # Panel 2: category bars
    ax2 = axes[2]
    colors = {1: "gold", 2: "mediumpurple", 3: "lightcoral", 4: "crimson"}
    labels = {
        1: f"only {region_a} (direction artifact)",
        2: f"only {region_b} (direction artifact)",
        3: "both primers, filtered (pop. variant / homopolymer)",
        4: "CONFIRMED noise hotspot",
    }
    for cat_id in [1, 2, 3, 4]:
        mask = cat == cat_id
        if mask.any():
            ax2.bar(positions[mask], np.ones(mask.sum()), color=colors[cat_id],
                    alpha=0.85, width=1.0, label=labels[cat_id])

    # Shade homopolymer regions
    for h_start, h_end, h_label in HOMOPOLYMER_REGIONS:
        if h_start <= shared_max and h_end >= shared_min:
            ax2.axvspan(max(h_start, shared_min), min(h_end, shared_max),
                        alpha=0.12, color="gray", zorder=0, label=f"homopolymer: {h_label}")

    ax2.set_yticks([])
    ax2.set_ylabel("Category")
    ax2.set_xlabel("rCRS position")
    ax2.legend(loc="upper right", fontsize=8, ncol=2)

    plt.tight_layout()
    out_path = out_dir / f"paired_{pair_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[{pair_name}] Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Variant hotspot analysis for HV regions")
    parser.add_argument("--labels", default=str(LABELS_CSV))
    parser.add_argument("--regions", nargs="+", default=REGIONS)
    parser.add_argument("--out-dir", default=str(PLOTS_DIR))
    parser.add_argument("--paired", action="store_true",
                        help="Run paired biological filter for (HV1F,HV1R) and (HV2F,HV3R)")
    args = parser.parse_args()

    df = pd.read_csv(args.labels)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    all_stats: dict[str, dict] = {}
    for region in args.regions:
        stats = analyze_region(region, df)
        all_stats[region] = stats
        print_hotspot_summary(stats)
        visualize_region(stats, out_dir)

    if args.paired:
        PAIRS = [("HV1", "HV1F", "HV1R"), ("HV1HV2", "HV2F", "HV3R")]
        for pair_name, ra, rb in PAIRS:
            if ra in all_stats and rb in all_stats:
                analyze_paired(pair_name, all_stats[ra], all_stats[rb], out_dir)
            else:
                # Load missing regions
                sa = all_stats.get(ra) or analyze_region(ra, df)
                sb = all_stats.get(rb) or analyze_region(rb, df)
                analyze_paired(pair_name, sa, sb, out_dir)


if __name__ == "__main__":
    main()
