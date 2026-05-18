"""
src/utils/generate_labels_v2.py
================================
Gán nhãn (0=clean, 1=noisy) cho các file .json trace trong pipeline_results/,
dựa trên nội dung cột Issues (Run1, Run2, Run3) của data/metadata_rerun.tsv.

Mỗi sample thường có đúng 4 file JSON: HV1F, HV1R, HV2F, HV3R.

─── Logic gán nhãn ───────────────────────────────────────────────────────────

Bước 1 — Parse Issues:
  Với mỗi sample, đọc tất cả cột Issues (Issues, Issues.1, Issues.2) và
  trích xuất tập hợp các vùng (HV region) được nhắc tên.

  Các dạng mã vùng được nhận diện:
    HV1F | 1F              → HV1F noisy
    HV1R | 1R              → HV1R noisy
    HV2F | 2F              → HV2F noisy
    HV3R | 3R | HV3F | 3F → HV3R noisy  (HV3F được map sang HV3R)
    HV3  (không có F/R)   → HV3R noisy
    HV2  (không có F/R)   → HV2F noisy
    HV1  (không có F/R)   → HV1F + HV1R noisy

  Trigger tất cả vùng đều noisy (label=1 cho cả 4 file):
    '4 trình tự failed/nhiễu' | 'fail 4 chiều' | 'NHIỄM MẪU' |
    'khả năng nhiễm mẫu'

Bước 2 — Quyết định gán nhãn:
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ Issues đề cập ≥1 region cụ thể                                           │
  │   → File đó thuộc region được đề cập  : Label = 1 (noisy)               │
  │   → File đó thuộc region khác          : Label = 0 (clean)              │
  ├──────────────────────────────────────────────────────────────────────────┤
  │ Issues là '4 trình tự failed/nhiễu' | 'fail 4 chiều'                     │
  │   → Tất cả 4 file (HV1F, HV1R, HV2F, HV3R) : Label = 1 (noisy)         │
  ├──────────────────────────────────────────────────────────────────────────┤
  │ Issues là 'NHIỄM MẪU' | 'khả năng nhiễm mẫu'                            │
  │   → BỎ QUA toàn bộ sample (sample bị loại, không gán nhãn)              │
  ├──────────────────────────────────────────────────────────────────────────┤
  │ Issues trống / NaN / không đề cập region nào (vd: '16355del?', '73A?')  │
  │   → BỎ QUA toàn bộ sample (không gán nhãn cho bất kỳ file nào)          │
  └──────────────────────────────────────────────────────────────────────────┘

  Ghi chú: Issues chứa thông tin về vị trí variant bất thường (như '16355del?',
  '73A?', '16189Y?') KHÔNG được coi là noise signal — đây là vấn đề interpreting
  variant chứ không phải chất lượng trace. Những sample này bị bỏ qua.

  Ghi chú về nhiễm mẫu: File trace của mẫu bị nhiễm có thể có tín hiệu hỗn
  hợp từ nhiều cá thể — không thể gán nhãn noisy/clean một cách tin cậy.
  Những sample này bị bỏ qua hoàn toàn khỏi tập nhãn.

─── Output ───────────────────────────────────────────────────────────────────

  labels_v2.csv       — (FilePath, Label)                compatible với train.py
  labels_v2.debug.csv — (FilePath, Label, SampleID, Region)  để kiểm tra/debug

─── Chạy ─────────────────────────────────────────────────────────────────────

  python src/utils/generate_labels_v2.py
"""


from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).resolve().parents[2]
METADATA_PATH  = ROOT / "data" / "metadata_rerun.tsv"
PIPELINE_ROOT  = ROOT / "data" / "pipeline_results"
OUTPUT_CSV     = ROOT / "labels_v2.csv"

# ── Constants ─────────────────────────────────────────────────────────────────
EXCLUDE_FILES   = {"comparison.json", "variants.json"}

# Regions in filename
REGIONS = ["HV1F", "HV1R", "HV2F", "HV3R"]

# Patterns: tất cả 4 file của sample đều label=1
# Áp dụng khi trace thực sự failed hoàn toàn (không phải lỗi variant hay nhiễm mẫu)
ALL_NOISY_PATTERNS = re.compile(
    r"4\s*trình\s*tự\s*(failed|nhiễu)"   # '4 trình tự failed' / '4 trình tự nhiễu'
    r"|fail\s*4\s*chiều"                   # 'fail 4 chiều'
    r"|fail\s*trình\s*tự"                 # 'Fail trình tự'
    r"|4\s*trình\s*tự\s*nhiễu",           # '4 trình tự nhiễu'
    re.IGNORECASE | re.UNICODE,
)

# Patterns: nhiễm mẫu → BỎ QUA toàn bộ sample (không gán nhãn)
# Tín hiệu hỗn hợp từ nhiều cá thể — không thể phán đoán noisy/clean tin cậy
CONTAMINATION_PATTERNS = re.compile(
    r"nhiễm\s*mẫu"               # 'nhiễm mẫu', 'NHIỄM MẪU'
    r"|khả\s*năng\s*nhiễm\s*mẫu" # 'khả năng nhiễm mẫu'
    r"|nhầm\s*mẫu"               # 'nhầm mẫu'
    r"|nghi\s*ngờ\s*nhiễm"       # 'nghi ngờ nhiễm mẫu'
    r"|nhiễm\s+mẫu",             # dạng khác
    re.IGNORECASE | re.UNICODE,
)

# ── Region-to-file mapping ────────────────────────────────────────────────────
# Maps text patterns → set of REGION codes (HV1F, HV1R, HV2F, HV3R)
# Order matters: longer / more specific first

REGION_PATTERNS: list[tuple[re.Pattern, set[str]]] = [
    # Full 4-char codes — use (?<![A-Za-z]) to avoid \b issue with underscore separator
    (re.compile(r"(?<![A-Za-z])HV1F(?![A-Za-z])", re.IGNORECASE), {"HV1F"}),
    (re.compile(r"(?<![A-Za-z])HV1R(?![A-Za-z])", re.IGNORECASE), {"HV1R"}),
    (re.compile(r"(?<![A-Za-z])HV2F(?![A-Za-z])", re.IGNORECASE), {"HV2F"}),
    (re.compile(r"(?<![A-Za-z])HV3R(?![A-Za-z])", re.IGNORECASE), {"HV3R"}),

    # HV3F appears in Run2 issues (như '73-302_HV3F nhiễu') → treat as HV3R
    (re.compile(r"(?<![A-Za-z])HV3F(?![A-Za-z])", re.IGNORECASE), {"HV3R"}),

    # 'HV3' alone (without F/R suffix) → HV3R (only HV3 file in dataset)
    (re.compile(r"(?<![A-Za-z])HV3(?![A-Za-z])", re.IGNORECASE), {"HV3R"}),

    # 'HV2' alone → HV2F
    (re.compile(r"(?<![A-Za-z])HV2(?![A-Za-z])", re.IGNORECASE), {"HV2F"}),

    # 'HV1' alone → both HV1F and HV1R
    (re.compile(r"(?<![A-Za-z])HV1(?![A-Za-z])", re.IGNORECASE), {"HV1F", "HV1R"}),

    # Short codes: '1F', '1R', '2F', '3R' — not preceded/followed by letter or digit
    (re.compile(r"(?<![A-Za-z\d])1F(?![A-Za-z\d])", re.IGNORECASE), {"HV1F"}),
    (re.compile(r"(?<![A-Za-z\d])1R(?![A-Za-z\d])", re.IGNORECASE), {"HV1R"}),
    (re.compile(r"(?<![A-Za-z\d])2F(?![A-Za-z\d])", re.IGNORECASE), {"HV2F"}),
    (re.compile(r"(?<![A-Za-z\d])3R(?![A-Za-z\d])", re.IGNORECASE), {"HV3R"}),

    # '3F' → HV3R (same region, F direction used in some Run2 notes)
    (re.compile(r"(?<![A-Za-z\d])3F(?![A-Za-z\d])", re.IGNORECASE), {"HV3R"}),
]


# Any mention of these keywords in Issues → the sample has at least some noise
# but we only flag the SPECIFIC region files
GENERIC_NOISE_KEYWORDS = re.compile(
    r"nhiễu|nền cao|dyeblob|dye blob|peak spike|peak shoulder|fail|failed"
    r"|polyC|polyAC|tín hiệu yếu|tín hiệu kém|tín hiệu phức tạp|ngắn"
    r"|mixed|nhầm mẫu|nhiễm",
    re.IGNORECASE | re.UNICODE,
)

# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_noisy_regions(text: str | float) -> set[str]:
    """
    Parse một chuỗi Issues và trả về tập hợp HV region codes bị noisy.

    Returns:
        set[str]: tập hợp region codes trong {'HV1F','HV1R','HV2F','HV3R'}.
            - Trả về set đầy đủ 4 region nếu Issues là '4 trình tự failed' / 'fail 4 chiều'.
            - Trả về set() (rỗng) nếu:
                * text trống / NaN
                * Issues là dạng nhiễm mẫu ('NHIỄM MẪU', 'khả năng nhiễm mẫu')  → skip
                * Issues không đề cập region nào (vd: '16355del?', '73A?')       → skip
    """
    if not isinstance(text, str):
        return set()
    text = text.strip()
    if not text or text.upper() in ("NAN", "NONE", "N/A", "ĐÃ XÁC MINH", "ĐÃ SO"):
        return set()

    # Nhiễm mẫu → BỎ QUA (trả về rỗng để sample bị skip)
    if CONTAMINATION_PATTERNS.search(text):
        return set()

    # 4 trình tự failed / fail 4 chiều → tất cả 4 region đều noisy
    if ALL_NOISY_PATTERNS.search(text):
        return set(REGIONS)

    # Tìm region codes cụ thể được đề cập
    noisy = set()
    for pat, region_set in REGION_PATTERNS:
        if pat.search(text):
            noisy |= region_set
    return noisy


def issues_for_sample(row: pd.Series, issues_cols: list[str]) -> set[str]:
    """Collect all noisy regions across all Issues columns for one metadata row."""
    noisy: set[str] = set()
    for col in issues_cols:
        val = row.get(col, float("nan"))
        noisy |= parse_noisy_regions(val)
    return noisy


# ── File discovery ────────────────────────────────────────────────────────────

def extract_sample_and_region(filename: str) -> tuple[str, str] | None:
    """
    Extract (sample_id, region) from a JSON filename.

    Supported patterns:
      WELL_DATE_SAMPLEID_REGION_RUN.json   (numeric sample like 253719)
      WELL_DATE_SAMPLEID_REGION_RUN.json   (alpha sample like LN_25_AA0835)
    """
    # Pattern: anything up to _REGION_DIGITS.json at the end
    m = re.search(
        r"_(\d{6}|LN_\d+_[A-Z]{2}\d+|MS_\d+_[A-Z]{2}\d+|AB_\d+_[A-Z]{2}\d+)"
        r"_(HV\d[FR])_\d+\.json$",
        filename,
        re.IGNORECASE,
    )
    if m:
        return m.group(1), m.group(2).upper()
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Load metadata
    meta = pd.read_csv(METADATA_PATH, sep="\t", dtype=str)
    # The sample ID column is 'Sample.1' (second header row merged by pandas)
    sample_col = "Sample.1"
    issues_cols = [c for c in meta.columns if c.startswith("Issues")]

    # Build lookup: sample_id -> set of noisy regions
    meta[sample_col] = meta[sample_col].str.strip()
    meta = meta.dropna(subset=[sample_col])
    meta = meta[meta[sample_col] != ""]

    print(f"Metadata rows loaded: {len(meta)}")
    print(f"Issues columns: {issues_cols}")

    # Build two lookups:
    #   sample_noisy   : sample_id -> set of NOISY regions (non-empty = region mentioned)
    #   sample_labeled : sample_id -> True if we should label this sample at all
    #
    # Rule: only label a sample if its Issues mention ≥1 HV region.
    # If Issues exist but no region is extractable (e.g. '16355del?', '73A?')
    # → skip the sample entirely (do not assign label 0 or 1).

    sample_noisy: dict[str, set[str]] = {}   # sample_id -> {noisy regions}
    sample_labeled: set[str] = set()          # sample_id has ≥1 region mentioned

    for _, row in meta.iterrows():
        sid = str(row[sample_col]).strip()
        noisy = issues_for_sample(row, issues_cols)

        if sid in sample_noisy:
            sample_noisy[sid] |= noisy
        else:
            sample_noisy[sid] = noisy

        if noisy:
            sample_labeled.add(sid)

    n_skip = sum(1 for sid, regs in sample_noisy.items()
                 if not regs and sid not in sample_labeled)
    print(f"  Samples with ≥1 region mentioned : {len(sample_labeled):,}")
    print(f"  Samples skipped (no region in Issues): {n_skip:,}")

    # Discover all trace JSON files
    all_files = [
        p for p in PIPELINE_ROOT.rglob("*.json")
        if p.name not in EXCLUDE_FILES
    ]
    print(f"Total JSON files found: {len(all_files)}")

    records = []
    stats = {
        "matched_and_labeled": 0,
        "skipped_no_region": 0,
        "skipped_non_trace": 0,
        "noisy": 0,
        "clean": 0,
    }

    for fpath in sorted(all_files):
        rel = fpath.relative_to(PIPELINE_ROOT.parent)  # relative to data/
        info = extract_sample_and_region(fpath.name)

        if info is None:
            # Non-trace file (no HV region in name)
            stats["skipped_non_trace"] += 1
            continue

        sample_id, region = info

        # Only label samples whose Issues explicitly mention ≥1 region
        if sample_id not in sample_labeled:
            stats["skipped_no_region"] += 1
            continue

        stats["matched_and_labeled"] += 1
        noisy_regions = sample_noisy[sample_id]
        label = 1 if region in noisy_regions else 0

        if label == 1:
            stats["noisy"] += 1
        else:
            stats["clean"] += 1

        records.append({
            "FilePath": str(rel),
            "Label": label,
            "SampleID": sample_id,
            "Region": region,
        })

    df_out = pd.DataFrame(records)

    # Save full output (with SampleID, Region for debugging)
    debug_path = OUTPUT_CSV.with_suffix(".debug.csv")
    df_out.to_csv(debug_path, index=False)

    # Save clean output (FilePath, Label only — compatible with train.py)
    df_out[["FilePath", "Label"]].to_csv(OUTPUT_CSV, index=False)

    # Print report
    total = len(records)
    print()
    print("=" * 60)
    print("  Label generation complete")
    print("=" * 60)
    print(f"  Total trace files labelled   : {total:,}")
    print(f"  Files labelled (with region) : {stats['matched_and_labeled']:,}")
    print(f"  Files skipped (no region)    : {stats['skipped_no_region']:,}")
    print(f"  Files skipped (non-trace)    : {stats['skipped_non_trace']:,}")
    print()
    if total > 0:
        print(f"  Label 0 (clean) : {stats['clean']:,}  ({stats['clean']/total*100:.1f}%)")
        print(f"  Label 1 (noisy) : {stats['noisy']:,}  ({stats['noisy']/total*100:.1f}%)")
    print()
    print(f"  Output CSV      : {OUTPUT_CSV}")
    print(f"  Debug CSV       : {debug_path}")
    print("=" * 60)

    # Per-region breakdown
    print()
    print("  Breakdown by region:")
    print(f"  {'Region':<8} {'Clean':>8} {'Noisy':>8} {'Total':>8} {'Noisy%':>8}")
    for reg in REGIONS:
        sub = df_out[df_out["Region"] == reg]
        nc = (sub["Label"] == 0).sum()
        nn = (sub["Label"] == 1).sum()
        nt = len(sub)
        pct = nn / nt * 100 if nt > 0 else 0
        print(f"  {reg:<8} {nc:>8,} {nn:>8,} {nt:>8,} {pct:>7.1f}%")

    # Samples summary
    samples_labeled_noisy = {sid for sid in sample_labeled if sample_noisy[sid]}
    samples_labeled_clean = sample_labeled - samples_labeled_noisy
    print()
    print(f"  Samples labeled (Issues have region) : {len(sample_labeled):,}")
    print(f"    of which ≥1 region is noisy        : {len(samples_labeled_noisy):,}")
    print(f"    of which all regions clean          : {len(samples_labeled_clean):,}")
    print(f"  Samples skipped (no region in Issues): {n_skip:,}")


if __name__ == "__main__":
    main()
