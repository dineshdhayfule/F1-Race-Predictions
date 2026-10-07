"""
Script to seed raw data directory from existing datasets.
Copies 2026 season rounds and circuit session data into f1-race-prediction/data/raw/.
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_SEPANG = ROOT.parent / "sepang race prediction" / "data" / "processed"
SRC_BAKU = ROOT.parent / "f1-baku-prediction" / "data" / "processed"
DEST_RAW = ROOT / "data" / "raw"

# Target directories
season_2026_dir = DEST_RAW / "season" / "2026"
sepang_2026_dir = DEST_RAW / "tracks" / "sepang" / "2026"
baku_2026_dir = DEST_RAW / "tracks" / "baku" / "2026"
baku_hist_dir = DEST_RAW / "tracks" / "baku" / "history"

season_2026_dir.mkdir(parents=True, exist_ok=True)
sepang_2026_dir.mkdir(parents=True, exist_ok=True)
baku_2026_dir.mkdir(parents=True, exist_ok=True)
baku_hist_dir.mkdir(parents=True, exist_ok=True)

if not SRC_SEPANG.exists() and not SRC_BAKU.exists():
    print("Legacy directories absent. Data in f1-race-prediction/data/raw is already self-contained.")
    exit(0)

# 1. Copy season rounds (rounds 1-15 race, quali, fp1-fp3)
copied_season = 0
for f in SRC_SEPANG.glob("2026_round_*"):
    shutil.copy2(f, season_2026_dir / f.name)
    copied_season += 1
print(f"Copied {copied_season} season round files to {season_2026_dir}")

# 2. Copy Sepang 2026 weekend
for session in ["fp1", "fp2", "fp3", "qualifying"]:
    src_file = SRC_SEPANG / f"2026_sepang_{session}.csv"
    if src_file.exists():
        shutil.copy2(src_file, sepang_2026_dir / f"2026_sepang_{session}.csv")
        print(f"Copied Sepang {session}")

# 3. Copy Baku 2026 weekend
for session in ["fp1", "fp2", "fp3", "qualifying", "race"]:
    src_file = SRC_SEPANG / f"2026_baku_{session}.csv"
    if src_file.exists():
        shutil.copy2(src_file, baku_2026_dir / f"2026_baku_{session}.csv")
        print(f"Copied Baku {session}")

# 4. Copy Baku historical data if present in baku project
for f in SRC_BAKU.glob("*.csv"):
    if "historical" in f.name.lower() or any(yr in f.name for yr in ["2021", "2022", "2023", "2024", "2025"]):
        shutil.copy2(f, baku_hist_dir / f.name)
        print(f"Copied Baku history: {f.name}")

print("Data seeding completed successfully.")
