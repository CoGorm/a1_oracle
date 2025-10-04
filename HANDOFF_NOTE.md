# Quick Handoff Note

## What You're Getting

A **broken-but-close** ArUco visual SLAM oracle that needs debugging. The architecture is sound, but one parameter choice causes a deadlock that prevents tags from being promoted.

## The 30-Second Explanation

**Problem:** Video has 30+ ArUco markers. We want to build a map and track camera position. But adding new markers using the noisy camera pose causes **map drift** (trajectory jumps to 191 m/s).

**Solution (attempted):** Promote new markers relative to already-mapped "anchor" markers instead of using camera pose. This should eliminate drift.

**Why it fails:** First anchor disappears after 1 second. No backup anchors can be created because `ANCHOR_MIN_VISIBILITY = 5` frames required. Result: 19 markers stuck in quarantine for 50 seconds.

## Files You Need

```
scripts/track_aruco_v3.py      ← The oracle (fix this)
scripts/results_review.py       ← Analysis tool
scripts/README.md               ← Full docs
PACKAGE_SUMMARY.md              ← Quick start
data/tag_unique.MOV             ← Test video
data/camera.xml                 ← Calibration
out/oracle_v3_full.csv          ← Current bad results
out/analysis_report.txt         ← Metrics showing failures
out/*.png                       ← Plots
```

## How to Test

```bash
# Run oracle
uv run --with opencv-python --with numpy --with scipy \
    scripts/track_aruco_v3.py \
    --video data/tag_unique.MOV \
    --camera data/camera.xml \
    --output out/test.csv

# Analyze
uv run --with pandas --with numpy --with matplotlib \
    scripts/results_review.py \
    --trajectory out/test.csv \
    --output-dir out/
```

Check `out/analysis_report.txt` for success/fail.

## The Easiest Fix (Probably)

In `scripts/track_aruco_v3.py` around line 200, change:

```python
# FROM THIS:
if anchor.visibility_count < ANCHOR_MIN_VISIBILITY:
    continue

# TO THIS:
min_vis_threshold = 1 if self.frame_idx < 100 else ANCHOR_MIN_VISIBILITY
if anchor.visibility_count < min_vis_threshold:
    continue
```

This allows newly-promoted tags to immediately become anchors during bootstrap phase.

**Expected result:** 15-20 tags mapped, speeds drop to < 50 m/s (better but maybe not perfect).

## Success Looks Like

Run analysis and see:
- ✅ Max speed < 5 m/s (currently 191 m/s)
- ✅ 25-35 tags mapped (currently 9)
- ✅ No trajectory jumps in plots

## Three Questions to Answer

1. **Does Option 1 fix work?** Try it first (30 min)
2. **If not, why not?** Check console output for promotion patterns
3. **What's the next step?** See "Potential Fixes" in PACKAGE_SUMMARY.md

## Documentation Hierarchy

1. **Start here:** PACKAGE_SUMMARY.md (5 min)
2. **Then read:** scripts/README.md (30 min)
3. **Finally check:** Inline code comments (as needed)

## Current State

```
Baseline v3 Results (FAIL):
├─ Max speed: 191.5 m/s          ❌ Target: < 5 m/s
├─ Speed violations: 435 frames   ❌ Target: 0
├─ Tags mapped: 9                 ❌ Target: 25-35
└─ Coverage: 41%                  ⚠️  Target: > 80%
```

## Key Insight

The problem is **not** in the anchor-relative math (that works). It's in the **anchor selection policy** - too conservative, causes bootstrap deadlock.

Good luck! 🎯

---

**P.S.** The code is well-commented. Read the docstrings in `track_aruco_v3.py` for the full algorithm explanation.

