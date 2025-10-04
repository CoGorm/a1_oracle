# ArUco Visual Navigation Oracle - External Review Package

## Quick Summary

This package contains a Python implementation of an ArUco marker-based visual SLAM system that attempts to solve the **map drift problem** through **anchor-relative tag promotion**.

**Current Status:** ❌ Not working correctly
- Max speed: 191 m/s (should be < 5 m/s)
- Only 9 tags mapped (should be 25-35)
- 40% of criteria passed

## What You're Getting

```
scripts/
  ├── track_aruco_v3.py       # Oracle implementation (650 lines)
  ├── results_review.py        # Analysis tool (350 lines)
  └── README.md                # Full documentation

data/
  ├── tag_unique.MOV           # Test video (2 minutes, 1080p, 30fps)
  ├── camera.xml               # Calibration data
  └── config.xml               # Metadata

out/
  ├── oracle_v3_full.csv       # Current results (BAD - 191 m/s jumps)
  ├── analysis_report.txt      # Detailed metrics
  ├── trajectory_plot.png      # 3D visualization
  ├── speed_profile.png        # Speed violations chart
  └── diagnostics.png          # Acceleration/turn rate

PACKAGE_SUMMARY.md             # This file
```

## The Problem

**Visual SLAM map drift:** When new tags are promoted to the map using the current (noisy) camera pose estimate, errors compound. This causes:
1. Trajectory jumps (unphysical speeds > 100 m/s)
2. Map inconsistency (tags drift from true positions)
3. Sparse coverage (only 9/30 tags successfully mapped)

## The Solution (Attempted)

**Anchor-relative promotion:** Instead of using the drifting camera pose, compute new tag positions relative to already-mapped "anchor" tags:

```
world_T_new_tag = world_T_anchor ⊗ anchor_T_new_tag
                  ↑              ↑
                  (known, stable) (directly measured via co-visibility)
```

This should eliminate drift by never using the accumulated camera pose for promotion.

## Why It's Failing

**Bootstrap anchor disappears:** The first tag (anchor) is visible for only 30 frames (~1 second), then disappears for 1500 frames (~50 seconds). During this gap:
- 19 tags detected but stuck in quarantine (can't promote without anchor)
- Camera pose estimated from only 1 tag (unstable)
- Trajectory exhibits large jumps

**See frames 4-27 in `out/analysis_report.txt`** - worst violations all occur during this anchor gap.

## What Success Looks Like

Run analysis tool to check:
```bash
uv run --with pandas --with numpy --with matplotlib \
    scripts/results_review.py \
    --trajectory out/YOUR_OUTPUT.csv \
    --output-dir out/
```

**Target metrics:**
- ✅ Max speed < 5 m/s
- ✅ Mean speed < 1 m/s  
- ✅ No frames > 10 m/s
- ✅ 25-35 tags mapped
- ✅ Smooth trajectory (visual inspection of plots)

**Current metrics (v3):**
- ❌ Max speed: 191 m/s
- ❌ Mean speed: 27.7 m/s
- ❌ 435 frames > 10 m/s
- ❌ Only 9 tags mapped
- ❌ Trajectory has discontinuities

## How to Run

```bash
# Process video
uv run --with opencv-python --with numpy --with scipy \
    scripts/track_aruco_v3.py \
    --video data/tag_unique.MOV \
    --camera data/camera.xml \
    --output out/my_attempt.csv

# Analyze results
uv run --with pandas --with numpy --with matplotlib \
    scripts/results_review.py \
    --trajectory out/my_attempt.csv \
    --output-dir out/
```

## Potential Fixes

### Option 1: Aggressive Early Promotion (Simplest)
Lower `ANCHOR_MIN_VISIBILITY` from 5 to 1 for the first 100 frames. This allows newly-promoted tags to become anchors immediately.

**Change in `track_aruco_v3.py` line ~200:**
```python
# OLD:
if anchor.visibility_count < ANCHOR_MIN_VISIBILITY:
    continue

# NEW:
min_vis = 1 if self.frame_idx < 100 else ANCHOR_MIN_VISIBILITY
if anchor.visibility_count < min_vis:
    continue
```

### Option 2: Multi-Tag Bootstrap
Promote 3-5 tags during bootstrap (first 30 frames) before first anchor disappears. This provides backup anchors.

### Option 3: Vision-Based Fallback
When no anchor available for > 10 seconds, allow promotion using camera pose (less stable but better than nothing).

### Option 4: Delayed Bootstrap
Buffer first 200 frames, analyze tag visibility patterns, select tag with best long-term visibility as anchor.

## Key Files to Modify

1. **`scripts/track_aruco_v3.py`** lines 195-210: Anchor selection logic
2. **`scripts/track_aruco_v3.py`** lines 375-395: Bootstrap seeding
3. **`scripts/track_aruco_v3.py`** lines 530-580: Promotion logic

## Documentation

- **`scripts/README.md`** - Full technical documentation (architecture, parameters, debugging)
- **`out/analysis_report.txt`** - Current results with detailed metrics
- **Inline comments** - Extensive code documentation

## Questions to Consider

1. **Why does tag 100 disappear so quickly?** Is camera moving away or tag going out of frame?
2. **Can we predict which tag will remain visible longest?** Buffer analysis?
3. **Should we promote multiple tags simultaneously during bootstrap?** Pro-active backup anchors?
4. **Is vision-based fallback acceptable?** Trade stability for coverage?

## Expected Time Investment

- **Understanding the problem:** 1-2 hours (read README, examine current results)
- **Implementing Option 1:** 30 minutes (one-line change + testing)
- **Implementing Option 2-4:** 2-4 hours (more invasive changes)
- **Validation & iteration:** 1-2 hours (run, analyze, tune parameters)

## Success Indicators

You'll know it's working when:
1. **Console output** shows steady tag promotion (not 1500 frame gaps)
2. **CSV output** has >25 tags worth of trajectory points
3. **Analysis report** shows speeds < 5 m/s, no violations
4. **Trajectory plot** shows smooth path without discontinuities
5. **Reproducible** - multiple runs produce similar results

## Contact

If you fix this or have questions, please document:
- What you changed (file, line numbers, reasoning)
- Before/after metrics from `analysis_report.txt`
- Any new parameters added

Good luck! 🚀

