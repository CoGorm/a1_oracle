# ArUco Visual Navigation Oracle v3 - Review Package

## Overview

This package contains a Python-based ArUco marker tracking system designed to establish a **ground truth trajectory** for visual SLAM evaluation. The oracle processes video footage of ArUco markers and estimates camera position over time.

## The Problem We're Solving

**Visual SLAM with ArUco markers** faces a critical challenge: **map drift**. When new markers are added to the map using the current (noisy) camera pose estimate, errors accumulate and the map becomes inconsistent, causing:
- Trajectory jumps (speeds > 100 m/s between frames)
- Map inconsistency (landmarks drift from true positions)
- Poor long-term tracking (only ~9 tags mapped instead of 30+)

### Current Status

**Oracle v3** implements **multi-reference anchor-relative promotion**:
- ✅ Tags are promoted relative to already-mapped "anchor" tags (not the drifting camera pose)
- ✅ Multiple anchors can be used (switches when primary anchor not visible)
- ✅ Robust consistency gating (translation scatter, rotation scatter)
- ❌ **Still exhibits map drift** (max speed 191 m/s, trajectory jumps)
- ❌ **Only 9 tags mapped** (out of ~30 visible in video)

**Root Cause:** The bootstrap anchor (first tag) disappears quickly, leaving no eligible reference for ~50 seconds. Quarantine fills with 19 tags that can't be promoted until an anchor returns.

## Goal

**Achieve a stable, drift-free trajectory** with:
- Max speed < 5 m/s (reasonable for handheld camera)
- Smooth trajectory (no jumps)
- All visible tags successfully mapped (25-35 tags)
- Consistent map (tags don't move once mapped)

## Files Provided

```
scripts/
  ├── track_aruco_v3.py       # Main oracle implementation
  ├── results_review.py        # Analysis and visualization tool
  └── README.md                # This file

data/
  ├── tag_unique.MOV           # Input video (~2 minutes, 30fps)
  ├── camera.xml               # Camera calibration (OpenCV format)
  └── config.xml               # Calibration metadata

out/                            # Output directory for results
```

## Quick Start

### Requirements

Python 3.8+ with:
- opencv-python (with contrib for ArUco)
- numpy
- scipy
- matplotlib
- pandas

Install via uv (recommended):
```bash
uv run --with opencv-python --with numpy --with scipy --with matplotlib --with pandas <script.py>
```

Or via pip:
```bash
pip3 install opencv-contrib-python numpy scipy matplotlib pandas
```

### Running the Oracle

```bash
# Process full video
uv run --with opencv-python --with numpy --with scipy \
    scripts/track_aruco_v3.py \
    --video data/tag_unique.MOV \
    --camera data/camera.xml \
    --output out/oracle_v3_full.csv

# Process 10-second clip (for quick testing)
# Note: Extract clip first with: ffmpeg -i data/tag_unique.MOV -t 10 data/tag_unique_10s.MOV
uv run --with opencv-python --with numpy --with scipy \
    scripts/track_aruco_v3.py \
    --video data/tag_unique_10s.MOV \
    --camera data/camera.xml \
    --output out/oracle_v3_10s.csv
```

### Analyzing Results

```bash
# Analyze trajectory for speed violations and map quality
uv run --with pandas --with numpy --with matplotlib \
    scripts/results_review.py \
    --trajectory out/oracle_v3_full.csv \
    --output-dir out/
```

This generates:
- `out/analysis_report.txt` - Detailed metrics
- `out/trajectory_plot.png` - XY/XZ/YZ trajectory views
- `out/speed_profile.png` - Speed over time with violation markers
- `out/diagnostics.png` - Distance, acceleration, turn rate

## Architecture

### Core Components

**1. Bootstrap Seeding**
- First detected tag becomes world origin
- Camera pose computed from `tag_T_camera`^-1
- Problem: If this tag disappears quickly, no anchor for promotion

**2. Anchor Pool Management**
- Tracks all mapped tags as potential anchors
- Quality score based on visibility count + recency
- Selects best visible anchor for promotion
- Problem: Requires 5+ visibility frames (too strict)

**3. Anchor-Relative Promotion**
- When anchor + candidate both visible:
  - Solve PnP for both tags → `camera_T_anchor`, `camera_T_candidate`
  - Compute `anchor_T_candidate = (camera_T_anchor)^-1 ⊗ camera_T_candidate`
  - Accumulate in sliding window (max 30 frames)
  - Check consistency (translation scatter < 2cm, rotation scatter < 2°)
  - Promote if ≥3 consistent frames: `world_T_candidate = world_T_anchor ⊗ anchor_T_candidate`

**4. Camera Pose Estimation**
- Multi-tag PnP RANSAC using all visible mapped tags
- Combines observations for robustness
- Used for trajectory output only (not for promotion)

### Key Parameters (in `track_aruco_v3.py`)

```python
# Anchor-relative promotion
ANCHOR_REL_MIN_FRAMES = 3           # Require 3 co-visible frames
ANCHOR_REL_MAX_WINDOW = 30          # Sliding window size
ANCHOR_REL_RMSE_PX = 2.5           # Per-tag RMSE < 2.5px
ANCHOR_REL_TRANS_SCATTER_M = 0.02   # Translation scatter < 2cm
ANCHOR_REL_ROT_SCATTER_DEG = 2.0    # Rotation scatter < 2°

# Anchor quality scoring
ANCHOR_MIN_VISIBILITY = 5           # Must be seen in ≥5 frames
ANCHOR_QUALITY_DECAY = 0.95         # Exponential decay for old anchors
```

## Current Issues & Potential Solutions

### Issue 1: Bootstrap Anchor Disappears

**Problem:** Tag 100 (bootstrap anchor) visible for ~30 frames, then disappears for 1500 frames. No eligible anchor → 19 tags stuck in quarantine.

**Potential Solutions:**
1. **Aggressive early promotion** - Lower `ANCHOR_MIN_VISIBILITY` to 1 for first 100 frames
2. **Multi-tag bootstrap** - Promote 3-5 tags during bootstrap before anchor disappears
3. **Vision-based fallback** - Allow promotion using camera pose when no anchor available (less stable but better than nothing)
4. **Delayed bootstrap** - Buffer first N frames, select tag with best long-term visibility

### Issue 2: Sparse Map Coverage

**Problem:** Only 9 tags mapped (should be 25-35). Most tags never promoted.

**Potential Solutions:**
1. Enable fallback promotion when anchor gap > 10 seconds
2. Relax consistency gates during low-anchor-count phases
3. Use recently-promoted tags as anchors immediately (don't wait for visibility count)

### Issue 3: Speed Violations Remain

**Problem:** Max speed 191 m/s at frames 4-22 (early in video).

**Analysis needed:**
- Is this during bootstrap before first anchor is established?
- Or during the long anchor gap (frames 30-1900)?
- Check `out/oracle_v3_full.csv` timestamps for violation locations

## Performance Metrics

### Good Oracle Characteristics
- ✅ Max speed < 5 m/s
- ✅ Mean speed < 1 m/s
- ✅ Speed violations: 0 frames
- ✅ Mapped tags: 25-35
- ✅ Smooth trajectory (no discontinuities)
- ✅ Consistent map (promoted tag positions don't change)

### Current v3 Performance
- ❌ Max speed: 191.5 m/s
- ❌ Speed >10 m/s: 435 frames
- ❌ Speed >50 m/s: 364 frames
- ❌ Mapped tags: 9
- ⚠️ Trajectory points: 1419/3432 (41% coverage)

## Debugging Tips

### Enable Verbose Logging

Modify `track_aruco_v3.py` to print anchor selection:
```python
# In process_frame(), after anchor selection:
if anchor_id is not None:
    print(f"  [Frame {self.frame_idx}] Using anchor {anchor_id}")
else:
    print(f"  [Frame {self.frame_idx}] NO ANCHOR AVAILABLE")
```

### Visualize Anchor Timeline

```python
# After processing, analyze anchor usage
import re
log_lines = []  # Collect stdout during processing
anchor_frames = {}
for line in log_lines:
    if match := re.search(r'anchor=(\d+)', line):
        anchor_id = int(match.group(1))
        anchor_frames.setdefault(anchor_id, []).append(frame_idx)
```

### Check Quarantine Health

```python
# In process_frame(), at end:
if self.frame_idx % 100 == 0:
    avg_frames_in_q = np.mean([len(e.anchor_T_tag_hist) for e in self.quarantine.values()])
    print(f"  Quarantine health: avg_frames={avg_frames_in_q:.1f}")
```

## Expected Output Format

CSV with columns:
```
time_s,x_m,y_m,z_m,dist_m
0.000,0.000000,0.000000,0.000000,0.000000
0.033,-0.000234,0.000123,0.001234,0.001290
...
```

Where:
- `time_s`: Timestamp (frame / fps)
- `x_m, y_m, z_m`: Camera position in world frame [meters]
- `dist_m`: Cumulative distance traveled [meters]

## Success Criteria for Fixed Oracle

1. **No speed violations** - Max speed < 5 m/s throughout video
2. **High map coverage** - At least 25 tags successfully mapped
3. **Continuous tracking** - Trajectory covers >90% of frames
4. **Visual validation** - Trajectory plot shows smooth path without jumps
5. **Reproducible** - Multiple runs produce same map + trajectory

## Contact & Questions

If you have questions or make improvements:
- Document changes in `CHANGES.md`
- Include before/after metrics
- Explain reasoning for parameter changes

## References

- OpenCV ArUco: https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
- IPPE solver: Infinitesimal Plane-based Pose Estimation
- SE(3) averaging: Markley et al., "Quaternion Averaging"
- Anchor-relative SLAM: Inspired by multi-robot map merging techniques

