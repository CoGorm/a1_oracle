# Oracle v3: Multi-Reference Anchor-Relative Promotion

## 🔍 Problem Analysis

### Current Oracle (v2) Issues
```python
# Line 309 in track_aruco_stable.py:
T_W_tag_proposed = T_W_C @ T_C_tag  # ← Uses camera pose (has error!)
```

**Results on full video:**
- ✅ 99.9% pose coverage
- ✅ 32 tags mapped
- ⚠️ **241 frames with speed >10 m/s**
- ⚠️ **Max speed 61.7 m/s** (impossible for handheld!)
- ⚠️ **Jumps around frames 1880-1950**

**Root cause**: Same as C++! Tags promoted using `T_W_C` (camera pose), which accumulates error over time.

---

## 🎯 v3 Design: Multi-Reference Promotion

### Key Idea
**Never use camera pose for promotion!** Instead:
1. Find a **co-visible mapped tag** (reference)
2. Compute **relative transform**: `ref_T_tag = (camera_T_ref)^-1 * camera_T_tag`
3. Promote: `world_T_tag = world_T_ref * ref_T_tag`

This is **drift-free** because it only uses **tag-to-tag relative measurements**, never the (possibly drifted) camera pose!

---

## 📝 Implementation

### Data Structures
```python
# Quarantine entry (per unknown tag)
quarantine[tid] = {
    "ref_id": int,  # Which mapped tag is used as reference
    "ref_T_tag_hist": list,  # Sliding window of relative transforms
    "rmse_hist": list,  # RMSE values for each observation
    "sightings": int,
    "last_frame": int
}
```

### Algorithm (Per Frame)
```python
# 1. Get camera pose (for trajectory only, not for promotion!)
T_W_C = estimate_camera_pose(known_tags)

# 2. For each unknown tag:
for unknown_tid in unknown:
    # Find co-visible mapped tag as reference
    ref_id = None
    for known_tid in known:
        if known_tid in world_map:
            ref_id = known_tid
            break  # Use first co-visible mapped tag
    
    if ref_id is None:
        continue  # Can't promote without a reference
    
    # Solve IPPE for both reference and unknown tag
    camera_T_ref = solve_ippe(ref_corners)
    camera_T_tag = solve_ippe(unknown_corners)
    
    # Gate on RMSE
    rmse_ref = reprojection_rmse(ref_corners, camera_T_ref)
    rmse_tag = reprojection_rmse(unknown_corners, camera_T_tag)
    
    if rmse_ref > 2.5 or rmse_tag > 2.5:
        continue
    
    # Compute relative transform (drift-free!)
    ref_T_tag = inv(camera_T_ref) @ camera_T_tag
    
    # Buffer in quarantine
    if unknown_tid not in quarantine:
        quarantine[unknown_tid] = {
            "ref_id": ref_id,
            "ref_T_tag_hist": [],
            "rmse_hist": [],
            "sightings": 0,
            "last_frame": -1
        }
    
    q = quarantine[unknown_tid]
    q["ref_T_tag_hist"].append(ref_T_tag)
    q["rmse_hist"].append(rmse_tag)
    q["sightings"] += 1
    q["last_frame"] = frame_idx
    
    # Enforce sliding window
    max_window = 30
    if len(q["ref_T_tag_hist"]) > max_window:
        q["ref_T_tag_hist"].pop(0)
        q["rmse_hist"].pop(0)
    
    # Check promotion criteria
    if q["sightings"] >= 3 and len(q["ref_T_tag_hist"]) >= 3:
        # Check consistency
        consistent = is_consistent(q["ref_T_tag_hist"], 
                                   trans_thresh=0.02, 
                                   rot_thresh_deg=2.0)
        
        if consistent:
            # Promote: world_T_tag = world_T_ref * median(ref_T_tag)
            ref_T_tag_avg = robust_se3_average(q["ref_T_tag_hist"])
            world_T_ref = world_map[q["ref_id"]]
            world_map[unknown_tid] = world_T_ref @ ref_T_tag_avg
            
            median_rmse = np.median(q["rmse_hist"])
            print(f"  ✅ Tag {unknown_tid} promoted via MULTI-REF "
                  f"(ref={q['ref_id']}, sightings={q['sightings']}, "
                  f"rmse={median_rmse:.2f}px)")
            
            del quarantine[unknown_tid]
```

### Helper Functions
```python
def is_consistent(T_hist, trans_thresh, rot_thresh_deg):
    """Check if SE(3) history is consistent."""
    if len(T_hist) < 2:
        return True
    
    # Compute median
    t_med, R_med = robust_se3_median(T_hist)
    
    # Check scatter
    for T in T_hist:
        dt = np.linalg.norm(T[:3,3] - t_med)
        if dt > trans_thresh:
            return False
        
        # Rotation distance
        R_delta = R_med.T @ T[:3,:3]
        angle = np.arccos((np.trace(R_delta) - 1) / 2)
        dtheta_deg = np.degrees(angle)
        
        if dtheta_deg > rot_thresh_deg:
            return False
    
    return True

def robust_se3_median(T_hist):
    """Compute median of SE(3) poses."""
    # Translation: component-wise median
    translations = np.array([T[:3,3] for T in T_hist])
    t_med = np.median(translations, axis=0)
    
    # Rotation: average quaternions
    from scipy.spatial.transform import Rotation
    quats = []
    for T in T_hist:
        R = Rotation.from_matrix(T[:3,:3])
        quats.append(R.as_quat())
    
    # Ensure consistent hemisphere
    q_ref = quats[0]
    for i in range(1, len(quats)):
        if np.dot(quats[i], q_ref) < 0:
            quats[i] = -quats[i]
    
    q_avg = np.mean(quats, axis=0)
    q_avg /= np.linalg.norm(q_avg)
    
    R_med = Rotation.from_quat(q_avg).as_matrix()
    
    return t_med, R_med

def robust_se3_average(T_hist):
    """Compute robust average SE(3)."""
    t_avg, R_avg = robust_se3_median(T_hist)
    T_avg = np.eye(4)
    T_avg[:3,:3] = R_avg
    T_avg[:3,3] = t_avg
    return T_avg
```

---

## 📈 Expected Results

### Before (v2 - Vision-Based Promotion)
```
Frames with speed >10 m/s: 241
Max speed: 61.7 m/s
Jumps: Yes (frames 1880-1950)
```

### After (v3 - Multi-Reference Promotion)
```
Frames with speed >10 m/s: <10
Max speed: <5 m/s
Jumps: No (drift-free map!)
```

---

## 🔬 Testing Plan

### Test 1: Run v3 on Full Video
```bash
uv run scripts/track_aruco_v3.py \
  --video data/tag_unique.MOV \
  --camera data/camera_auto.yaml \
  --tag-size 0.16 \
  --dict DICT_6X6_1000 \
  --out-csv oracle_v3_full.csv
```

**Expected**:
- 99.9% pose coverage ✅
- 30+ tags mapped ✅
- <10 speed gates >10 m/s ✅
- Max speed <5 m/s ✅
- Total distance ~5-10m (not 292m!) ✅

### Test 2: Compare v2 vs v3
```python
import pandas as pd
import numpy as np

v2 = pd.read_csv('oracle_full_check.csv')
v3 = pd.read_csv('oracle_v3_full.csv')

for name, traj in [('v2', v2), ('v3', v3)]:
    positions = traj[['x_m', 'y_m', 'z_m']].values
    speeds = np.linalg.norm(np.diff(positions, axis=0), axis=1) / (1/30)
    
    print(f"{name}:")
    print(f"  Max speed: {speeds.max():.1f} m/s")
    print(f"  Frames >10 m/s: {(speeds > 10).sum()}")
    print(f"  Total distance: {traj['dist_m'].iloc[-1]:.1f} m")
```

### Test 3: Visual Inspection
```python
import matplotlib.pyplot as plt

v2 = pd.read_csv('oracle_full_check.csv')
v3 = pd.read_csv('oracle_v3_full.csv')

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# XY view
axes[0,0].plot(v2['x_m'], v2['y_m'], 'r-', alpha=0.5, label='v2')
axes[0,0].plot(v3['x_m'], v3['y_m'], 'g-', alpha=0.5, label='v3')
axes[0,0].set_xlabel('X (m)')
axes[0,0].set_ylabel('Y (m)')
axes[0,0].set_title('Top View (XY)')
axes[0,0].legend()
axes[0,0].grid(True)

# XZ view
axes[0,1].plot(v2['x_m'], v2['z_m'], 'r-', alpha=0.5, label='v2')
axes[0,1].plot(v3['x_m'], v3['z_m'], 'g-', alpha=0.5, label='v3')
axes[0,1].set_xlabel('X (m)')
axes[0,1].set_ylabel('Z (m)')
axes[0,1].set_title('Front View (XZ)')
axes[0,1].legend()
axes[0,1].grid(True)

# Speed over time
times_v2 = v2['time_s'].values
speeds_v2 = np.concatenate([[0], np.linalg.norm(np.diff(v2[['x_m','y_m','z_m']].values, axis=0), axis=1) / (1/30)])

times_v3 = v3['time_s'].values
speeds_v3 = np.concatenate([[0], np.linalg.norm(np.diff(v3[['x_m','y_m','z_m']].values, axis=0), axis=1) / (1/30)])

axes[1,0].plot(times_v2, speeds_v2, 'r-', alpha=0.5, label='v2')
axes[1,0].plot(times_v3, speeds_v3, 'g-', alpha=0.5, label='v3')
axes[1,0].axhline(10, color='k', linestyle='--', label='10 m/s threshold')
axes[1,0].set_xlabel('Time (s)')
axes[1,0].set_ylabel('Speed (m/s)')
axes[1,0].set_title('Speed over Time')
axes[1,0].legend()
axes[1,0].grid(True)
axes[1,0].set_ylim([0, min(speeds_v2.max(), 70)])

# Distance over time
axes[1,1].plot(v2['time_s'], v2['dist_m'], 'r-', alpha=0.5, label='v2')
axes[1,1].plot(v3['time_s'], v3['dist_m'], 'g-', alpha=0.5, label='v3')
axes[1,1].set_xlabel('Time (s)')
axes[1,1].set_ylabel('Distance (m)')
axes[1,1].set_title('Cumulative Distance')
axes[1,1].legend()
axes[1,1].grid(True)

plt.tight_layout()
plt.savefig('oracle_v2_vs_v3_comparison.png', dpi=150)
print("Saved: oracle_v2_vs_v3_comparison.png")
```

---

## 🎯 Implementation Steps

### Step 1: Copy and Rename
```bash
cp scripts/track_aruco_stable.py scripts/track_aruco_v3.py
```

### Step 2: Add Helper Functions
Add `robust_se3_median()`, `robust_se3_average()`, `is_consistent()` at the top of the file.

### Step 3: Modify Quarantine Structure
Change quarantine dict to include:
- `ref_id`: Reference tag ID
- `ref_T_tag_hist`: Sliding window of relative transforms
- `rmse_hist`: RMSE values

### Step 4: Modify Promotion Logic
Replace lines 289-327 with multi-reference logic.

### Step 5: Test
Run on both 10s and full videos, compare to v2.

---

## 💡 Key Advantages of v3

### 1. **Drift-Free Map Building**
- Only uses tag-to-tag measurements
- Never uses camera pose (which has accumulated error)

### 2. **Scalable**
- Works with ANY co-visible mapped tag as reference
- Not limited to original anchor

### 3. **Robust**
- Consistency gating ensures bad measurements are rejected
- Sliding window + median averaging reduces noise

### 4. **Simple**
- Clean conceptual model: "measure new landmarks relative to known ones"
- Easy to debug and verify

---

## 🚀 Bottom Line

**v2 (current)**:
- ⚠️ Uses camera pose for promotion → accumulates drift
- ⚠️ 241 speed gates, max 61.7 m/s
- ⚠️ 292m total distance (too high!)

**v3 (proposed)**:
- ✅ Uses tag-to-tag measurements → drift-free!
- ✅ <10 speed gates, max <5 m/s
- ✅ ~5-10m total distance (realistic!)
- ✅ Matches hand-held camera motion

**Let's build v3!** 🎯

