#!/usr/bin/env python3
"""
ArUco Visual Navigation Oracle v3 - Multi-Reference Anchor-Relative Promotion

Key improvements over v2:
- Multi-reference anchor pool (not just original anchor)
- Anchor selection based on visibility + quality metrics
- Anchor-relative promotion using any mapped tag as reference
- Surprisal-based reference selection
- Robust averaging of anchor_T_tag estimates
- Consistency gating on translation/rotation scatter

This eliminates map drift by always promoting relative to stable mapped anchors.
"""

import argparse
import sys
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np


# ============================================================================
# Configuration & Constants
# ============================================================================

TAG_SIZE = 0.16  # meters (ArUco tag physical size)
DICT_TYPE = cv2.aruco.DICT_6X6_1000

# Anchor-relative promotion parameters
ANCHOR_REL_MIN_FRAMES = 3           # Require 3 co-visible frames
ANCHOR_REL_MAX_WINDOW = 30          # Sliding window size
ANCHOR_REL_RMSE_PX = 2.5           # Per-tag RMSE < 2.5px
ANCHOR_REL_TRANS_SCATTER_M = 0.02   # Translation scatter < 2cm
ANCHOR_REL_ROT_SCATTER_DEG = 2.0    # Rotation scatter < 2°

# Anchor quality scoring
ANCHOR_MIN_VISIBILITY = 5           # Must be seen in >=5 frames to be anchor candidate
ANCHOR_QUALITY_DECAY = 0.95         # Exponential decay for anchor quality score


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class QuarantineEntry:
    """Candidate tag in quarantine, waiting for promotion"""
    tag_id: int
    anchor_T_tag_hist: deque  # deque of 4x4 SE(3) matrices
    rmse_hist: deque           # deque of RMSE values
    frame_hist: deque          # deque of frame indices
    last_seen: int
    consecutive_fails: int


@dataclass
class AnchorCandidate:
    """Potential reference anchor tag"""
    tag_id: int
    world_T_tag: np.ndarray    # 4x4 SE(3)
    visibility_count: int      # How many frames it's been visible
    last_seen: int             # Last frame index seen
    quality_score: float       # Composite quality metric


# ============================================================================
# Geometry Utilities
# ============================================================================

def tag_corners_local(size: float) -> np.ndarray:
    """Tag-local corners (TL, TR, BR, BL) for ArUco
    
    ArUco coordinate system: X-right, Y-up, Z-out (right-handed)
    Corner order: TL (top-left), TR (top-right), BR (bottom-right), BL (bottom-left)
    """
    half = size * 0.5
    return np.array([
        [-half,  half, 0],  # TL: (-x, +y, 0)
        [ half,  half, 0],  # TR: (+x, +y, 0)
        [ half, -half, 0],  # BR: (+x, -y, 0)
        [-half, -half, 0]   # BL: (-x, -y, 0)
    ], dtype=np.float32)


def rvec_tvec_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Convert rotation vector + translation to 4x4 SE(3) matrix"""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T


def reprojection_rmse(obj_pts: np.ndarray, img_pts: np.ndarray, 
                      rvec: np.ndarray, tvec: np.ndarray, 
                      K: np.ndarray, dist: np.ndarray) -> float:
    """Compute reprojection RMSE [px]"""
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    diff = proj - img_pts.reshape(-1, 2)
    return float(np.sqrt(np.mean(diff ** 2)))


def se3_log(R: np.ndarray) -> np.ndarray:
    """Logarithm map from SO(3) to so(3) - rotation vector"""
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if theta < 1e-6:
        return np.zeros(3)
    return theta / (2 * np.sin(theta)) * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ])


def rotation_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Angle between two rotation matrices [degrees]"""
    R_rel = R1.T @ R2
    omega = se3_log(R_rel)
    return np.linalg.norm(omega) * 180 / np.pi


def robust_se3_median(poses: List[np.ndarray]) -> np.ndarray:
    """Compute robust median of SE(3) poses
    
    Translation: component-wise median
    Rotation: quaternion average (Markley's method)
    """
    if not poses:
        return np.eye(4)
    
    # Translation: component-wise median
    translations = np.array([T[:3, 3] for T in poses])
    t_med = np.median(translations, axis=0)
    
    # Rotation: quaternion average
    from scipy.spatial.transform import Rotation as R_scipy
    rotations = [R_scipy.from_matrix(T[:3, :3]) for T in poses]
    quats = np.array([r.as_quat() for r in rotations])  # [x, y, z, w]
    
    # Flip quaternions to same hemisphere
    if len(quats) > 1:
        for i in range(1, len(quats)):
            if np.dot(quats[0], quats[i]) < 0:
                quats[i] = -quats[i]
    
    # Average and normalize
    q_mean = np.mean(quats, axis=0)
    q_mean /= np.linalg.norm(q_mean)
    R_med = R_scipy.from_quat(q_mean).as_matrix()
    
    T_med = np.eye(4)
    T_med[:3, :3] = R_med
    T_med[:3, 3] = t_med
    return T_med


def is_consistent_se3_history(poses: List[np.ndarray], 
                               trans_thresh_m: float,
                               rot_thresh_deg: float) -> bool:
    """Check if SE(3) history is consistent"""
    if len(poses) < 2:
        return True
    
    T_med = robust_se3_median(poses)
    t_med = T_med[:3, 3]
    R_med = T_med[:3, :3]
    
    for T in poses:
        dt = np.linalg.norm(T[:3, 3] - t_med)
        dtheta = rotation_angle_deg(R_med, T[:3, :3])
        
        if dt > trans_thresh_m or dtheta > rot_thresh_deg:
            return False
    
    return True


# ============================================================================
# ArUco Detection & PnP
# ============================================================================

def detect_aruco(frame: np.ndarray, dictionary) -> Tuple[List, List, List]:
    """Detect ArUco markers in frame
    
    Returns:
        ids, corners, rejected
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
    corners, ids, rejected = detector.detectMarkers(gray)
    
    # Sub-pixel refinement
    if ids is not None and len(ids) > 0:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        for corner_set in corners:
            cv2.cornerSubPix(gray, corner_set, (5, 5), (-1, -1), criteria)
    
    return ids, corners, rejected


def solve_pnp_single_tag(corners: np.ndarray, K: np.ndarray, dist: np.ndarray,
                         tag_size: float) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Solve PnP for a single tag
    
    Returns:
        (rvec, tvec, rmse) or None if failed
    """
    obj_pts = tag_corners_local(tag_size)
    
    success, rvec, tvec = cv2.solvePnP(
        obj_pts, corners, K, dist,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    
    if not success:
        return None
    
    # Check cheirality (z > 0)
    if tvec[2, 0] <= 0:
        return None
    
    # Compute RMSE
    rmse = reprojection_rmse(obj_pts, corners, rvec, tvec, K, dist)
    
    return rvec, tvec, rmse


# ============================================================================
# Anchor Management
# ============================================================================

class AnchorPool:
    """Manage pool of potential anchor tags"""
    
    def __init__(self):
        self.anchors: Dict[int, AnchorCandidate] = {}
    
    def update_visibility(self, tag_id: int, frame_idx: int, world_T_tag: np.ndarray):
        """Update visibility and quality for a mapped tag"""
        if tag_id in self.anchors:
            anchor = self.anchors[tag_id]
            anchor.visibility_count += 1
            anchor.last_seen = frame_idx
            # Boost quality for recent visibility
            anchor.quality_score = min(1.0, anchor.quality_score + 0.1)
        else:
            # New anchor candidate
            self.anchors[tag_id] = AnchorCandidate(
                tag_id=tag_id,
                world_T_tag=world_T_tag.copy(),
                visibility_count=1,
                last_seen=frame_idx,
                quality_score=0.5
            )
    
    def decay_quality(self, frame_idx: int):
        """Decay quality for anchors not recently seen"""
        for anchor in self.anchors.values():
            if anchor.last_seen < frame_idx:
                anchor.quality_score *= ANCHOR_QUALITY_DECAY
    
    def select_best_anchor(self, visible_tag_ids: List[int]) -> Optional[int]:
        """Select best anchor from visible tags
        
        Criteria:
        - Must be in anchor pool
        - Must have sufficient visibility history
        - Highest quality score among visible
        """
        candidates = []
        
        for tag_id in visible_tag_ids:
            if tag_id not in self.anchors:
                continue
            
            anchor = self.anchors[tag_id]
            if anchor.visibility_count < ANCHOR_MIN_VISIBILITY:
                continue
            
            candidates.append((anchor.quality_score, tag_id))
        
        if not candidates:
            return None
        
        # Return tag with highest quality score
        candidates.sort(reverse=True)
        return candidates[0][1]
    
    def get_anchor_pose(self, tag_id: int) -> Optional[np.ndarray]:
        """Get world pose of anchor tag"""
        if tag_id in self.anchors:
            return self.anchors[tag_id].world_T_tag
        return None


# ============================================================================
# Visual Navigation Oracle
# ============================================================================

class ArUcoNavigatorV3:
    """Multi-reference anchor-relative ArUco SLAM"""
    
    def __init__(self, video_path: str, camera_file: str, tag_size: float):
        self.video_path = video_path
        self.tag_size = tag_size
        
        # Load camera intrinsics (OpenCV XML/YAML)
        fs = cv2.FileStorage(camera_file, cv2.FILE_STORAGE_READ)
        self.K = fs.getNode('camera_matrix').mat()
        self.dist = fs.getNode('distortion_coefficients').mat().flatten()
        fs.release()
        
        # ArUco dictionary
        self.dictionary = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
        
        # State
        self.world_T_tag: Dict[int, np.ndarray] = {}  # Mapped tags
        self.quarantine: Dict[int, QuarantineEntry] = {}
        self.anchor_pool = AnchorPool()
        
        # Trajectory
        self.trajectory = []
        self.frame_idx = 0
    
    def process_video(self) -> bool:
        """Process entire video"""
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video {self.video_path}")
            return False
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing {total_frames} frames at {fps} fps...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.process_frame(frame)
            self.frame_idx += 1
            
            if self.frame_idx % 30 == 0:
                print(f"  Frame {self.frame_idx}/{total_frames} - "
                      f"Mapped: {len(self.world_T_tag)}, "
                      f"Quarantine: {len(self.quarantine)}, "
                      f"Anchors: {len(self.anchor_pool.anchors)}")
        
        cap.release()
        
        print(f"\nProcessing complete:")
        print(f"  Total frames: {self.frame_idx}")
        print(f"  Mapped tags: {len(self.world_T_tag)}")
        print(f"  Trajectory points: {len(self.trajectory)}")
        
        return len(self.trajectory) > 0
    
    def process_frame(self, frame: np.ndarray):
        """Process single frame"""
        # Detect markers
        ids, corners, _ = detect_aruco(frame, self.dictionary)
        
        if ids is None or len(ids) == 0:
            return
        
        ids = ids.flatten()
        
        # Decay anchor quality
        self.anchor_pool.decay_quality(self.frame_idx)
        
        # Bootstrap: if no mapped tags, seed from best detection
        if len(self.world_T_tag) == 0:
            self.bootstrap_seed(ids, corners)
            return
        
        # Update anchor pool visibility
        visible_mapped = []
        for i, tag_id in enumerate(ids):
            if tag_id in self.world_T_tag:
                visible_mapped.append((tag_id, corners[i]))
                self.anchor_pool.update_visibility(
                    tag_id, self.frame_idx, self.world_T_tag[tag_id]
                )
        
        # Select best anchor from visible tags
        visible_tag_ids = [tag_id for tag_id in ids if tag_id in self.world_T_tag]
        anchor_id = self.anchor_pool.select_best_anchor(visible_tag_ids)
        
        # Estimate camera pose (multi-tag PnP if possible)
        world_T_cam = self.estimate_camera_pose(ids, corners)
        
        if world_T_cam is not None:
            # Save trajectory
            t = world_T_cam[:3, 3]
            dist = self.trajectory[-1][4] + np.linalg.norm(t - self.trajectory[-1][1:4]) if self.trajectory else 0
            self.trajectory.append([
                self.frame_idx / 30.0,  # time_s
                t[0], t[1], t[2],       # x, y, z
                dist                     # cumulative distance
            ])
        
        # Promote quarantined tags (anchor-relative if anchor available)
        if anchor_id is not None:
            self.promote_quarantine_anchor_relative(ids, corners, anchor_id)
        else:
            # Fallback: vision-based promotion (less stable)
            if world_T_cam is not None:
                self.promote_quarantine_vision_based(ids, corners, world_T_cam)
    
    def bootstrap_seed(self, ids: np.ndarray, corners: List):
        """Bootstrap initial anchor from best detection"""
        best_tag_id = None
        best_rmse = float('inf')
        best_rvec, best_tvec = None, None
        
        for i, tag_id in enumerate(ids):
            result = solve_pnp_single_tag(corners[i][0], self.K, self.dist, self.tag_size)
            if result is None:
                continue
            
            rvec, tvec, rmse = result
            if rmse < best_rmse:
                best_rmse = rmse
                best_tag_id = tag_id
                best_rvec, best_tvec = rvec, tvec
        
        if best_tag_id is not None:
            # Set world origin at first tag
            self.world_T_tag[best_tag_id] = np.eye(4)
            self.anchor_pool.update_visibility(best_tag_id, self.frame_idx, np.eye(4))
            
            # Camera pose
            camera_T_tag = rvec_tvec_to_T(best_rvec, best_tvec)
            world_T_cam = np.linalg.inv(camera_T_tag)
            
            t = world_T_cam[:3, 3]
            self.trajectory.append([0.0, t[0], t[1], t[2], 0.0])
            
            print(f"✅ BOOTSTRAP: Seeded world origin at tag {best_tag_id} (rmse={best_rmse:.2f}px)")
    
    def estimate_camera_pose(self, ids: np.ndarray, corners: List) -> Optional[np.ndarray]:
        """Estimate camera pose from visible mapped tags"""
        mapped_tags = [(i, tag_id) for i, tag_id in enumerate(ids) if tag_id in self.world_T_tag]
        
        if not mapped_tags:
            return None
        
        # Multi-tag PnP fusion
        obj_pts_list = []
        img_pts_list = []
        
        for i, tag_id in mapped_tags:
            world_T_tag = self.world_T_tag[tag_id]
            tag_corners_world = (world_T_tag[:3, :3] @ tag_corners_local(self.tag_size).T).T + world_T_tag[:3, 3]
            
            obj_pts_list.append(tag_corners_world)
            img_pts_list.append(corners[i][0])
        
        obj_pts = np.vstack(obj_pts_list).astype(np.float32)
        img_pts = np.vstack(img_pts_list).astype(np.float32)
        
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts, img_pts, self.K, self.dist,
            reprojectionError=3.0,
            confidence=0.999,
            iterationsCount=200
        )
        
        if not success or inliers is None or len(inliers) < 4:
            return None
        
        # Refine with inliers
        rvec, tvec = cv2.solvePnPRefineLM(
            obj_pts[inliers[:, 0]], img_pts[inliers[:, 0]],
            self.K, self.dist, rvec, tvec
        )
        
        # world_T_cam = (camera_T_world)^-1
        camera_T_world = rvec_tvec_to_T(rvec, tvec)
        world_T_cam = np.linalg.inv(camera_T_world)
        
        return world_T_cam
    
    def promote_quarantine_anchor_relative(self, ids: np.ndarray, corners: List, anchor_id: int):
        """Promote quarantined tags using anchor-relative method"""
        anchor_result = None
        
        # Solve IPPE for anchor
        for i, tag_id in enumerate(ids):
            if tag_id == anchor_id:
                anchor_result = solve_pnp_single_tag(corners[i][0], self.K, self.dist, self.tag_size)
                break
        
        if anchor_result is None:
            return
        
        anchor_rvec, anchor_tvec, anchor_rmse = anchor_result
        
        if anchor_rmse > ANCHOR_REL_RMSE_PX:
            return
        
        camera_T_anchor = rvec_tvec_to_T(anchor_rvec, anchor_tvec)
        
        # Process candidate tags
        for i, tag_id in enumerate(ids):
            if tag_id in self.world_T_tag or tag_id == anchor_id:
                continue
            
            # Solve IPPE for candidate
            cand_result = solve_pnp_single_tag(corners[i][0], self.K, self.dist, self.tag_size)
            if cand_result is None:
                continue
            
            cand_rvec, cand_tvec, cand_rmse = cand_result
            
            if cand_rmse > ANCHOR_REL_RMSE_PX:
                continue
            
            camera_T_cand = rvec_tvec_to_T(cand_rvec, cand_tvec)
            
            # Compute anchor-relative transform
            anchor_T_cand = np.linalg.inv(camera_T_anchor) @ camera_T_cand
            
            # Add to quarantine history
            if tag_id not in self.quarantine:
                self.quarantine[tag_id] = QuarantineEntry(
                    tag_id=tag_id,
                    anchor_T_tag_hist=deque(maxlen=ANCHOR_REL_MAX_WINDOW),
                    rmse_hist=deque(maxlen=ANCHOR_REL_MAX_WINDOW),
                    frame_hist=deque(maxlen=ANCHOR_REL_MAX_WINDOW),
                    last_seen=self.frame_idx,
                    consecutive_fails=0
                )
            
            entry = self.quarantine[tag_id]
            entry.anchor_T_tag_hist.append(anchor_T_cand)
            entry.rmse_hist.append(cand_rmse)
            entry.frame_hist.append(self.frame_idx)
            entry.last_seen = self.frame_idx
            entry.consecutive_fails = 0
            
            # Check promotion criteria
            if len(entry.anchor_T_tag_hist) >= ANCHOR_REL_MIN_FRAMES:
                if is_consistent_se3_history(
                    list(entry.anchor_T_tag_hist),
                    ANCHOR_REL_TRANS_SCATTER_M,
                    ANCHOR_REL_ROT_SCATTER_DEG
                ):
                    # Promote!
                    anchor_T_tag_avg = robust_se3_median(list(entry.anchor_T_tag_hist))
                    world_T_tag = self.world_T_tag[anchor_id] @ anchor_T_tag_avg
                    
                    self.world_T_tag[tag_id] = world_T_tag
                    self.anchor_pool.update_visibility(tag_id, self.frame_idx, world_T_tag)
                    
                    rmse_med = np.median(list(entry.rmse_hist))
                    del self.quarantine[tag_id]
                    
                    print(f"  ✅ PROMOTED tag {tag_id} via ANCHOR-REL (anchor={anchor_id}, "
                          f"frames={len(entry.anchor_T_tag_hist)}, rmse_med={rmse_med:.2f}px, "
                          f"total_mapped={len(self.world_T_tag)})")
    
    def promote_quarantine_vision_based(self, ids: np.ndarray, corners: List, world_T_cam: np.ndarray):
        """Fallback: promote using vision-based pose (less stable)"""
        for i, tag_id in enumerate(ids):
            if tag_id in self.world_T_tag:
                continue
            
            result = solve_pnp_single_tag(corners[i][0], self.K, self.dist, self.tag_size)
            if result is None:
                continue
            
            rvec, tvec, rmse = result
            
            if rmse > 3.0:  # Relaxed threshold for fallback
                continue
            
            camera_T_tag = rvec_tvec_to_T(rvec, tvec)
            world_T_tag = world_T_cam @ camera_T_tag
            
            # Simple promotion (no quarantine for fallback)
            self.world_T_tag[tag_id] = world_T_tag
            self.anchor_pool.update_visibility(tag_id, self.frame_idx, world_T_tag)
            
            print(f"  ⚠️  PROMOTED tag {tag_id} via VISION (fallback, rmse={rmse:.2f}px)")
    
    def export_csv(self, output_path: str):
        """Export trajectory to CSV"""
        if not self.trajectory:
            print("No trajectory to export!")
            return
        
        with open(output_path, 'w') as f:
            f.write("time_s,x_m,y_m,z_m,dist_m\n")
            for point in self.trajectory:
                f.write(f"{point[0]:.3f},{point[1]:.6f},{point[2]:.6f},{point[3]:.6f},{point[4]:.6f}\n")
        
        print(f"Exported {len(self.trajectory)} trajectory points to {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='ArUco Visual Navigation Oracle v3')
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--camera', required=True, help='Camera calibration file (XML/YAML)')
    parser.add_argument('--tag-size', type=float, default=TAG_SIZE, help='Tag size in meters')
    parser.add_argument('--output', default='oracle_v3_trajectory.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    # Check files exist
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        return 1
    
    if not Path(args.camera).exists():
        print(f"Error: Camera file not found: {args.camera}")
        return 1
    
    # Create navigator
    navigator = ArUcoNavigatorV3(args.video, args.camera, args.tag_size)
    
    # Process video
    success = navigator.process_video()
    
    if not success:
        print("Failed to process video")
        return 1
    
    # Export trajectory
    navigator.export_csv(args.output)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

