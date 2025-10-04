#!/usr/bin/env python3
"""
ArUco Oracle Results Review Tool

Analyzes trajectory CSV and generates diagnostic plots and metrics.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_trajectory(csv_path):
    """Load trajectory from CSV"""
    df = pd.read_csv(csv_path)
    
    # Validate columns
    required = ['time_s', 'x_m', 'y_m', 'z_m', 'dist_m']
    if not all(col in df.columns for col in required):
        raise ValueError(f"CSV must contain columns: {required}")
    
    return df


def compute_speeds(df, fps=30.0):
    """Compute frame-to-frame speeds [m/s]"""
    dt = 1.0 / fps
    
    positions = df[['x_m', 'y_m', 'z_m']].values
    diffs = np.diff(positions, axis=0)
    speeds = np.linalg.norm(diffs, axis=1) / dt
    
    return speeds


def compute_accelerations(speeds, fps=30.0):
    """Compute frame-to-frame accelerations [m/s²]"""
    dt = 1.0 / fps
    return np.diff(speeds) / dt


def compute_turn_rates(df, fps=30.0):
    """Compute frame-to-frame turn rates [deg/s]"""
    dt = 1.0 / fps
    
    positions = df[['x_m', 'y_m', 'z_m']].values
    diffs = np.diff(positions, axis=0)
    
    # Heading in XY plane
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])  # radians
    heading_diffs = np.diff(headings)
    
    # Wrap to [-pi, pi]
    heading_diffs = np.arctan2(np.sin(heading_diffs), np.cos(heading_diffs))
    
    turn_rates = np.abs(heading_diffs) * 180 / np.pi / dt  # deg/s
    
    return turn_rates


def print_report(df, speeds, output_dir):
    """Print detailed analysis report"""
    report_path = Path(output_dir) / 'analysis_report.txt'
    
    with open(report_path, 'w') as f:
        def write(s):
            print(s)
            f.write(s + '\n')
        
        write("=" * 70)
        write("ArUco Oracle Trajectory Analysis Report")
        write("=" * 70)
        write("")
        
        # Basic info
        write("BASIC INFORMATION")
        write("-" * 70)
        write(f"  Total trajectory points: {len(df)}")
        write(f"  Time range: {df['time_s'].iloc[0]:.2f}s to {df['time_s'].iloc[-1]:.2f}s")
        write(f"  Duration: {df['time_s'].iloc[-1] - df['time_s'].iloc[0]:.2f}s")
        write(f"  Total distance: {df['dist_m'].iloc[-1]:.2f} m")
        write("")
        
        # Position statistics
        write("POSITION STATISTICS")
        write("-" * 70)
        write(f"  X range: [{df['x_m'].min():.3f}, {df['x_m'].max():.3f}] m")
        write(f"  Y range: [{df['y_m'].min():.3f}, {df['y_m'].max():.3f}] m")
        write(f"  Z range: [{df['z_m'].min():.3f}, {df['z_m'].max():.3f}] m")
        write(f"  Max distance from origin: {np.max(np.linalg.norm(df[['x_m', 'y_m', 'z_m']].values, axis=1)):.3f} m")
        write("")
        
        # Speed statistics
        write("SPEED STATISTICS")
        write("-" * 70)
        write(f"  Mean speed: {np.mean(speeds):.2f} m/s")
        write(f"  Median speed: {np.median(speeds):.2f} m/s")
        write(f"  Std dev: {np.std(speeds):.2f} m/s")
        write(f"  Max speed: {np.max(speeds):.2f} m/s")
        write(f"  95th percentile: {np.percentile(speeds, 95):.2f} m/s")
        write("")
        
        # Speed violations
        write("SPEED VIOLATIONS (potential trajectory jumps)")
        write("-" * 70)
        
        thresholds = [3, 5, 10, 50, 100]
        for thresh in thresholds:
            count = np.sum(speeds > thresh)
            pct = 100 * count / len(speeds) if len(speeds) > 0 else 0
            status = "✅ PASS" if count == 0 else "❌ FAIL"
            write(f"  Speed > {thresh:3d} m/s: {count:4d} frames ({pct:5.1f}%) {status}")
        write("")
        
        # Worst violations
        if np.max(speeds) > 5:
            write("WORST SPEED VIOLATIONS")
            write("-" * 70)
            worst_idx = np.argsort(speeds)[-min(10, len(speeds)):]
            write(f"  {'Frame':<8} {'Time (s)':<10} {'Speed (m/s)':<12} {'Position (m)'}")
            write(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*30}")
            for idx in worst_idx[::-1]:
                time = df['time_s'].iloc[idx]
                speed = speeds[idx]
                pos = df[['x_m', 'y_m', 'z_m']].iloc[idx].values
                write(f"  {idx:<8d} {time:<10.2f} {speed:<12.2f} [{pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}]")
            write("")
        
        # Success criteria
        write("SUCCESS CRITERIA")
        write("-" * 70)
        
        criteria = [
            ("Max speed < 5 m/s", np.max(speeds) < 5),
            ("Mean speed < 1 m/s", np.mean(speeds) < 1),
            ("No speed > 10 m/s", np.sum(speeds > 10) == 0),
            ("Trajectory coverage > 40%", len(df) > 1200),  # 3432 frames * 0.4
        ]
        
        all_pass = True
        for criterion, passed in criteria:
            status = "✅ PASS" if passed else "❌ FAIL"
            write(f"  {status} {criterion}")
            all_pass = all_pass and passed
        
        write("")
        if all_pass:
            write("🎉 ALL CRITERIA PASSED - Oracle is working correctly!")
        else:
            write("⚠️  SOME CRITERIA FAILED - Oracle needs improvement")
        write("")
        
        write("=" * 70)
    
    print(f"\nFull report saved to: {report_path}")


def plot_trajectory(df, output_dir):
    """Plot 3D trajectory from multiple viewpoints"""
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    x, y, z = df['x_m'].values, df['y_m'].values, df['z_m'].values
    
    # XY view (top-down)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, y, 'b-', linewidth=0.5, alpha=0.7)
    ax1.scatter(x[0], y[0], c='green', s=100, marker='o', label='Start', zorder=5)
    ax1.scatter(x[-1], y[-1], c='red', s=100, marker='s', label='End', zorder=5)
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_title('Top View (XY)')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.legend()
    
    # XZ view (side)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x, z, 'b-', linewidth=0.5, alpha=0.7)
    ax2.scatter(x[0], z[0], c='green', s=100, marker='o', label='Start', zorder=5)
    ax2.scatter(x[-1], z[-1], c='red', s=100, marker='s', label='End', zorder=5)
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Z [m]')
    ax2.set_title('Side View (XZ)')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    ax2.legend()
    
    # YZ view (front)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(y, z, 'b-', linewidth=0.5, alpha=0.7)
    ax3.scatter(y[0], z[0], c='green', s=100, marker='o', label='Start', zorder=5)
    ax3.scatter(y[-1], z[-1], c='red', s=100, marker='s', label='End', zorder=5)
    ax3.set_xlabel('Y [m]')
    ax3.set_ylabel('Z [m]')
    ax3.set_title('Front View (YZ)')
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    ax3.legend()
    
    # 3D view
    ax4 = fig.add_subplot(gs[1, :], projection='3d')
    ax4.plot(x, y, z, 'b-', linewidth=0.5, alpha=0.7)
    ax4.scatter(x[0], y[0], z[0], c='green', s=100, marker='o', label='Start', zorder=5)
    ax4.scatter(x[-1], y[-1], z[-1], c='red', s=100, marker='s', label='End', zorder=5)
    ax4.set_xlabel('X [m]')
    ax4.set_ylabel('Y [m]')
    ax4.set_zlabel('Z [m]')
    ax4.set_title('3D Trajectory')
    ax4.legend()
    
    plt.tight_layout()
    
    plot_path = Path(output_dir) / 'trajectory_plot.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Trajectory plot saved to: {plot_path}")
    plt.close()


def plot_speed_profile(df, speeds, output_dir):
    """Plot speed over time with violation markers"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    
    time = df['time_s'].iloc[:-1].values  # speeds is one shorter
    
    # Speed profile
    ax1.plot(time, speeds, 'b-', linewidth=0.5, alpha=0.7, label='Speed')
    ax1.axhline(5, color='orange', linestyle='--', label='5 m/s (reasonable limit)')
    ax1.axhline(10, color='red', linestyle='--', label='10 m/s (violation threshold)')
    
    # Mark violations
    violations = speeds > 10
    if np.any(violations):
        ax1.scatter(time[violations], speeds[violations], c='red', s=20, 
                   alpha=0.6, label=f'Violations ({np.sum(violations)})')
    
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Speed [m/s]')
    ax1.set_title('Speed Profile')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([0, min(np.percentile(speeds, 99), 50)])  # Cap at 99th percentile or 50 m/s
    
    # Speed histogram
    ax2.hist(speeds, bins=100, alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(speeds), color='green', linestyle='--', 
               label=f'Mean: {np.mean(speeds):.2f} m/s')
    ax2.axvline(np.median(speeds), color='blue', linestyle='--', 
               label=f'Median: {np.median(speeds):.2f} m/s')
    ax2.axvline(5, color='orange', linestyle='--', label='5 m/s limit')
    ax2.axvline(10, color='red', linestyle='--', label='10 m/s limit')
    ax2.set_xlabel('Speed [m/s]')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Speed Distribution')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim([0, min(np.percentile(speeds, 99), 50)])
    
    plt.tight_layout()
    
    plot_path = Path(output_dir) / 'speed_profile.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Speed profile saved to: {plot_path}")
    plt.close()


def plot_diagnostics(df, speeds, output_dir):
    """Plot additional diagnostic information"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    time = df['time_s'].iloc[:-1].values
    
    # Distance traveled
    ax1 = axes[0]
    ax1.plot(df['time_s'].values, df['dist_m'].values, 'b-', linewidth=1)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Distance [m]')
    ax1.set_title('Cumulative Distance Traveled')
    ax1.grid(True, alpha=0.3)
    
    # Acceleration
    ax2 = axes[1]
    if len(speeds) > 1:
        accel = compute_accelerations(speeds)
        time_accel = time[:-1]
        ax2.plot(time_accel, accel, 'g-', linewidth=0.5, alpha=0.7)
        ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Acceleration [m/s²]')
        ax2.set_title('Acceleration Profile')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([np.percentile(accel, 1), np.percentile(accel, 99)])
    
    # Turn rate
    ax3 = axes[2]
    if len(df) > 2:
        turn_rates = compute_turn_rates(df)
        time_turn = time[:-1]
        ax3.plot(time_turn, turn_rates, 'r-', linewidth=0.5, alpha=0.7)
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('Turn Rate [deg/s]')
        ax3.set_title('Turn Rate (XY plane)')
        ax3.grid(True, alpha=0.3)
        if len(turn_rates) > 0:
            ax3.set_ylim([0, np.percentile(turn_rates, 99)])
    
    plt.tight_layout()
    
    plot_path = Path(output_dir) / 'diagnostics.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Diagnostics plot saved to: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze ArUco oracle trajectory and generate diagnostic plots'
    )
    parser.add_argument('--trajectory', required=True, help='Input trajectory CSV')
    parser.add_argument('--output-dir', default='out/', help='Output directory for plots')
    parser.add_argument('--fps', type=float, default=30.0, help='Video framerate')
    
    args = parser.parse_args()
    
    # Check input exists
    traj_path = Path(args.trajectory)
    if not traj_path.exists():
        print(f"Error: Trajectory file not found: {traj_path}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading trajectory from: {traj_path}")
    df = load_trajectory(traj_path)
    
    print("Computing metrics...")
    speeds = compute_speeds(df, args.fps)
    
    print("\nGenerating report...")
    print_report(df, speeds, output_dir)
    
    print("\nGenerating plots...")
    plot_trajectory(df, output_dir)
    plot_speed_profile(df, speeds, output_dir)
    plot_diagnostics(df, speeds, output_dir)
    
    print("\n✅ Analysis complete!")
    print(f"   All outputs saved to: {output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

