#!/usr/bin/env python3

"""
Trajectory visualisation for structural inspection mission.
Combined 2D and 3D views showing TSP route vs actual path planner route.
"""

import argparse
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as patches

# Plot parameters
MARKER_SIZE_2D_SMALL = 40
MARKER_SIZE_2D_LARGE = 55
MARKER_SIZE_3D_SMALL = 25
MARKER_SIZE_3D_LARGE = 40
LABEL_FONT_SMALL = 6
LABEL_FONT_LARGE = 7
LINE_WIDTH_SMALL = 1.0
LINE_WIDTH_LARGE = 1.5
FIGURE_SIZE = (16, 7)
DPI = 200


# File I/O

def read_scenario(file_path):
    """Load scenario from YAML file"""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def load_trajectory(file_path):
    """Load trajectory CSV, returns Nx3 numpy array"""
    return np.loadtxt(file_path, delimiter=',')


# Data Processing 

def extract_visit_order(trajectory, viewpoints, threshold=1.5):  # Accounts for position drift
    """Order viewpoints were visited by matching trajectory
    points to viewpoint positions within a distance threshold"""
    vp_positions = {}
    for vid, vp in viewpoints.items():
        vp_positions[vid] = np.array([vp['x'], vp['y'], vp['z']])
    
    visit_order = []
    visited = set()
    
    for point in trajectory:
        point = np.array(point)
        for vid, vp_pos in vp_positions.items():
            if vid not in visited:
                dist = np.linalg.norm(point - vp_pos)
                if dist < threshold:
                    visit_order.append(vid)
                    visited.add(vid)
                    break
    
    return visit_order


def create_cuboid_faces(center, size):
    """Create faces for a 3D cuboid obstacle for plotting"""
    x, y, z = center
    d, w, h = size
    dx, dy = d / 2, w / 2
    
    vertices = [
        [x - dx, y - dy, 0], [x + dx, y - dy, 0],
        [x + dx, y + dy, 0], [x - dx, y + dy, 0],
        [x - dx, y - dy, h], [x + dx, y - dy, h],
        [x + dx, y + dy, h], [x - dx, y + dy, h],
    ]
    
    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
    ]
    return faces


# Plotting

def plot_2d(ax, viewpoints, obstacles, start_pos, trajectory, visit_order, params):
    """Plot 2D top-down view of the mission"""
    start_x, start_y, start_z = start_pos
    
    # Obstacles
    for obs in obstacles.values():
        ox, oy = float(obs["x"]), float(obs["y"])
        od, ow = float(obs["d"]), float(obs["w"])
        rect = patches.Rectangle(
            (ox - od / 2, oy - ow / 2), od, ow,
            linewidth=1, edgecolor='dimgray',
            facecolor='gray', alpha=0.4
        )
        ax.add_patch(rect)
    
    # TSP route
    tsp_x = [start_x]
    tsp_y = [start_y]
    for vid in visit_order:
        vp = viewpoints[vid]
        tsp_x.append(float(vp['x']))
        tsp_y.append(float(vp['y']))
    ax.plot(tsp_x, tsp_y, 'k--', linewidth=params['line_width'], 
            label='TSP Route', zorder=2)
    
    # Actual path
    ax.plot(trajectory[:, 0], trajectory[:, 1],
            color='green', linewidth=params['line_width'] + 0.5,
            label='A* Path', zorder=3)
    
    # Viewpoints
    vp_x = [float(vp["x"]) for vp in viewpoints.values()]
    vp_y = [float(vp["y"]) for vp in viewpoints.values()]
    ax.scatter(vp_x, vp_y, c='red', s=params['marker_2d'], marker='o',
               label='Markers', zorder=5, edgecolors='darkred', linewidths=0.8,
               alpha=1.0)
    
    # Viewpoint labels
    for vid, vp in viewpoints.items():
        ax.annotate(str(vid), (float(vp["x"]), float(vp["y"])),
                    textcoords="offset points", xytext=(4, 4),
                    fontsize=params['label_font'], fontweight='bold', color='#2c3e50')
    
    # Start position
    ax.scatter([start_x], [start_y], c='blue', s=70, marker='s',
               label='Start', zorder=5, edgecolors='darkblue', linewidths=1)
    
    ax.set_xlabel('X (meters)', fontweight='bold')
    ax.set_ylabel('Y (meters)', fontweight='bold')
    ax.set_title('Drone Mission Path (2D)', fontweight='bold', fontsize=11)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)


def plot_3d(ax, viewpoints, obstacles, start_pos, trajectory, visit_order, params):
    """Plot 3D view of the mission"""
    start_x, start_y, start_z = start_pos
    
    # Obstacles as 3D cuboids
    for obs in obstacles.values():
        center = (float(obs["x"]), float(obs["y"]), float(obs["z"]))
        size = (float(obs["d"]), float(obs["w"]), float(obs["h"]))
        faces = create_cuboid_faces(center, size)
        poly = Poly3DCollection(faces, alpha=0.2, facecolor='gray',
                                edgecolor='dimgray', linewidth=0.5)
        ax.add_collection3d(poly)
    
    # TSP route
    tsp_x = [start_x]
    tsp_y = [start_y]
    tsp_z = [start_z]
    for vid in visit_order:
        vp = viewpoints[vid]
        tsp_x.append(float(vp['x']))
        tsp_y.append(float(vp['y']))
        tsp_z.append(float(vp['z']))
    ax.plot(tsp_x, tsp_y, tsp_z, 'k--', linewidth=params['line_width'],
            label='TSP Route', zorder=2)
    
    # Actual path
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
            color='green', linewidth=params['line_width'] + 0.5,
            label='A* Path', zorder=4)
    
    # Viewpoints
    vp_x = [float(vp["x"]) for vp in viewpoints.values()]
    vp_y = [float(vp["y"]) for vp in viewpoints.values()]
    vp_z = [float(vp["z"]) for vp in viewpoints.values()]
    ax.scatter(vp_x, vp_y, vp_z, c='red', s=params['marker_3d'], marker='o',
               label='Markers', zorder=5, edgecolors='darkred', linewidths=0.8,
               depthshade=False)
    
    # Viewpoint labels
    for vid, vp in viewpoints.items():
        ax.text(float(vp["x"]) + 0.3, float(vp["y"]) + 0.3, float(vp["z"]) + 0.3,
                str(vid), fontsize=params['label_font'], fontweight='bold',
                color='#2c3e50')
    
    # Start position
    ax.scatter([start_x], [start_y], [start_z], c='blue', s=60, marker='s',
               label='Start', zorder=5, edgecolors='darkblue', linewidths=1,
               depthshade=False)
    
    ax.set_xlabel('X (meters)', fontweight='bold', labelpad=8)
    ax.set_ylabel('Y (meters)', fontweight='bold', labelpad=8)
    ax.set_zlabel('Z (meters)', fontweight='bold', labelpad=8)
    ax.set_title('Drone Mission Path (3D)', fontweight='bold', fontsize=11)
    ax.legend(loc='upper left', fontsize=8)
    ax.view_init(elev=20, azim=45)


def plot_combined(scenario, trajectory_file, output_file, title=None):
    """Create combined 2D + 3D trajectory plot for a given scenario"""
    viewpoints = scenario.get("viewpoint_poses", {})
    obstacles = scenario.get("obstacles", {})
    start_pose = scenario.get("drone_start_pose", {"x": 0, "y": 0})
    
    start_x = float(start_pose.get("x", 0))
    start_y = float(start_pose.get("y", 0))
    start_z = 1.0
    start_pos = (start_x, start_y, start_z)
    
    num_viewpoints = len(viewpoints)
    num_obstacles = len(obstacles)
    
    # Avoid clutter on S3
    if num_viewpoints > 15:
        params = {
            'marker_2d': MARKER_SIZE_2D_SMALL,
            'marker_3d': MARKER_SIZE_3D_SMALL,
            'label_font': LABEL_FONT_SMALL,
            'line_width': LINE_WIDTH_SMALL
        }
    else:
        params = {
            'marker_2d': MARKER_SIZE_2D_LARGE,
            'marker_3d': MARKER_SIZE_3D_LARGE,
            'label_font': LABEL_FONT_LARGE,
            'line_width': LINE_WIDTH_LARGE
        }
    
    trajectory = load_trajectory(trajectory_file)
    visit_order = extract_visit_order(trajectory, viewpoints)
    print(f"Loaded trajectory: {len(trajectory)} waypoints")
    print(f"Visit order: {visit_order}")
    
    # Create figure with 2D and 3D subplots
    fig = plt.figure(figsize=FIGURE_SIZE)
    ax_2d = fig.add_subplot(121)
    ax_3d = fig.add_subplot(122, projection='3d')
    
    plot_2d(ax_2d, viewpoints, obstacles, start_pos, trajectory, visit_order, params)
    plot_3d(ax_3d, viewpoints, obstacles, start_pos, trajectory, visit_order, params)
    
    # Axis limits
    vp_x = [float(vp["x"]) for vp in viewpoints.values()]
    vp_y = [float(vp["y"]) for vp in viewpoints.values()]
    vp_z = [float(vp["z"]) for vp in viewpoints.values()]
    all_x = vp_x + [start_x]
    all_y = vp_y + [start_y]
    all_z = vp_z + [start_z]
    margin = 2
    
    ax_2d.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax_2d.set_ylim(min(all_y) - margin, max(all_y) + margin)
    
    ax_3d.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax_3d.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax_3d.set_zlim(0, max(all_z) + margin)
    
    if title:
        fig.suptitle(title, fontweight='bold', fontsize=13, y=0.98)
    else:
        fig.suptitle(f'Scenario - {num_viewpoints} viewpoints, {num_obstacles} obstacles',
                     fontweight='bold', fontsize=13, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_file}")
    
    return fig


# Main 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot combined 2D + 3D trajectory visualization')
    
    parser.add_argument('scenario', type=str,
                        help='Scenario YAML file')
    parser.add_argument('-t', '--trajectory', type=str, required=True,
                        help='Trajectory CSV file')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output PNG filename')
    parser.add_argument('--title', type=str, default=None,
                        help='Custom plot title')
    
    args = parser.parse_args()
    
    print(f"Loading scenario: {args.scenario}")
    scenario = read_scenario(args.scenario)
    
    vp_count = len(scenario.get("viewpoint_poses", {}))
    obs_count = len(scenario.get("obstacles", {}))
    print(f"Viewpoints: {vp_count}, Obstacles: {obs_count}")
    
    fig = plot_combined(
        scenario,
        trajectory_file=args.trajectory,
        output_file=args.output,
        title=args.title
    )
    
    print("Done")