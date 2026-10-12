#!/usr/bin/env python3

# Copyright 2024 Universidad Politécnica de Madrid
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the Universidad Politécnica de Madrid nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import argparse
from time import sleep
import time
import yaml
import os
import csv
import random
import heapq
import numpy as np
from datetime import datetime

import networkx as nx
from python_tsp.exact import solve_tsp_dynamic_programming

from as2_python_api.drone_interface import DroneInterface
from as2_msgs.msg import YawMode
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
from ompl import base as ob
from ompl import geometric as og

TAKE_OFF_HEIGHT = 1.0  # Height in meters
TAKE_OFF_SPEED = 1.0  # Max speed in m/s
SLEEP_TIME = 0.5  # Sleep time between behaviors in seconds
SPEED = 1.0  # Max speed in m/s
LAND_SPEED = 0.5  # Max speed in m/s

# Tuned based on scenario coordinate ranges
GRID_X = 20
GRID_Y = 20
GRID_Z = 15
GRID_RES = 0.5  # Grid cell size in meters
OBS_TOLERANCE = 1.0  # Obstacle inflation margin


# Read the YAML scenario file
def read_scenario(file_path):
    with open(file_path, 'r') as file:
        scenario = yaml.safe_load(file)
    return scenario

# TSP Solvers

def tsp_greedy(viewpoints, start):
    """Greedy nearest-neighbor TSP solver"""
    t0 = time.time()
    ids = list(viewpoints.keys())
    positions = {i: np.array([viewpoints[i]["x"], viewpoints[i]["y"], viewpoints[i]["z"]]) for i in ids}
    
    route = []
    remaining = set(ids)
    current = np.array(start)
    
    while remaining:
        nearest = min(remaining, key=lambda i: np.linalg.norm(positions[i] - current))
        route.append(nearest)
        current = positions[nearest]
        remaining.remove(nearest)
    
    return route, (time.time() - t0) * 1000


def tsp_dp(viewpoints, start):
    """Dynamic programming TSP solver using python_tsp library"""
    t0 = time.time()
    ids = list(viewpoints.keys())
    
    # Include start position as node 0
    positions = [np.array(start, dtype=float)]
    for i in ids:
        positions.append(np.array([float(viewpoints[i]["x"]), 
                                   float(viewpoints[i]["y"]), 
                                   float(viewpoints[i]["z"])]))
    
    # Build distance matrix
    n = len(positions)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = np.linalg.norm(positions[i] - positions[j])
    
    permutation, _ = solve_tsp_dynamic_programming(distance_matrix)
    
   
    k = permutation.index(0)
    permutation = permutation[k:] + permutation[:k]
    
    route = [ids[i - 1] for i in permutation if i != 0]
    return route, (time.time() - t0) * 1000


def tsp_networkx(viewpoints, start):
    """NetworkX approximation TSP solver"""
    t0 = time.time()
    ids = list(viewpoints.keys())
    G = nx.Graph()
    
    start_pos = np.array(start, dtype=float)
    G.add_node("START")
    
    for i in ids:
        pos = np.array([float(viewpoints[i]["x"]), 
                        float(viewpoints[i]["y"]), 
                        float(viewpoints[i]["z"])])
        G.add_edge("START", i, weight=float(np.linalg.norm(start_pos - pos)))
    
    for idx_a, a in enumerate(ids):
        pos_a = np.array([float(viewpoints[a]["x"]), 
                          float(viewpoints[a]["y"]), 
                          float(viewpoints[a]["z"])])
        for idx_b, b in enumerate(ids):
            if idx_a < idx_b:
                pos_b = np.array([float(viewpoints[b]["x"]), 
                                  float(viewpoints[b]["y"]), 
                                  float(viewpoints[b]["z"])])
                G.add_edge(a, b, weight=float(np.linalg.norm(pos_a - pos_b))) 
    
    tour = nx.approximation.greedy_tsp(G, source="START")
    
    tour = [x for x in tour if x != "START"]
    if tour and tour[-1] == tour[0]:
        tour = tour[:-1]
    
    return tour, (time.time() - t0) * 1000


def solve_tsp(viewpoints, start, method):
    """Solve TSP using specified method"""
    methods = {"greedy": tsp_greedy, "dp": tsp_dp, "networkx": tsp_networkx}
    return methods[method](viewpoints, start)

# Collision Detection

def check_collision(point, obstacles):
    """Check if a point collides with any obstacle or the ground"""
    if point[2] <= 0.2:  # Ground collision threshold
        return True
    
    for obs in obstacles.values():
        half_d = (obs["d"] + 2 * OBS_TOLERANCE) / 2
        half_w = (obs["w"] + 2 * OBS_TOLERANCE) / 2
        half_h = (obs["h"] + 2 * OBS_TOLERANCE) / 2
        
        if (obs["x"] - half_d <= point[0] <= obs["x"] + half_d and
            obs["y"] - half_w <= point[1] <= obs["y"] + half_w and
            obs["z"] - half_h <= point[2] <= obs["z"] + half_h):
            return True
    return False


def check_segment_free(p1, p2, obstacles):
    """Check if a straight line between two points is collision free"""
    for i in range(21):  # 21 evenly spaced collision checks
        t = i / 20
        point = [p1[j] + t * (p2[j] - p1[j]) for j in range(3)]
        if check_collision(point, obstacles):
            return False
    return True

# Path Planning

def build_grid(obstacles):
    """Build 3D occupancy grid from obstacles"""
    shape = (int(2 * GRID_X / GRID_RES), 
             int(2 * GRID_Y / GRID_RES), 
             int(GRID_Z / GRID_RES))
    grid = np.zeros(shape, dtype=np.int8)
    grid[:, :, 0] = 1  # Ground plane
    
    for obs in obstacles.values():
        ox = int((obs["x"] + GRID_X) / GRID_RES)
        oy = int((obs["y"] + GRID_Y) / GRID_RES)
        oz = int(obs["z"] / GRID_RES)
        dx = int((obs["d"] + 2 * OBS_TOLERANCE) / GRID_RES)
        dy = int((obs["w"] + 2 * OBS_TOLERANCE) / GRID_RES)
        dz = int((obs["h"] + 2 * OBS_TOLERANCE) / GRID_RES)
        
        for i in range(max(0, ox - dx // 2), min(shape[0], ox + dx // 2 + 1)):
            for j in range(max(0, oy - dy // 2), min(shape[1], oy + dy // 2 + 1)):
                for k in range(max(0, oz - dz // 2), min(shape[2], oz + dz // 2 + 1)):
                    grid[i, j, k] = 1
    return grid


def world_to_grid(point):
    """Convert world coordinates to grid cell indices"""
    return (int((point[0] + GRID_X) / GRID_RES),
            int((point[1] + GRID_Y) / GRID_RES),
            int(point[2] / GRID_RES))


def grid_to_world(cell):
    """Convert grid cell indices back to world coordinates"""
    return [cell[0] * GRID_RES - GRID_X,
            cell[1] * GRID_RES - GRID_Y,
            cell[2] * GRID_RES]


def plan_astar(start, goal, obstacles):
    """A* path planning on the 3D grid"""
    t0 = time.time()
    grid = build_grid(obstacles)
    start_cell = world_to_grid(start)
    goal_cell = world_to_grid(goal)
    
    for cell in [start_cell, goal_cell]:
        if not all(0 <= cell[i] < grid.shape[i] for i in range(3)):
            return None, (time.time() - t0) * 1000
        if grid[cell]:
            return None, (time.time() - t0) * 1000
    
    def heuristic(a, b):
        return np.sqrt(sum((a[i] - b[i])**2 for i in range(3)))
    
    # 26-connected neighbourhood
    directions = [(dx, dy, dz) for dx in [-1, 0, 1] 
                  for dy in [-1, 0, 1] for dz in [-1, 0, 1] 
                  if dx or dy or dz]
    
    open_list = [(0, start_cell)]
    came_from = {}
    g_score = {start_cell: 0}
    
    while open_list:
        _, current = heapq.heappop(open_list)
        
        if current == goal_cell:
            path = []
            while current in came_from:
                path.append(grid_to_world(current))
                current = came_from[current]
            path.append(grid_to_world(start_cell))
            path.reverse()
            path[0] = list(start)
            path[-1] = list(goal)
            return path, (time.time() - t0) * 1000
        
        for d in directions:
            neighbor = (current[0] + d[0], current[1] + d[1], current[2] + d[2])
            
            if not all(0 <= neighbor[i] < grid.shape[i] for i in range(3)):
                continue
            if grid[neighbor]:
                continue
            
            tentative_g = g_score[current] + np.sqrt(sum(x**2 for x in d))
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal_cell)  # f = g + h
                heapq.heappush(open_list, (f_score, neighbor))
    
    return None, (time.time() - t0) * 1000


def plan_dijkstra(start, goal, obstacles):
    """Dijkstra's path planning on the 3D grid"""
    t0 = time.time()
    grid = build_grid(obstacles)
    start_cell = world_to_grid(start)
    goal_cell = world_to_grid(goal)
    
    for cell in [start_cell, goal_cell]:
        if not all(0 <= cell[i] < grid.shape[i] for i in range(3)):
            return None, (time.time() - t0) * 1000
        if grid[cell]:
            return None, (time.time() - t0) * 1000
    
    directions = [(dx, dy, dz) for dx in [-1, 0, 1] 
                  for dy in [-1, 0, 1] for dz in [-1, 0, 1] 
                  if dx or dy or dz]
    
    open_list = [(0, start_cell)]
    came_from = {}
    g_score = {start_cell: 0}
    
    while open_list:
        current_cost, current = heapq.heappop(open_list)
        
        if current == goal_cell:
            path = []
            while current in came_from:
                path.append(grid_to_world(current))
                current = came_from[current]
            path.append(grid_to_world(start_cell))
            path.reverse()
            path[0] = list(start)
            path[-1] = list(goal)
            return path, (time.time() - t0) * 1000
        
        for d in directions:
            neighbor = (current[0] + d[0], current[1] + d[1], current[2] + d[2])
            
            if not all(0 <= neighbor[i] < grid.shape[i] for i in range(3)):
                continue
            if grid[neighbor]:
                continue
            
            tentative_g = g_score[current] + np.sqrt(sum(x**2 for x in d))
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                heapq.heappush(open_list, (tentative_g, neighbor))  # No heuristic expands by cost only
    
    return None, (time.time() - t0) * 1000


def plan_rrt(start, goal, obstacles, max_iter=5000, step_size=0.5):
    """RRT path planning in continuous space"""
    t0 = time.time()
    tree = [(list(start), -1)]
    
    for _ in range(max_iter):  # Max tree iterations
        if random.random() < 0.1:  # Goal bias
            sample = list(goal)
        else:
            sample = [random.uniform(-GRID_X, GRID_X),
                      random.uniform(-GRID_Y, GRID_Y),
                      random.uniform(0.5, GRID_Z)]
        
        nearest_idx = min(range(len(tree)), 
                          key=lambda i: np.linalg.norm(np.array(tree[i][0]) - np.array(sample)))
        nearest = tree[nearest_idx][0]
        
        direction = np.array(sample) - np.array(nearest)
        distance = np.linalg.norm(direction)
        if distance > step_size:
            direction = direction / distance * step_size
        
        new_point = [nearest[i] + direction[i] for i in range(3)]
        
        if check_collision(new_point, obstacles):
            continue
        if not check_segment_free(nearest, new_point, obstacles):
            continue
        
        tree.append((new_point, nearest_idx))
        
        if np.linalg.norm(np.array(new_point) - np.array(goal)) < step_size:
            if check_segment_free(new_point, goal, obstacles):
                tree.append((list(goal), len(tree) - 1))
                
                path = []
                idx = len(tree) - 1
                while idx != -1:
                    path.append(tree[idx][0])
                    idx = tree[idx][1]
                path.reverse()
                return path, (time.time() - t0) * 1000
    
    return None, (time.time() - t0) * 1000


def plan_ompl(start, goal, obstacles):
    """OMPL RRT* path planning"""
    t0 = time.time()
    
    space = ob.RealVectorStateSpace(3)
    bounds = ob.RealVectorBounds(3)
    bounds.setLow(0, -GRID_X)
    bounds.setHigh(0, GRID_X)
    bounds.setLow(1, -GRID_Y)
    bounds.setHigh(1, GRID_Y)
    bounds.setLow(2, 0.3)  # Min flight height
    bounds.setHigh(2, GRID_Z)
    space.setBounds(bounds)
    
    si = ob.SpaceInformation(space)
    si.setStateValidityChecker(
        ob.StateValidityCheckerFn(lambda st: not check_collision([st[0], st[1], st[2]], obstacles)))
    si.setup()
    
    start_state = ob.State(space)
    goal_state = ob.State(space)
    start_state[0], start_state[1], start_state[2] = start[0], start[1], start[2]
    goal_state[0], goal_state[1], goal_state[2] = goal[0], goal[1], goal[2]
    
    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start_state, goal_state)
    
    planner = og.RRTstar(si)
    planner.setProblemDefinition(pdef)
    planner.setup()
    
    if planner.solve(10.0):  # Planning timeout per segment
        solution = pdef.getSolutionPath()
        path = [[solution.getState(i)[0], 
                 solution.getState(i)[1], 
                 solution.getState(i)[2]] for i in range(solution.getStateCount())]
        return path, (time.time() - t0) * 1000
    
    return None, (time.time() - t0) * 1000


def plan_path(start, goal, obstacles, method):
    """Plan a path using the specified method"""
    planners = {"astar": plan_astar, "dijkstra": plan_dijkstra, 
                "rrt": plan_rrt, "ompl": plan_ompl}
    return planners[method](start, goal, obstacles)

# Path Optimization

def optimize_path(path, obstacles):
    """Remove unnecessary waypoints while keeping the path collision free"""
    if len(path) < 3:
        return path

    optimized_path = [path[0]]

    # Greedy line-of-sight simplification
    i = 0
    while i < len(path) - 1:
        j = i + 2
        while j < len(path) and check_segment_free(path[i], path[j], obstacles):
            j += 1
        optimized_path.append(path[j - 1])
        i = j - 1

    return optimized_path


def calculate_path_length(path):
    """Calculate total length of a path"""
    if not path or len(path) < 2:
        return 0.0
    return sum(np.linalg.norm(np.array(path[i + 1]) - np.array(path[i])) 
               for i in range(len(path) - 1))

# Drone Interface with Camera

class DroneWithCamera(DroneInterface, Node):
    """DroneInterface extended with camera and ArUco detection"""
    
    def __init__(self, drone_id="drone0", use_sim_time=True, verbose=False):
        super().__init__(drone_id=drone_id, use_sim_time=use_sim_time, 
                         verbose=verbose, spin_rate=20.0)
        self.camera_image = None
        self.bridge = CvBridge()
        self.create_subscription(Image, "sensor_measurements/hd_camera/image_raw", 
                                 self.camera_callback, 10)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
    
    def camera_callback(self, msg):
        """Callback to update the current camera frame"""
        self.camera_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
    
    def detect_aruco(self, timeout=3.0):
        """Detect ArUco markers for up to timeout seconds"""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.camera_image is None:
                sleep(0.1)
                continue
            gray = cv2.cvtColor(self.camera_image, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
            if ids is not None and len(ids) > 0:
                return True, ids.flatten().tolist()
            sleep(0.1)
        return False, []


def convert_waypoints_to_path_msg(waypoints):
    """Convert a list of [x,y,z] waypoints into a ROS2 Path message"""
    path_msg = Path()
    path_msg.header.frame_id = "map"
    path_msg.header.stamp = rclpy.clock.Clock().now().to_msg()

    for waypoint in waypoints:
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = path_msg.header.stamp
        pose.pose.position.x = float(waypoint[0])
        pose.pose.position.y = float(waypoint[1])
        pose.pose.position.z = float(waypoint[2])
        pose.pose.orientation.w = 1.0
        path_msg.poses.append(pose)

    return path_msg

# Results Logging

def save_results_csv(results):
    """Add results to the comparison CSV file"""
    os.makedirs('./results', exist_ok=True)
    filepath = './results/comparison_results.csv'
    file_exists = os.path.exists(filepath)
    
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['time', 'scenario', 'tsp', 'path', 'opt', 'visited', 
                            'total', 'success%', 'dist_m', 'overhead%', 'time_s', 
                            'tsp_ms', 'plan_ms', 'wpts'])
        writer.writerow([
            datetime.now().strftime("%H:%M"),
            results['scenario'], results['tsp'], results['path'], results['opt'],
            results['visited'], results['total'], f"{results['success_pct']:.1f}",
            f"{results['distance']:.2f}", f"{results['overhead']:.1f}",
            f"{results['time']:.2f}", f"{results['tsp_ms']:.1f}",
            f"{results.get('plan_ms', 0):.1f}", results['waypoints']
        ])
    print(f'Results saved to {filepath}')


def print_results(results):
    """Print a summary of the mission results"""
    print(f"\nRESULTS: {results['scenario']} | TSP={results['tsp']} "
          f"Path={results['path']}")
    print(f"  Success:   {results['visited']}/{results['total']} ({results['success_pct']:.1f}%)")
    print(f"  Distance:  {results['distance']:.2f}m (overhead: {results['overhead']:.1f}%)")
    print(f"  Time:      {results['time']:.2f}s | TSP: {results['tsp_ms']:.1f}ms | "
          f"Plan: {results.get('plan_ms', 0):.1f}ms")
    print(f"  Waypoints: {results['waypoints']}")
    print(f"  Visited:   {results.get('visited_ids', [])}")

# Mission Functions

def drone_start(drone_interface: DroneInterface) -> bool:
    """
    Take off the drone.

    :param drone_interface: DroneInterface object
    :return: Bool indicating if the take off was successful
    """
    print('Start mission')

    # Arm
    print('Arm')
    success = drone_interface.arm()
    print(f'Arm success: {success}')

    # Offboard
    print('Offboard')
    success = drone_interface.offboard()
    print(f'Offboard success: {success}')

    # Take Off
    print('Take Off')
    success = drone_interface.takeoff(height=TAKE_OFF_HEIGHT, speed=TAKE_OFF_SPEED)
    print(f'Take Off success: {success}')

    return success


def drone_run(drone_interface: DroneInterface, scenario: dict, 
              tsp_method: str, path_method: str) -> dict:
    """
    Run the mission for a single drone.

    :param drone_interface: DroneInterface object
    :param scenario: Scenario dictionary from YAML
    :param tsp_method: TSP solving method
    :param path_method: Path planning method
    :return: Results dictionary
    """
    print('Run mission')
    
    viewpoints = scenario.get("viewpoint_poses", {})
    obstacles = scenario.get("obstacles", {})
    num_viewpoints = len(viewpoints)
    
    print(f'\n[CFG] TSP:{tsp_method} Path:{path_method}')
    print(f'[SC] {num_viewpoints} viewpoints, {len(obstacles)} obstacles')
    
    # Get start position
    start_pose = scenario.get("drone_start_pose", {"x": 0, "y": 0})
    start = [float(start_pose.get("x", 0)), float(start_pose.get("y", 0)), TAKE_OFF_HEIGHT]
    print(f'[START] {start}')
    
    # Solve TSP for optimal visiting order
    order, tsp_time = solve_tsp(viewpoints, start, tsp_method)
    print(f'[TSP] {tsp_time:.1f}ms: {order}')
    
    # Baseline for overhead calc
    straight_distance = 0.0
    prev_pos = np.array(start)
    for vid in order:
        vp_pos = np.array([viewpoints[vid]["x"], viewpoints[vid]["y"], viewpoints[vid]["z"]])
        straight_distance += np.linalg.norm(vp_pos - prev_pos)
        prev_pos = vp_pos
    
    total_distance = 0.0
    total_waypoints = 0
    total_plan_time = 0.0
    visited_ids = set()
    current_pos = start
    all_trajectory_points = [start]
    mission_start = time.time()
    
    # Go to each viewpoint
    for i, vid in enumerate(order):
        vp = viewpoints[vid]
        goal = [vp["x"], vp["y"], vp["z"]]
        yaw = float(vp.get("w", 0))
        
        print(f'Planning path from {current_pos} to {goal}')
        
        # Plan path
        raw_path, plan_time = plan_path(current_pos, goal, obstacles, path_method)
        total_plan_time += plan_time
        
        if raw_path is None:
            print(f'[SKIP] {vid} - no valid path found')
            continue
        
        # Optimise the path
        final_path = optimize_path(raw_path, obstacles)
            
        path_distance = calculate_path_length(final_path)
        total_distance += path_distance
        total_waypoints += len(final_path)
        all_trajectory_points.extend(final_path[1:])
        print(f'[PATH] {i + 1}/{num_viewpoints} -> {vid}: {len(final_path)} wpts, {path_distance:.1f}m')
        
        # Follow path
        path_msg = convert_waypoints_to_path_msg(final_path)
        drone_interface.follow_path(path_msg, speed=SPEED, 
                                    yaw_mode=YawMode.FIXED_YAW, yaw_angle=yaw)
        sleep(SLEEP_TIME)
        
        # Detect ArUco marker
        success, markers = drone_interface.detect_aruco(3.0)
        if success:
            visited_ids.add(vid)
            print(f'[OK] {vid}: marker {markers[0]}')
        else:
            print(f'[MISS] {vid}')
        
        current_pos = goal
        sleep(SLEEP_TIME)
    
    mission_time = time.time() - mission_start
    num_visited = len(visited_ids)
    
    print("---------------------------------")
    print(f"Viewpoint visited ({num_visited}/{num_viewpoints})")
    print(f"Total Flight Distance: {total_distance:.2f} meters")
    print("---------------------------------")
    
    # Save trajectory to file
    os.makedirs('./results', exist_ok=True)
    scenario_name = args.scenario.split('/')[-1].replace('.yaml', '')
    traj_file = f'./results/trajectory_{scenario_name}_{tsp_method}_{path_method}.csv'
    with open(traj_file, 'w') as f:
        for pt in all_trajectory_points:
            f.write(f'{pt[0]},{pt[1]},{pt[2]}\n')
    print(f'Trajectory saved to {traj_file}')

    return {
        'scenario': args.scenario.split('/')[-1].replace('.yaml', ''),
        'tsp': tsp_method,
        'path': path_method,
        'opt': 'on',
        'visited': num_visited,
        'total': num_viewpoints,
        'success_pct': num_visited / num_viewpoints * 100 if num_viewpoints else 0,
        'distance': total_distance,
        'overhead': (total_distance - straight_distance) / straight_distance * 100 if straight_distance else 0,
        'time': mission_time,
        'tsp_ms': tsp_time,
        'plan_ms': total_plan_time,
        'waypoints': total_waypoints,
        'visited_ids': sorted(list(visited_ids))
    }


def drone_end(drone_interface: DroneInterface) -> bool:
    """
    End the mission for a single drone.

    :param drone_interface: DroneInterface object
    :return: Bool indicating if the land was successful
    """
    print('End mission')

    # Land
    print('Land')
    success = drone_interface.land(speed=LAND_SPEED)
    print(f'Land success: {success}')
    if not success:
        return success

    # Manual
    print('Manual')
    success = drone_interface.manual()
    print(f'Manual success: {success}')

    return success


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Single drone mission with TSP and path planning comparison')

    parser.add_argument('scenario', type=str, help="scenario file to attempt to execute")
    parser.add_argument('-n', '--namespace',
                        type=str,
                        default='drone0',
                        help='ID of the drone to be used in the mission')
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-s', '--use_sim_time',
                        action='store_true',
                        default=True,
                        help='Use simulation time')
    parser.add_argument('--tsp',
                        choices=['greedy', 'dp', 'networkx'],
                        default='greedy',
                        help='TSP solving method')
    parser.add_argument('--path',
                        choices=['astar', 'dijkstra', 'rrt', 'ompl'],
                        default='astar',
                        help='Path planning method')

    args = parser.parse_args()
    drone_namespace = args.namespace
    verbosity = args.verbose
    use_sim_time = args.use_sim_time

    print(f'Running mission for drone {drone_namespace}')

    print(f"Reading scenario {args.scenario}")
    scenario = read_scenario(args.scenario)

    rclpy.init()

    uav = DroneWithCamera(
        drone_id=drone_namespace,
        use_sim_time=use_sim_time,
        verbose=verbosity)

    success = drone_start(uav)
    try:
        start_time = time.time()
        if success:
            results = drone_run(uav, scenario, args.tsp, args.path)
            print_results(results)
            save_results_csv(results)
        duration = time.time() - start_time
        print("---------------------------------")
        print(f"Tour of {args.scenario} took {duration} seconds")
        print("---------------------------------")
    except KeyboardInterrupt as e:
        pass
    # sleep(5.0)
    success = drone_end(uav)

    uav.shutdown()
    rclpy.shutdown()
    print('Clean exit')
    exit(0)