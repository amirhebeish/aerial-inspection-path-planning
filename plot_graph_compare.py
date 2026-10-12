#!/usr/bin/env python3

"""
Performance comparison plots for structural inspection mission.
Generates 5 figures comparing TSP methods and path planners.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Plot parameters
FIGURE_SIZE = (10, 6)
DPI = 200
CSV_PATH = './results/comparison_results.csv'

# matplotlib settings
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Colours for TSP methods
TSP_COLORS = {'greedy': 'green', 'dp': 'red', 'networkx': 'royalblue'}
TSP_LABELS = {'greedy': 'Greedy', 'dp': 'Dynamic Programming', 'networkx': 'NetworkX'}

# Colours for path planners
PLANNER_COLORS = {'astar': 'green', 'dijkstra': 'red', 'rrt': 'royalblue', 'ompl': 'purple'}
PLANNER_LABELS = {'astar': 'A*', 'dijkstra': 'Dijkstra', 'rrt': 'RRT', 'ompl': 'OMPL'}

# Scenarios
VP_COUNTS = {'scenario1': 10, 'scenario2': 10, 'scenario3': 20, 'scenario4': 5}
OBS_COUNTS = {'scenario1': 1, 'scenario2': 5, 'scenario3': 1, 'scenario4': 10}


#  Helper Functions 

def load_data():
    """Load results CSV file."""
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH}")
    return df

#  TSP Comparison Plots (Figures 1-2) 

def fig1_tsp_computation_time(df):
    """Fig 1: TSP computation time vs scenario (using A*)"""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Fix planner to isolate TSP effect
    df_astar = df[df['path'] == 'astar']
    
    scenarios = sorted(df_astar['scenario'].unique())
    tsp_methods = ['greedy', 'dp', 'networkx']
    
    x = np.arange(len(scenarios))
    width = 0.25
    
    for i, tsp in enumerate(tsp_methods):
        times = []
        for sc in scenarios:
            subset = df_astar[(df_astar['scenario'] == sc) & (df_astar['tsp'] == tsp)]
            times.append(subset['tsp_ms'].mean())
        
        offset = (i - 1) * width
        bars = ax.bar(x + offset, times, width, label=TSP_LABELS[tsp], 
                      color=TSP_COLORS[tsp], edgecolor='white', linewidth=0.5)
        
    
    ax.set_ylabel('Computation Time (ms)', fontweight='bold')
    ax.set_xlabel('Scenario', fontweight='bold')
    ax.set_title('TSP Method Computation Time (A* path planner)', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{s[-1]}\n({VP_COUNTS[s]} vp)' for s in scenarios])
    ax.legend(loc='upper left')
    ax.set_yscale('log')  # DP explodes at S3
    ax.set_ylim(0.05, 500000)
    
    plt.tight_layout()
    plt.savefig('fig1_tsp_time.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved fig1_tsp_time.png")


def fig2_tsp_path_quality(df):
    """Fig 2: Path distance by TSP method across scenarios (using A*)"""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    df_astar = df[df['path'] == 'astar']
    
    scenarios = sorted(df_astar['scenario'].unique())
    tsp_methods = ['greedy', 'dp', 'networkx']
    
    x = np.arange(len(scenarios))
    width = 0.25
    
    for i, tsp in enumerate(tsp_methods):
        distances = []
        for sc in scenarios:
            subset = df_astar[(df_astar['scenario'] == sc) & (df_astar['tsp'] == tsp)]
            distances.append(subset['dist_m'].mean())
        
        offset = (i - 1) * width
        bars = ax.bar(x + offset, distances, width, label=TSP_LABELS[tsp],
                      color=TSP_COLORS[tsp], edgecolor='white', linewidth=0.5)
        
    
    ax.set_ylabel('Total Flight Distance (m)', fontweight='bold')
    ax.set_xlabel('Scenario', fontweight='bold')
    ax.set_title('Path Quality by TSP Method (A* path planner)', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{s[-1]}\n({VP_COUNTS[s]} vp)' for s in scenarios])
    ax.legend(loc='upper right')
    ax.set_ylim(0, None)
    
    plt.tight_layout()
    plt.savefig('fig2_tsp_distance.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved fig2_tsp_distance.png")


# Path Planner Comparison Plots (Figures 3-5) 

def fig3_planner_computation_time(df):
    """Fig 3: Path planner computation time across scenarios (using Greedy TSP)"""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Fix TSP to isolate planner effect
    df_greedy = df[df['tsp'] == 'greedy']
    
    scenarios = sorted(df_greedy['scenario'].unique())
    planners = ['astar', 'dijkstra', 'rrt', 'ompl']
    
    x = np.arange(len(scenarios))
    width = 0.2
    
    for i, planner in enumerate(planners):
        times = []
        for sc in scenarios:
            subset = df_greedy[(df_greedy['scenario'] == sc) & (df_greedy['path'] == planner)]
            times.append(subset['plan_ms'].mean())
        
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, times, width, label=PLANNER_LABELS[planner],
                      color=PLANNER_COLORS[planner], edgecolor='white', linewidth=0.5)
        
    
    ax.set_ylabel('Planning Time (ms)', fontweight='bold')
    ax.set_xlabel('Scenario', fontweight='bold')
    ax.set_title('Path Planner Computation Time (Greedy TSP)', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{s[-1]}\n({OBS_COUNTS[s]} obs)' for s in scenarios])
    ax.legend(loc='upper right')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('fig3_planner_time.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved fig3_planner_time.png")


def fig4_planner_overhead(df):
    """Fig 4: Distance overhead by path planner (using Greedy TSP)"""
    fig, ax = plt.subplots(figsize=(9, 6))
    
    df_greedy = df[df['tsp'] == 'greedy']
    
    planners = ['astar', 'dijkstra', 'rrt', 'ompl']
    colors = [PLANNER_COLORS[p] for p in planners]
    labels = [PLANNER_LABELS[p] for p in planners]
    
    avg_overhead = []
    max_overhead = []
    
    for planner in planners:
        subset = df_greedy[df_greedy['path'] == planner]
        avg_overhead.append(subset['overhead%'].mean())
        max_overhead.append(subset['overhead%'].max())
    
    x = np.arange(len(planners))
    width = 0.35
    
    bars1 = ax.bar(x - width / 2, avg_overhead, width, label='Average', color=colors,
                   edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width / 2, max_overhead, width, label='Maximum', color=colors,
                   alpha=0.5, edgecolor='white', linewidth=0.5, hatch='//')
    
    ax.set_ylabel('Distance Overhead (%)', fontweight='bold')
    ax.set_xlabel('Path Planner', fontweight='bold')
    ax.set_title('Distance Overhead by Path Planner (Greedy TSP)', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper left')
    ax.set_ylim(0, max(max_overhead) * 1.15)
    
    plt.tight_layout()
    plt.savefig('fig4_planner_overhead.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved fig4_planner_overhead.png")


def fig5_mission_time(df):
    """Fig 5: Total mission time by path planner across scenarios (using Greedy TSP)"""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    df_greedy = df[df['tsp'] == 'greedy']
    
    scenarios = sorted(df_greedy['scenario'].unique())
    planners = ['astar', 'dijkstra', 'rrt', 'ompl']
    
    x = np.arange(len(scenarios))
    width = 0.2
    
    for i, planner in enumerate(planners):
        times = []
        for sc in scenarios:
            subset = df_greedy[(df_greedy['scenario'] == sc) & (df_greedy['path'] == planner)]
            times.append(subset['time_s'].mean())
        
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, times, width, label=PLANNER_LABELS[planner],
                      color=PLANNER_COLORS[planner], edgecolor='white', linewidth=0.5)
        
    
    ax.set_ylabel('Total Mission Time (seconds)', fontweight='bold')
    ax.set_xlabel('Scenario', fontweight='bold')
    ax.set_title('Total Mission Time (Greedy TSP)', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{s[-1]}\n({VP_COUNTS[s]} vp, {OBS_COUNTS[s]} obs)' for s in scenarios])
    ax.legend(loc='upper right')
    ax.set_ylim(0, None)
    
    plt.tight_layout()
    plt.savefig('fig5_mission_time.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved fig5_mission_time.png")


# Main 

if __name__ == '__main__':
    print("Generating performance comparison plots...")
    
    df = load_data()
    print(f"Scenarios: {sorted(df['scenario'].unique().tolist())}")
    
    # TSP comparison
    fig1_tsp_computation_time(df)
    fig2_tsp_path_quality(df)
    
    # Path planner comparison
    fig3_planner_computation_time(df)
    fig4_planner_overhead(df)
    fig5_mission_time(df)
    
    print("Done, 5 figures saved.")