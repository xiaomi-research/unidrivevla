
import json
import numpy as np
import re
import argparse
from tqdm import tqdm
import math

class StreamingStats:
    """
    Computes streaming statistics (mean, variance, std, min, max).
    Uses Welford's online algorithm for numerical stability.
    """
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min = float('inf')
        self.max = float('-inf')

    def update(self, value):
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2
        if value < self.min:
            self.min = value
        if value > self.max:
            self.max = value

    @property
    def variance(self):
        if self.count < 2:
            return 0.0
        # Use count - 1 for sample variance, but for a large dataset, count is fine
        return self.M2 / self.count

    @property
    def std(self):
        return math.sqrt(self.variance)

def parse_trajectory(traj_str):
    """
    Parses trajectory string like "[PT, (x1,y1), (x2,y2), ...]" into a list of tuples.
    """
    matches = re.findall(r'\((-?[\d\.]+),\s*(-?[\d\.]+)\)', traj_str)
    points = [(float(x), float(y)) for x, y in matches]
    return points

def calculate_normalization_stats(jsonl_path):
    stats_x = StreamingStats()
    stats_y = StreamingStats()

    print(f"Reading and analyzing {jsonl_path}...")
    with open(jsonl_path, 'r') as f:
        for line in tqdm(f):
            try:
                data = json.loads(line)
                conversations = data.get('conversations', [])

                gpt_response = None
                for turn in conversations:
                    if turn.get('from') == 'gpt':
                        gpt_response = turn.get('value')
                        break

                if gpt_response:
                    match = re.search(r'\[PT, .*?\]', gpt_response)
                    if match:
                        traj_str = match.group(0)
                        residuals = parse_trajectory(traj_str)
                        
                        for x, y in residuals:
                            stats_x.update(x)
                            stats_y.update(y)
            except (json.JSONDecodeError, ValueError) as e:
                # Ignore lines with parsing errors
                continue

    if stats_x.count == 0:
        print("No valid trajectories found.")
        return

    print("\n--- Trajectory Residual Statistics ---")
    print(f"Total points processed: {stats_x.count}")
    
    print("\nX-Residuals:")
    print(f"  Mean: {stats_x.mean:.4f}")
    print(f"  Std Dev: {stats_x.std:.4f}")
    print(f"  Min: {stats_x.min:.4f}")
    print(f"  Max: {stats_x.max:.4f}")

    print("\nY-Residuals:")
    print(f"  Mean: {stats_y.mean:.4f}")
    print(f"  Std Dev: {stats_y.std:.4f}")
    print(f"  Min: {stats_y.min:.4f}")
    print(f"  Max: {stats_y.max:.4f}")

    # --- Analysis for normalization choice ---
    print("\n--- Normalization Analysis ---")
    x_std_from_mean_max = (stats_x.max - stats_x.mean) / stats_x.std if stats_x.std > 0 else 0
    x_std_from_mean_min = (stats_x.mean - stats_x.min) / stats_x.std if stats_x.std > 0 else 0
    y_std_from_mean_max = (stats_y.max - stats_y.mean) / stats_y.std if stats_y.std > 0 else 0
    y_std_from_mean_min = (stats_y.mean - stats_y.min) / stats_y.std if stats_y.std > 0 else 0

    print("\nOutlier Check (distance from mean in std devs):")
    print(f"  X-Max: {x_std_from_mean_max:.2f} stds")
    print(f"  X-Min: {x_std_from_mean_min:.2f} stds")
    print(f"  Y-Max: {y_std_from_mean_max:.2f} stds")
    print(f"  Y-Min: {y_std_from_mean_min:.2f} stds")

    is_outlier_heavy = x_std_from_mean_max > 6 or x_std_from_mean_min > 6 or y_std_from_mean_max > 6 or y_std_from_mean_min > 6
    
    print("\nRecommendation:")
    if is_outlier_heavy:
        print("The data appears to have significant outliers (min/max values are more than 6 standard deviations from the mean).")
        print("Z-Score Normalization (Standardization) is STRONGLY RECOMMENDED as it is more robust to such outliers.")
        print("Min-Max normalization would be heavily skewed by these extreme values and is not advised.")
    else:
        print("The data does not show extreme outliers.")
        print("Both Z-Score and Min-Max normalization could be suitable. However, Z-Score is generally a safer and more common default for this type of data as it centers the distribution and is less sensitive to unseen outliers in test/validation sets.")

    print("\nFinal stats to use in your code:")
    print(f"  Mean (x, y): ({stats_x.mean:.4f}, {stats_y.mean:.4f})")
    print(f"  Std  (x, y): ({stats_x.std:.4f}, {stats_y.std:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and analyze residual trajectory normalization stats from a JSONL file.")
    parser.add_argument("jsonl_path", help="Path to the QA JSONL file")
    args = parser.parse_args()

    calculate_normalization_stats(args.jsonl_path)

