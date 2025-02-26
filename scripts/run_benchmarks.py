#!/usr/bin/env python
"""
Benchmark runner for elote.

This script runs benchmarks and generates a report with performance metrics.
It can be used to track performance changes over time.

Usage:
    python scripts/run_benchmarks.py [--compare BASELINE]

Options:
    --compare BASELINE    Compare results with a baseline JSON file
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path


def run_benchmarks():
    """Run benchmarks and return the JSON output."""
    benchmark_dir = Path("benchmark_results")
    benchmark_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_output = benchmark_dir / f"benchmark_{timestamp}.json"
    
    cmd = [
        "uv", "run", "pytest", 
        "tests/test_benchmarks.py", 
        "--benchmark-json", 
        str(json_output),
        "--benchmark-enable",
        "-v"
    ]
    
    print(f"Running benchmarks: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    return json_output


def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def generate_report(current_file, baseline_file=None):
    """Generate a performance report."""
    current_data = load_json(current_file)
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"BENCHMARK RESULTS: {current_file}")
    print("=" * 80)
    
    # Extract benchmark data
    benchmarks = current_data.get('benchmarks', [])
    
    # Group by test name
    grouped = {}
    for bench in benchmarks:
        name = bench['name']
        group = name.split('[')[0] if '[' in name else name
        if group not in grouped:
            grouped[group] = []
        grouped[group].append(bench)
    
    # Print results by group
    for group, benches in sorted(grouped.items()):
        print(f"\n## {group}")
        print("-" * 80)
        print(f"{'Test':<50} {'Min (ms)':<12} {'Mean (ms)':<12} {'Max (ms)':<12} {'StdDev':<12}")
        print("-" * 80)
        
        for bench in sorted(benches, key=lambda x: x['name']):
            name = bench['name']
            if '[' in name:
                name = name.split('[')[1].rstrip(']')
            else:
                name = "default"
                
            min_time = bench['stats']['min'] * 1000  # Convert to ms
            mean_time = bench['stats']['mean'] * 1000
            max_time = bench['stats']['max'] * 1000
            stddev = bench['stats']['stddev'] * 1000
            
            print(f"{name:<50} {min_time:<12.3f} {mean_time:<12.3f} {max_time:<12.3f} {stddev:<12.3f}")
    
    # Compare with baseline if provided
    if baseline_file:
        print("\n" + "=" * 80)
        print(f"COMPARISON WITH BASELINE: {baseline_file}")
        print("=" * 80)
        
        baseline_data = load_json(baseline_file)
        baseline_benchmarks = {b['name']: b for b in baseline_data.get('benchmarks', [])}
        
        print(f"{'Test':<50} {'Current (ms)':<12} {'Baseline (ms)':<12} {'Change %':<12}")
        print("-" * 80)
        
        for bench in benchmarks:
            name = bench['name']
            if name in baseline_benchmarks:
                current_mean = bench['stats']['mean'] * 1000
                baseline_mean = baseline_benchmarks[name]['stats']['mean'] * 1000
                change_pct = ((current_mean - baseline_mean) / baseline_mean) * 100
                
                # Use color indicators for performance changes
                if change_pct > 5:  # Worse performance
                    change_str = f"\033[91m{change_pct:+.2f}%\033[0m"  # Red
                elif change_pct < -5:  # Better performance
                    change_str = f"\033[92m{change_pct:+.2f}%\033[0m"  # Green
                else:  # Similar performance
                    change_str = f"{change_pct:+.2f}%"
                
                print(f"{name:<50} {current_mean:<12.3f} {baseline_mean:<12.3f} {change_str:<12}")
            else:
                print(f"{name:<50} {bench['stats']['mean'] * 1000:<12.3f} {'N/A':<12} {'N/A':<12}")
    
    print("\n" + "=" * 80)
    return current_file


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks for elote")
    parser.add_argument("--compare", help="Compare with baseline JSON file")
    args = parser.parse_args()
    
    # Run benchmarks
    result_file = run_benchmarks()
    
    # Generate report
    baseline_file = args.compare
    generate_report(result_file, baseline_file)
    
    print(f"\nBenchmark results saved to: {result_file}")
    print("To compare with these results in the future, run:")
    print(f"  python scripts/run_benchmarks.py --compare {result_file}")


if __name__ == "__main__":
    main() 