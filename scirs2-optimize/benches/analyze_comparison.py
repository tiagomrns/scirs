#!/usr/bin/env python3
"""
Analysis script to compare SciRS2-Optimize and SciPy benchmark results

This script loads benchmark results from both implementations and generates
detailed comparison reports, including performance ratios, accuracy comparisons,
and visualization plots.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os


def load_results(scipy_file: str = 'scipy_benchmark_results.json',
                scirs_file: str = 'scirs_benchmark_results.json') -> Tuple[Dict, Dict]:
    """Load benchmark results from JSON files"""
    
    scipy_results = {}
    scirs_results = {}
    
    if os.path.exists(scipy_file):
        with open(scipy_file, 'r') as f:
            scipy_results = json.load(f)
    else:
        print(f"Warning: {scipy_file} not found")
    
    if os.path.exists(scirs_file):
        with open(scirs_file, 'r') as f:
            scirs_results = json.load(f)
    else:
        print(f"Warning: {scirs_file} not found")
    
    return scipy_results, scirs_results


def compare_timing_performance(scipy_results: Dict, scirs_results: Dict) -> pd.DataFrame:
    """Compare timing performance between implementations"""
    
    comparisons = []
    
    # Compare unconstrained optimization
    if 'unconstrained' in scipy_results and 'unconstrained' in scirs_results:
        scipy_df = pd.DataFrame(scipy_results['unconstrained'])
        scirs_df = pd.DataFrame(scirs_results['unconstrained'])
        
        for _, scipy_row in scipy_df.iterrows():
            # Find matching SciRS2 result
            scirs_match = scirs_df[
                (scirs_df['problem'] == scipy_row['problem']) &
                (scirs_df['method'] == scipy_row['method'])
            ]
            
            if not scirs_match.empty:
                scirs_row = scirs_match.iloc[0]
                comparisons.append({
                    'category': 'unconstrained',
                    'problem': scipy_row['problem'],
                    'method': scipy_row['method'],
                    'scipy_time': scipy_row['avg_time'],
                    'scirs_time': scirs_row['avg_time'],
                    'speedup': scipy_row['avg_time'] / scirs_row['avg_time'],
                    'scipy_iters': scipy_row.get('iterations', -1),
                    'scirs_iters': scirs_row.get('iterations', -1),
                    'scipy_success': scipy_row.get('success', False),
                    'scirs_success': scirs_row.get('success', False),
                })
    
    return pd.DataFrame(comparisons)


def generate_performance_plots(comparison_df: pd.DataFrame):
    """Generate performance comparison plots"""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Speedup by method
    ax = axes[0, 0]
    speedup_by_method = comparison_df.groupby('method')['speedup'].agg(['mean', 'std'])
    speedup_by_method['mean'].plot(kind='bar', ax=ax, yerr=speedup_by_method['std'])
    ax.axhline(y=1, color='r', linestyle='--', label='Equal performance')
    ax.set_title('Average Speedup by Method (SciPy time / SciRS2 time)')
    ax.set_ylabel('Speedup Factor')
    ax.set_xlabel('Method')
    ax.legend()
    
    # 2. Speedup by problem
    ax = axes[0, 1]
    speedup_by_problem = comparison_df.groupby('problem')['speedup'].agg(['mean', 'std'])
    speedup_by_problem['mean'].plot(kind='bar', ax=ax, yerr=speedup_by_problem['std'])
    ax.axhline(y=1, color='r', linestyle='--', label='Equal performance')
    ax.set_title('Average Speedup by Problem')
    ax.set_ylabel('Speedup Factor')
    ax.set_xlabel('Problem')
    ax.legend()
    
    # 3. Iteration comparison
    ax = axes[1, 0]
    valid_iters = comparison_df[
        (comparison_df['scipy_iters'] > 0) & 
        (comparison_df['scirs_iters'] > 0)
    ]
    if not valid_iters.empty:
        ax.scatter(valid_iters['scipy_iters'], valid_iters['scirs_iters'])
        max_iters = max(valid_iters['scipy_iters'].max(), valid_iters['scirs_iters'].max())
        ax.plot([0, max_iters], [0, max_iters], 'r--', label='Equal iterations')
        ax.set_xlabel('SciPy Iterations')
        ax.set_ylabel('SciRS2 Iterations')
        ax.set_title('Iteration Count Comparison')
        ax.legend()
    
    # 4. Success rate comparison
    ax = axes[1, 1]
    scipy_success = comparison_df.groupby('method')['scipy_success'].mean()
    scirs_success = comparison_df.groupby('method')['scirs_success'].mean()
    
    x = np.arange(len(scipy_success))
    width = 0.35
    
    ax.bar(x - width/2, scipy_success, width, label='SciPy')
    ax.bar(x + width/2, scirs_success, width, label='SciRS2')
    ax.set_xlabel('Method')
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(scipy_success.index, rotation=45)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def generate_detailed_report(comparison_df: pd.DataFrame, 
                           output_file: str = 'comparison_report.md'):
    """Generate a detailed markdown report"""
    
    with open(output_file, 'w') as f:
        f.write("# SciRS2-Optimize vs SciPy Detailed Comparison Report\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        
        avg_speedup = comparison_df['speedup'].mean()
        f.write(f"- **Average Speedup**: {avg_speedup:.2f}x ")
        if avg_speedup > 1:
            f.write("(SciRS2 is faster)\n")
        else:
            f.write("(SciPy is faster)\n")
        
        success_comparison = (
            comparison_df['scirs_success'].sum() / len(comparison_df),
            comparison_df['scipy_success'].sum() / len(comparison_df)
        )
        f.write(f"- **Success Rates**: SciRS2: {success_comparison[0]:.1%}, "
                f"SciPy: {success_comparison[1]:.1%}\n")
        
        # Method-wise comparison
        f.write("\n## Method-wise Performance\n\n")
        f.write("| Method | Avg Speedup | SciRS2 Wins | SciPy Wins | Ties |\n")
        f.write("|--------|-------------|-------------|------------|------|\n")
        
        for method in comparison_df['method'].unique():
            method_data = comparison_df[comparison_df['method'] == method]
            avg_speedup = method_data['speedup'].mean()
            scirs_wins = (method_data['speedup'] > 1.1).sum()
            scipy_wins = (method_data['speedup'] < 0.9).sum()
            ties = len(method_data) - scirs_wins - scipy_wins
            
            f.write(f"| {method} | {avg_speedup:.2f}x | {scirs_wins} | "
                   f"{scipy_wins} | {ties} |\n")
        
        # Problem-wise comparison
        f.write("\n## Problem-wise Performance\n\n")
        f.write("| Problem | Avg Speedup | Best Method (SciRS2) | Best Method (SciPy) |\n")
        f.write("|---------|-------------|---------------------|--------------------|\n")
        
        for problem in comparison_df['problem'].unique():
            problem_data = comparison_df[comparison_df['problem'] == problem]
            avg_speedup = problem_data['speedup'].mean()
            
            best_scirs = problem_data.loc[problem_data['scirs_time'].idxmin()]
            best_scipy = problem_data.loc[problem_data['scipy_time'].idxmin()]
            
            f.write(f"| {problem} | {avg_speedup:.2f}x | "
                   f"{best_scirs['method']} ({best_scirs['scirs_time']*1000:.2f}ms) | "
                   f"{best_scipy['method']} ({best_scipy['scipy_time']*1000:.2f}ms) |\n")
        
        # Detailed Results
        f.write("\n## Detailed Results\n\n")
        f.write("### All Benchmark Results\n\n")
        f.write("| Problem | Method | SciPy Time (ms) | SciRS2 Time (ms) | Speedup | "
                "SciPy Iters | SciRS2 Iters |\n")
        f.write("|---------|--------|-----------------|------------------|---------|"
                "-------------|-------------|\n")
        
        for _, row in comparison_df.iterrows():
            f.write(f"| {row['problem']} | {row['method']} | "
                   f"{row['scipy_time']*1000:.2f} | {row['scirs_time']*1000:.2f} | "
                   f"{row['speedup']:.2f}x | {row['scipy_iters']} | "
                   f"{row['scirs_iters']} |\n")
        
        # Recommendations
        f.write("\n## Recommendations\n\n")
        f.write("Based on the benchmark results:\n\n")
        
        # Find best performers
        best_methods = comparison_df.groupby('method')['speedup'].mean().sort_values(ascending=False)
        
        f.write(f"1. **Best Overall Method**: {best_methods.index[0]} "
                f"(avg speedup: {best_methods.iloc[0]:.2f}x)\n")
        
        # Problem-specific recommendations
        f.write("\n2. **Problem-Specific Recommendations**:\n")
        for problem in comparison_df['problem'].unique():
            problem_data = comparison_df[comparison_df['problem'] == problem]
            best = problem_data.loc[problem_data['speedup'].idxmax()]
            f.write(f"   - {problem}: Use {best['method']} "
                   f"(speedup: {best['speedup']:.2f}x)\n")
        
        f.write("\n## Notes\n\n")
        f.write("- Speedup > 1 means SciRS2 is faster than SciPy\n")
        f.write("- Speedup < 1 means SciPy is faster than SciRS2\n")
        f.write("- Times are averaged over multiple runs\n")


def analyze_accuracy_differences(scipy_results: Dict, scirs_results: Dict):
    """Analyze accuracy differences between implementations"""
    
    accuracy_comparisons = []
    
    if 'unconstrained' in scipy_results and 'unconstrained' in scirs_results:
        scipy_df = pd.DataFrame(scipy_results['unconstrained'])
        scirs_df = pd.DataFrame(scirs_results['unconstrained'])
        
        for _, scipy_row in scipy_df.iterrows():
            scirs_match = scirs_df[
                (scirs_df['problem'] == scipy_row['problem']) &
                (scirs_df['method'] == scipy_row['method'])
            ]
            
            if not scirs_match.empty:
                scirs_row = scirs_match.iloc[0]
                
                # Compare final values
                scipy_val = scipy_row.get('final_value', np.nan)
                scirs_val = scirs_row.get('final_value', np.nan)
                
                if not np.isnan(scipy_val) and not np.isnan(scirs_val):
                    abs_diff = abs(scipy_val - scirs_val)
                    rel_diff = abs_diff / (abs(scipy_val) + 1e-10)
                    
                    accuracy_comparisons.append({
                        'problem': scipy_row['problem'],
                        'method': scipy_row['method'],
                        'scipy_value': scipy_val,
                        'scirs_value': scirs_val,
                        'absolute_diff': abs_diff,
                        'relative_diff': rel_diff,
                        'acceptable': abs_diff < 1e-6,
                    })
    
    return pd.DataFrame(accuracy_comparisons)


def main():
    """Run the comparison analysis"""
    
    print("Loading benchmark results...")
    scipy_results, scirs_results = load_results()
    
    if not scipy_results:
        print("\nNo SciPy results found. Running scipy_benchmarks.py first...")
        os.system("python scipy_benchmarks.py")
        scipy_results, _ = load_results()
    
    if not scipy_results or not scirs_results:
        print("Error: Missing benchmark results. Please run benchmarks first.")
        return
    
    print("\nComparing performance...")
    comparison_df = compare_timing_performance(scipy_results, scirs_results)
    
    if comparison_df.empty:
        print("No matching results found for comparison")
        return
    
    print("\nGenerating plots...")
    generate_performance_plots(comparison_df)
    
    print("\nAnalyzing accuracy...")
    accuracy_df = analyze_accuracy_differences(scipy_results, scirs_results)
    
    print("\nGenerating detailed report...")
    generate_detailed_report(comparison_df)
    
    print("\nComparison analysis complete!")
    print(f"- Performance plots saved to: performance_comparison.png")
    print(f"- Detailed report saved to: comparison_report.md")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    print(f"Average speedup: {comparison_df['speedup'].mean():.2f}x")
    print(f"Median speedup: {comparison_df['speedup'].median():.2f}x")
    print(f"Best speedup: {comparison_df['speedup'].max():.2f}x "
          f"({comparison_df.loc[comparison_df['speedup'].idxmax(), 'method']} on "
          f"{comparison_df.loc[comparison_df['speedup'].idxmax(), 'problem']})")
    
    if not accuracy_df.empty:
        print(f"\nAccuracy comparison:")
        print(f"Average absolute difference: {accuracy_df['absolute_diff'].mean():.2e}")
        print(f"Acceptable accuracy rate: {accuracy_df['acceptable'].mean():.1%}")


if __name__ == '__main__':
    main()