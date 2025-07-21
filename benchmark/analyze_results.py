#!/usr/bin/env python3
"""
YOLOs-CPP Benchmark Analysis and Visualization Tool
Generates comparison charts and analysis from benchmark CSV results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import numpy as np
from pathlib import Path

def load_benchmark_data(csv_file):
    """Load benchmark data from CSV file"""
    try:
        df = pd.read_csv(csv_file)
        
        # Filter out invalid rows (status messages, incomplete data)
        # Valid rows should have model_type as 'yolo8', 'yolo11', etc.
        valid_models = ['yolo8', 'yolo11', 'yolo5', 'yolo7', 'yolo9', 'yolo10', 'yolo12']
        df = df[df['model_type'].isin(valid_models)]
        
        # Convert numeric columns to float, replacing any non-numeric values with NaN
        numeric_columns = ['load_ms', 'preprocess_ms', 'inference_ms', 'postprocess_ms', 
                          'total_ms', 'fps', 'memory_mb', 'system_memory_mb', 
                          'cpu_usage_%', 'gpu_usage_%', 'gpu_memory_mb', 
                          'latency_avg_ms', 'latency_min_ms', 'latency_max_ms', 
                          'map_score', 'frame_count']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with all NaN numeric values
        df = df.dropna(subset=numeric_columns, how='all')
        
        # Replace negative system memory with absolute value (monitoring issue)
        if 'system_memory_mb' in df.columns:
            df['system_memory_mb'] = df['system_memory_mb'].abs()
        
        # Replace NaN values in critical columns with default values
        df['fps'] = df['fps'].fillna(0)
        df['latency_avg_ms'] = df['latency_avg_ms'].fillna(0)
        df['cpu_usage_%'] = df['cpu_usage_%'].fillna(0)
        
        print(f"Loaded {len(df)} valid benchmark records from {csv_file}")
        
        if len(df) == 0:
            print("Warning: No valid benchmark records found!")
            return None
            
        return df
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")
        return None

def create_performance_comparison(df, output_dir):
    """Create performance comparison charts"""
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. FPS Comparison by Model and Environment
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('YOLOs-CPP Performance Analysis', fontsize=16, fontweight='bold')
    
    # FPS comparison
    ax1 = axes[0, 0]
    fps_data = df.groupby(['model_type', 'environment'])['fps'].mean().unstack()
    fps_data.plot(kind='bar', ax=ax1, rot=45)
    ax1.set_title('Average FPS by Model and Environment')
    ax1.set_ylabel('Frames Per Second (FPS)')
    ax1.legend(title='Environment')
    ax1.grid(True, alpha=0.3)
    
    # Latency comparison
    ax2 = axes[0, 1]
    latency_data = df.groupby(['model_type', 'environment'])['latency_avg_ms'].mean().unstack()
    latency_data.plot(kind='bar', ax=ax2, rot=45)
    ax2.set_title('Average Latency by Model and Environment')
    ax2.set_ylabel('Latency (ms)')
    ax2.legend(title='Environment')
    ax2.grid(True, alpha=0.3)
    
    # Memory usage comparison
    ax3 = axes[1, 0]
    memory_data = df.groupby(['model_type', 'environment'])['system_memory_mb'].mean().unstack()
    memory_data.plot(kind='bar', ax=ax3, rot=45)
    ax3.set_title('Memory Usage by Model and Environment')
    ax3.set_ylabel('Memory Usage (MB)')
    ax3.legend(title='Environment')
    ax3.grid(True, alpha=0.3)
    
    # CPU vs GPU Usage
    ax4 = axes[1, 1]
    cpu_gpu_data = df[['model_type', 'cpu_usage_%', 'gpu_usage_%', 'environment']]
    cpu_gpu_melted = pd.melt(cpu_gpu_data, 
                            id_vars=['model_type', 'environment'], 
                            value_vars=['cpu_usage_%', 'gpu_usage_%'],
                            var_name='resource_type', value_name='usage_%')
    sns.barplot(data=cpu_gpu_melted, x='model_type', y='usage_%', 
                hue='resource_type', ax=ax4)
    ax4.set_title('Resource Usage by Model')
    ax4.set_ylabel('Usage (%)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created performance comparison chart")

def create_detailed_analysis(df, output_dir):
    """Create detailed analysis charts"""
    
    # Performance vs Resource Usage Scatter Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # FPS vs CPU Usage
    ax1 = axes[0]
    for env in df['environment'].unique():
        env_data = df[df['environment'] == env]
        # Filter out NaN values
        env_data_clean = env_data.dropna(subset=['cpu_usage_%', 'fps'])
        
        if len(env_data_clean) > 0:
            ax1.scatter(env_data_clean['cpu_usage_%'], env_data_clean['fps'], 
                       label=env, alpha=0.7, s=60)
    
    ax1.set_xlabel('CPU Usage (%)')
    ax1.set_ylabel('FPS')
    ax1.set_title('Performance vs CPU Usage')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Latency Distribution
    ax2 = axes[1]
    for model in df['model_type'].unique():
        model_data = df[df['model_type'] == model]['latency_avg_ms']
        # Filter out NaN and infinite values
        model_data = model_data.dropna()
        model_data = model_data[np.isfinite(model_data)]
        
        if len(model_data) > 0:
            ax2.hist(model_data, alpha=0.6, label=f'{model}', bins=20)
    
    ax2.set_xlabel('Latency (ms)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Latency Distribution by Model')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detailed_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created detailed analysis chart")

def create_cost_efficiency_analysis(df, output_dir):
    """Create cost-efficiency analysis"""
    
    # Assume some cost metrics (these would be actual RunPod costs)
    cost_per_hour = {'CPU': 0.50, 'GPU': 2.00}  # Example costs in USD/hour
    
    # Calculate efficiency metrics
    df['cost_per_hour'] = df['environment'].map(cost_per_hour)
    df['frames_per_dollar'] = (df['fps'] * 3600) / df['cost_per_hour']  # frames per dollar per hour
    df['efficiency_score'] = df['fps'] / df['cost_per_hour']  # FPS per dollar per hour
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Cost efficiency by model
    ax1 = axes[0]
    efficiency_data = df.groupby(['model_type', 'environment'])['efficiency_score'].mean().unstack()
    efficiency_data.plot(kind='bar', ax=ax1, rot=45)
    ax1.set_title('Cost Efficiency (FPS per $/hour)')
    ax1.set_ylabel('FPS per Dollar per Hour')
    ax1.legend(title='Environment')
    ax1.grid(True, alpha=0.3)
    
    # Performance vs Cost scatter
    ax2 = axes[1]
    for model in df['model_type'].unique():
        model_data = df[df['model_type'] == model]
        ax2.scatter(model_data['cost_per_hour'], model_data['fps'], 
                   label=model, alpha=0.7, s=60)
    
    ax2.set_xlabel('Cost per Hour ($)')
    ax2.set_ylabel('FPS')
    ax2.set_title('Performance vs Cost')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cost_efficiency.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created cost efficiency analysis")

def generate_summary_report(df, output_dir):
    """Generate a comprehensive summary report"""
    
    report_file = os.path.join(output_dir, 'benchmark_analysis_report.md')
    
    with open(report_file, 'w') as f:
        f.write("# YOLOs-CPP Benchmark Analysis Report\n\n")
        f.write(f"**Generated**: {pd.Timestamp.now()}\n")
        f.write(f"**Total Benchmarks**: {len(df)}\n\n")
        
        # Model Summary
        f.write("## Model Performance Summary\n\n")
        model_summary = df.groupby('model_type').agg({
            'fps': ['mean', 'std'],
            'latency_avg_ms': ['mean', 'std'],
            'cpu_usage_%': ['mean'],
            'gpu_usage_%': ['mean'],
            'memory_mb': ['mean']
        }).round(2)
        
        # Convert to simple table format instead of using tabulate
        f.write("| Model | FPS (avg) | Latency (avg) | CPU % (avg) | GPU % (avg) | Memory (avg) |\n")
        f.write("|-------|-----------|---------------|-------------|-------------|-------------|\n")
        
        for idx, row in model_summary.iterrows():
            f.write(f"| {idx} | {row[('fps', 'mean')]:.2f} | {row[('latency_avg_ms', 'mean')]:.2f} | {row[('cpu_usage_%', 'mean')]:.2f} | {row[('gpu_usage_%', 'mean')]:.2f} | {row[('memory_mb', 'mean')]:.2f} |\n")
        f.write("\n\n")
        
        # Environment Comparison
        f.write("## Environment Comparison\n\n")
        env_summary = df.groupby('environment').agg({
            'fps': ['mean', 'std'],
            'latency_avg_ms': ['mean', 'std'],
            'load_ms': ['mean']
        }).round(2)
        
        # Convert to simple table format
        f.write("| Environment | FPS (avg) | FPS (std) | Latency (avg) | Latency (std) | Load Time (avg) |\n")
        f.write("|-------------|-----------|-----------|---------------|---------------|----------------|\n")
        
        for idx, row in env_summary.iterrows():
            f.write(f"| {idx} | {row[('fps', 'mean')]:.2f} | {row[('fps', 'std')]:.2f} | {row[('latency_avg_ms', 'mean')]:.2f} | {row[('latency_avg_ms', 'std')]:.2f} | {row[('load_ms', 'mean')]:.2f} |\n")
        f.write("\n\n")
        
        # Key Insights
        f.write("## Key Insights\n\n")
        
        # Best performing model
        best_fps_model = df.loc[df['fps'].idxmax()]
        f.write(f"- **Best FPS Performance**: {best_fps_model['model_type']} on {best_fps_model['environment']} "
                f"with {best_fps_model['fps']:.2f} FPS\n")
        
        # Lowest latency
        best_latency_model = df.loc[df['latency_avg_ms'].idxmin()]
        f.write(f"- **Lowest Latency**: {best_latency_model['model_type']} on {best_latency_model['environment']} "
                f"with {best_latency_model['latency_avg_ms']:.2f}ms\n")
        
        # Most efficient
        df['efficiency'] = df['fps'] / (df['cpu_usage_%'] + df['gpu_usage_%'] + 1)  # Avoid division by zero
        best_efficiency_model = df.loc[df['efficiency'].idxmax()]
        f.write(f"- **Most Resource Efficient**: {best_efficiency_model['model_type']} on {best_efficiency_model['environment']}\n")
        
        # Performance delta
        if len(df['environment'].unique()) > 1:
            cpu_avg_fps = df[df['environment'] == 'CPU']['fps'].mean()
            gpu_avg_fps = df[df['environment'] == 'GPU']['fps'].mean()
            if not pd.isna(cpu_avg_fps) and not pd.isna(gpu_avg_fps):
                performance_delta = ((gpu_avg_fps - cpu_avg_fps) / cpu_avg_fps) * 100
                f.write(f"- **GPU Performance Delta**: {performance_delta:.1f}% improvement over CPU\n")
        
        f.write("\n## Recommendations\n\n")
        f.write("- For **maximum throughput**: Use GPU environment with the best performing model\n")
        f.write("- For **cost efficiency**: Consider CPU environment for non-critical applications\n")
        f.write("- For **real-time applications**: Focus on models with lowest latency\n")
        f.write("- For **resource-constrained environments**: Use the most efficient model/environment combination\n")
        
        f.write("\n## Charts Generated\n\n")
        f.write("- `performance_comparison.png`: Overall performance comparison\n")
        f.write("- `detailed_analysis.png`: Detailed performance analysis\n")
        f.write("- `cost_efficiency.png`: Cost-efficiency analysis\n")
    
    print(f"Generated comprehensive report: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze YOLOs-CPP benchmark results')
    parser.add_argument('csv_file', help='Path to the benchmark CSV file')
    parser.add_argument('--output-dir', '-o', default='./analysis_output', 
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    df = load_benchmark_data(args.csv_file)
    if df is None:
        return
    
    # Generate analysis
    print("Generating performance analysis...")
    create_performance_comparison(df, args.output_dir)
    create_detailed_analysis(df, args.output_dir)
    create_cost_efficiency_analysis(df, args.output_dir)
    generate_summary_report(df, args.output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")
    print("\nFiles generated:")
    for file in os.listdir(args.output_dir):
        print(f"  - {file}")

if __name__ == "__main__":
    main()
