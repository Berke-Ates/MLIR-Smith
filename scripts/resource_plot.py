#!/bin/python3

# Copyright (c) 2023, Scalable Parallel Computing Lab, ETH Zurich

# This script plots the results from the `measure_resources.sh` script.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
import os


# Function to create and save the plots
def plot(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: The file '{csv_path}' does not exist.")
        return

    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Filter out entries where the code size is more than 30 KB
    # This leads to nicer plots as we use a geometric distributon
    df = df[df["Code Size (B)"] <= 20 * 1000]

    # Adjust the y-values from µs to ms by dividing by 1000
    df["Execution Time (ms)"] = df["Execution Time (µs)"] / 1000

    # Adjust the y-values from KB to MB by dividing by 1000
    df["Memory Usage (MB)"] = df["Memory Usage (KB)"] / 1000

    # Adjust the y-values from B to KB by dividing by 1000
    df["Code Size (KB)"] = df["Code Size (B)"] / 1000

    # Group in bins.
    bin_edges = np.linspace(
        df["Code Size (KB)"].min(), df["Code Size (KB)"].max(), 1001
    )
    df["Code Size Bin (KB)"] = pd.cut(
        df["Code Size (KB)"], bins=bin_edges, labels=bin_edges[1:], right=False
    )

    # For each code size, take the median time & memory.
    df = (
        df.groupby("Code Size Bin (KB)")
        .agg({"Execution Time (ms)": "median", "Memory Usage (MB)": "median"})
        .reset_index()
    )

    # Create Memory Usage vs. Code Size plot
    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=df, x="Code Size Bin (KB)", y="Memory Usage (MB)", color="#3c407c"
    )
    # plt.title("Memory Usage vs. Code Size", fontsize=22)
    plt.xlabel("Code Size (KB)", fontsize=20)
    plt.ylabel("Memory Usage (MB)", fontsize=20)
    plt.tick_params(labelsize=18)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig("memory_vs_loc.pdf")  # Save the plot
    plt.close()  # Close the plot to prevent it from being displayed

    # Create Execution Time vs. Code Size plot
    plt.figure(figsize=(10, 5))
    sns.lineplot(
        data=df, x="Code Size Bin (KB)", y="Execution Time (ms)", color="#3c407c"
    )
    # plt.title("Execution Time vs. Code Size", fontsize=22)
    plt.xlabel("Code Size (KB)", fontsize=20)
    plt.ylabel("Execution Time (ms)", fontsize=20)
    plt.tick_params(labelsize=18)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig("time_vs_loc.pdf")  # Save the plot
    plt.close()  # Close the plot to prevent it from being displayed


# Check if the CSV file path is provided as a command-line argument
if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <csv_file_path>")
else:
    csv_path = sys.argv[1]
    plot(csv_path)
