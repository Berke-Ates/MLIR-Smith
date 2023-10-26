#!/bin/python3

# Copyright (c) 2023, Scalable Parallel Computing Lab, ETH Zurich

# This script plots the results from the `measure_resources.sh` script.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os

# Function to create and save the plots
def plot(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: The file '{csv_path}' does not exist.")
        return

    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Create Memory Usage vs. Code Size plot
    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=df, x='Code Size (B)', y='Memory Usage (KB)')
    plt.title('Memory Usage vs. Code Size')
    plt.xlabel('Code Size (B)')
    plt.ylabel('Memory Usage (KB)')
    plt.savefig('memory_vs_loc.pdf')  # Save the plot
    plt.close()  # Close the plot to prevent it from being displayed

    # Create Execution Time vs. Code Size plot
    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=df, x='Code Size (B)', y='Execution Time (s)')
    plt.title('Execution Time vs. Code Size')
    plt.xlabel('Code Size (B)')
    plt.ylabel('Execution Time (s)')
    plt.savefig('time_vs_loc.pdf')  # Save the plot
    plt.close()  # Close the plot to prevent it from being displayed

# Check if the CSV file path is provided as a command-line argument
if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <csv_file_path>")
else:
    csv_path = sys.argv[1]
    plot(csv_path)


