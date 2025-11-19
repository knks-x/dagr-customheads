import os
import h5py
import csv
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

def count_events(h5_path):
    with h5py.File(h5_path, "r") as f:
        # the four arrays have equal length, pick one (e.g., x)
        n_events = f["events/x"].shape[0]
    return n_events

def count_events_in_folder(folder_path):
    # find .h5 or .hdf5 file inside this folder
    h5_files = [f for f in os.listdir(folder_path) if f.endswith((".h5", ".hdf5"))]
    if len(h5_files) == 0:
        raise FileNotFoundError(f"No hdf5 file found in {folder_path}")
    if len(h5_files) > 1:
        print("Warning: multiple hdf5 files, using first:", h5_files)
    h5_path = os.path.join(folder_path, h5_files[0])

    with h5py.File(h5_path, "r") as f:
        n_events = f["events/x"].shape[0]
    return n_events

def event_dist(folder_path=str):
    # Read CSV
    df = pd.read_csv(folder_path)  # replace with your CSV file path

    # Plot
    plt.bar(df["folder"], df["n_events"], edgecolor="black")

    # Labels and title
    plt.xlabel("Folder Name")
    plt.ylabel("Number of Events")
    plt.title("Events per Folder")

    # Rotate x labels if they’re long
    plt.xticks(rotation=30, ha="right")

    # Show plot
    plt.tight_layout()
    plt.show()

    plt.savefig("events_histogram.png", dpi=300)

def event_dist_sorted(folder_path=str, plot=False):
    df = pd.read_csv(folder_path)  # replace with your CSV file path
    
    #ensure numeric values
    df["n_events"] = pd.to_numeric(df["n_events"], errors="coerce")

    # Sort by events (descending)
    df_sorted = df.sort_values("n_events", ascending=False).reset_index(drop=True)

    # Plot
    plt.bar(df["folder"], df_sorted["n_events"], edgecolor="black")

    # Labels and title
    #plt.xlabel("Folder Name")
    plt.ylabel("Number of Events")
    plt.title("Events per Folder")

    # Rotate x labels if they’re long
    plt.xticks(rotation=30, ha="right")

    # Layout fix
    plt.tight_layout()

    # Save the plot
    plt.savefig("events_histogram_sorted.png", dpi=300)

    if plot:
        min_val = df['n_events'].min()
        max_val = df['n_events'].max()
        mean_val = df['n_events'].mean()
        median_val = df['n_events'].median()
        std_val = df['n_events'].std()

        print(f"Min events: {min_val}")
        print(f"Max events: {max_val}")
        print(f"Mean events: {mean_val:.2f}")
        print(f"Median events: {median_val}")
        print(f"Std dev: {std_val:.2f}")

        # Plot boxplot
        plt.figure(figsize=(8, 6))
        plt.boxplot(df['n_events'], vert=True, patch_artist=True, showfliers=True)
        plt.ylabel("Number of Events")
        plt.title("Distribution of Events per Folder")
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Save boxplot
        plt.tight_layout()
        plt.savefig("events_boxplot.png", dpi=300)


