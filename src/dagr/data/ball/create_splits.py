
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json
import os
import numpy as np

from dagr.data.ball_data import BallDataset
from dagr.utils.args import FLAGS


def average_map(folder_maps):
    if len(folder_maps) == 0:
        raise ValueError("folder_maps is empty! Cannot compute average map.")
        
    maps = [f["map"] for f in folder_maps]  # pick the 2D map
    return np.mean(maps, axis=0)

def plot_heatmap(avg_map, split, out_dir=None):
    """
    Plot heatmap from average histogram map.
    avg_map: (H, W, 2) array (neg, pos channels)
    """
    title= f"average {split} coverage"
    filename = f"avg_{split} _heatmap.png"
    save_path = os.path.join(out_dir, filename)


    # Collapse channels: sum neg+pos or take one channel separately
    coverage = avg_map.sum(axis=-1)   # shape (H, W)

    print("avg_map shape:", avg_map.shape)
    print("coverage shape:", coverage.shape)


    plt.figure(figsize=(6,5))
    plt.imshow(coverage, cmap="hot", interpolation="nearest")
    plt.colorbar(label="Relative event probability")
    plt.title(title)
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

def compare_trajectories(train_maps, val_maps, test_maps, out_dir=None):

    filename = "pca_ball_trajectories"
    save_path = os.path.join(out_dir, filename)
    # Collect all vectors + labels
    all_vecs = [f["vector"] for f in train_maps] + \
            [f["vector"] for f in val_maps] + \
            [f["vector"] for f in test_maps]

    labels = (["train"] * len(train_maps) + 
            ["val"] * len(val_maps) + 
            ["test"] * len(test_maps))

    # PCA to 2D
    pca = PCA(n_components=2)
    vecs_pca = pca.fit_transform(all_vecs)

    plt.figure(figsize=(8,6))
    for split in ["train", "val", "test"]:
        idx = [i for i, l in enumerate(labels) if l == split]
        plt.scatter(vecs_pca[idx,0], vecs_pca[idx,1], label=split, alpha=0.6)

    plt.legend()
    plt.title("PCA of Ball Trajectories (by split)")

    if out_dir:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

