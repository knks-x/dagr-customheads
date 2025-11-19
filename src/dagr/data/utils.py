import numpy as np
import torch
from torch_geometric.data import Data


def to_data(**kwargs):
    # convert all tracks to correct format
    for k, v in kwargs.items():
        if k.startswith("bbox"):
            kwargs[k] = torch.from_numpy(v)

    xy = np.stack([kwargs['x'], kwargs['y']], axis=-1).astype("int16")

    t = kwargs['t'].astype("int32")
    p = kwargs['p'].reshape((-1,1))

    kwargs['x'] = torch.from_numpy(p)
    kwargs['pos'] = torch.from_numpy(xy)
    kwargs['t'] = torch.from_numpy(t)

    """# --- Debug print for min/max ---
    print("----Debug inside/after data to_data")
    print(f"[DEBUG to_data] pos x: min={xy[:,0].min()}, max={xy[:,0].max()}")
    print(f"[DEBUG to_data] pos y: min={xy[:,1].min()}, max={xy[:,1].max()}")
    print(f"[DEBUG to_data] t: min={t.min()}, max={t.max()}, Δt={t.max()-t.min()}")
    print(f"[DEBUG to_data] polarity: min={p.min()}, max={p.max()}")

    print("[DEBUG] --- GT Boxes inside to data ---")
    # debug print for all bbox keys
    for k in kwargs:
        if k.startswith("bbox"):
            print(f"[DEBUG] --- GT Boxes '{k}' ---")
            print(f"First 5 boxes:\n{kwargs[k][:5]}")
            print(f"Total boxes: {kwargs[k].shape[0]}")"""

    return Data(**kwargs)
