from pathlib import Path
from typing import Optional, Callable

import os 
import json
import numpy as np 
import pandas as pd 
import h5py 
import hdf5plugin 
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random 

from scipy.interpolate import interp1d


from torch.utils.data import Dataset
from torch_geometric.data import Data

from dagr.data.ball_utils import to_data, crop_tracks

#from dagr.data.utils import to_data
from dagr.data.augment import init_transforms

class BallRealDataset(Dataset):
    def __init__(self, 
                root: Path,
                height=720, 
                width=1280, 
                scale=2,
                nr_events_window=250,                          #at max 1000 events per window
                overlap_ratio=0,                                #consecutive windows will overlap by 50%
                split= str, 
                transform: Optional[Callable]=None,
                debug=False,
                ball_diameter_min=35,
                ball_diameter_max=45,
                max_files=None,                               
                split_dir= Path,
                crop_to_dsec = False):                           

        """self.root = root
        self.scale = scale
        self.height = height                                   
        self.width = width                                    
        self.overlap_ratio = overlap_ratio
        self.split = split
        self.nr_events_window = nr_events_window
        self.max_files = max_files
        self.box_size_min = ball_diameter_min
        self.box_size_max = ball_diameter_max
        self.time_window = 1000000
        self.classes = ("object")
        self.debug = debug
        self.split_dir= split_dir
        self.width_dsec=640
        self.height_dsec=430
        self.width_dsec_scaled = self.width_dsec // self.scale    #320                               
        self.height_dsec_scaled = self.height_dsec // self.scale    #215
        """

        self.split = split
        self.root = root / self.split
        self.overlap_ratio = overlap_ratio
        self.split_path = self.root / self.split
        self.scale = scale
        self.nr_events_window = nr_events_window
        self.time_window = 1000000
        self.classes = ("object")
        self.transform = transform
        self.box_size = 40                          # approximate ball size in original space (e.g., 40 px)
        self.event_files = []
        self.position_files= None
        self.sample_indices = []
                
        self.crop_to_dsec = crop_to_dsec
        self.height_org= height
        self.width_org=width
        self.height = self.height_org // self.scale
        self.width = self.width_org // self.scale

        #load files
        all_files = sorted([f for f in os.listdir(self.root) if f.endswith(".hdf5")])
        if max_files:
            all_files = all_files[:max_files]
        self.event_files = [os.path.join(self.root, f) for f in all_files]

            
        if crop_to_dsec:
            self.crop_events_around_gt(crop_width=640, crop_height=430)
            self.height = 640 // self.scale
            self.width = 430 // self.scale

        self.sample_indices = []
        self._calculate_sample_indices()
        

        if transform is not None and hasattr(transform, "transforms"):
            init_transforms(transform.transforms, self.height, self.width)
        self.transform = transform

    def _calculate_sample_indices(self):
        """Create sample indices for all windows in all files."""
        for file_idx, f in enumerate(self.event_files):
            event_file = self.root / f
            with h5py.File(event_file, "r") as h5f:
                n_events = len(h5f["event_x"])
            step_size = int(self.nr_events_window * (1 - self.overlap_ratio))
            n_windows = max(0, (n_events - self.nr_events_window) // step_size + 1)
            for w in range(n_windows):
                self.sample_indices.append((file_idx, w))

    def __len__(self):
        return len(self.sample_indices)

    def _load_file_data(self, file_idx):
        """Load events and GT positions for one file."""
        fpath = self.event_files[file_idx]
        with h5py.File(fpath, "r") as f:
            x = f["event_x"][:].astype(np.float32)
            y = f["event_y"][:].astype(np.float32)
            p = f["event_p"][:].astype(np.int8)
            ts = f["event_t"][:].astype(np.float32)

            gt_x = f["gt_x"][:].astype(np.float32)
            gt_y = f["gt_y"][:].astype(np.float32)
            gt_t = f["gt_t"][:].astype(np.float32)

        # Sort by timestamp
        sort_idx = np.argsort(ts)
        x, y, p, ts = x[sort_idx], y[sort_idx], p[sort_idx], ts[sort_idx]

        if len(gt_t) == 0 or len(gt_x) == 0 or len(gt_y) == 0:
            print(f"⚠️ Warning: No valid GT data in file {fpath}, skipping.")
            return {
                "x": np.array([]),
                "y": np.array([]),
                "p": np.array([]),
                "ts": np.array([]),
                "interp_x": None,
                "interp_y": None,
                "gt_times": np.array([])
            }

        interp_x = interp1d(gt_t, gt_x, kind="linear", fill_value="extrapolate")
        interp_y = interp1d(gt_t, gt_y, kind="linear", fill_value="extrapolate")

        return {
            "x": x,
            "y": y,
            "p": p,
            "ts": ts,
            "interp_x": interp_x,
            "interp_y": interp_y,
            "gt_times": gt_t
        }

    def _get_window_from_file(self, file_idx, window_idx):
        """Extract a window of events and corresponding GT bbox."""
        data = self._load_file_data(file_idx)
        if len(data["x"]) == 0 or data["interp_x"] is None:
            return None, None, None, None

        x, y, p, ts = data["x"], data["y"], data["p"], data["ts"]
        interp_x, interp_y = data["interp_x"], data["interp_y"]

        step_size = int(self.nr_events_window * (1 - self.overlap_ratio))
        start_idx = window_idx * step_size
        end_idx = start_idx + self.nr_events_window
        if end_idx > len(x):
            end_idx = len(x)
            start_idx = max(0, end_idx - self.nr_events_window)

        window_x = x[start_idx:end_idx]
        window_y = y[start_idx:end_idx]
        window_p = p[start_idx:end_idx]
        window_ts = ts[start_idx:end_idx]

        if len(window_x) == 0:
            return None, None, None, None

        ts_0, ts_1 = window_ts[0], window_ts[-1]

        #gt at the end
        gt_time = window_ts[-1]
        gt_x_center = float(interp_x(gt_time))
        gt_y_center = float(interp_y(gt_time))

        # Compute top-left of bbox
        gt_x = gt_x_center - self.box_size / 2
        gt_y = gt_y_center - self.box_size / 2

        # Clip to frame boundaries (original size)
        gt_x = np.clip(gt_x, 0, self.width_org - self.box_size)
        gt_y = np.clip(gt_y, 0, self.height_org - self.box_size)

        gt_x_scaled = gt_x / self.scale
        gt_y_scaled = gt_y / self.scale
        box_size_scaled = self.box_size / self.scale
        
        """if self.crop_to_dsec:
            gt_x = np.clip(gt_x, 0, self.width - self.box_size)
            gt_y = np.clip(gt_y, 0, self.height - self.box_size)"""

        #gt_x_l = gt_x - self.box_size / 2
        #gt_y_l = gt_y - self.box_size / 2
        #gt_x_scaled = gt_x_l / self.scale
        #gt_y_scaled = gt_y_l / self.scale
        #box_size_scaled = self.box_size / self.scale

        bbox = np.array([[
            gt_x_scaled,
            gt_y_scaled,
            box_size_scaled,
            box_size_scaled,
            0,
            1
        ]], dtype=np.float32)

        events = np.column_stack([window_x, window_y, window_ts, window_p]).astype(np.float32)

        return events, bbox, ts_0, ts_1

    
    def preprocess_ball_events(self, events):
        # events: [N, 4] -> x, y, t, p

        if len(events) == 0:
            return {"x": np.array([], dtype=np.int32),
                    "y": np.array([], dtype=np.int32),
                    "t": np.array([], dtype=np.int64),
                    "p": np.array([], dtype=np.int8)}

        # convert array to dict for consistency with DSEC
        events_dict = {
            "x": events[:, 0].astype(np.int32),
            "y": events[:, 1].astype(np.int32),
            "t": events[:, 2].astype(np.int64),
            "p": events[:, 3].astype(np.int8),
        }

        #ensure events are integers
        events_dict["x"] = events_dict["x"].astype(np.int32)
        events_dict["y"] = events_dict["y"].astype(np.int32)

        # relative timestamps (same as DSEC)
        if len(events_dict["t"]) > 0:
            events_dict["t"] = self.time_window + events_dict["t"] - events_dict["t"][-1]

        # polarity in {-1, 1}
        events_dict["p"] = 2 * events_dict["p"] - 1

        events_dict["p"] = events_dict["p"].reshape(-1, 1)

        return events_dict


    def __getitem__(self, idx):
        file_idx, window_idx = self.sample_indices[idx]
        events, bbox, t0, t1 = self._get_window_from_file(file_idx, window_idx)
        
        if events is None:
            # safety fallback
            return self.__getitem__((idx + 1) % len(self))

        events = self.preprocess_ball_events(events)
    
        data=to_data(**events, 
        bbox0=bbox, 
        t0= t0,
        t1= t1,
        width=self.width, 
        height=self.height,
        time_window=self.time_window, 
        scale=self.scale )

        return data
    
    def crop_events_around_gt(self, crop_width=640, crop_height=430):

        cropped_files = []

        for event_file in self.event_files:
            cropped_path = event_file.replace(".hdf5", f"_croparound_{crop_width}x{crop_height}.hdf5")

            # If cropped file already exists, reuse it
            if os.path.exists(cropped_path):
                print(f"♻️ Cropped file already exists, reusing: {cropped_path}")
                cropped_files.append(cropped_path)
                continue

            with h5py.File(event_file, "r") as f:
                x = f["event_x"][:].astype(np.float32)
                y = f["event_y"][:].astype(np.float32)
                p = f["event_p"][:].astype(np.int8)
                t = f["event_t"][:].astype(np.float32)

                gt_x = f["gt_x"][:].astype(np.float32)
                gt_y = f["gt_y"][:].astype(np.float32)
                gt_t = f["gt_t"][:].astype(np.float32)

            if len(gt_x) == 0:
                print(f"⚠️ No GT in {event_file}, skipping.")
                continue

            # Compute GT trajectory center (mean position)
            cx_mean = np.mean(gt_x)
            cy_mean = np.mean(gt_y)

            # Define crop boundaries centered on GT trajectory
            x_min = max(0, cx_mean - crop_width / 2)
            y_min = max(0, cy_mean - crop_height / 2)
            x_max = x_min + crop_width
            y_max = y_min + crop_height

            # Keep only events inside crop
            mask = (x >= x_min) & (x < x_max) & (y >= y_min) & (y < y_max)
            if np.sum(mask) == 0:
                print(f"⚠️ No events left after cropping {event_file}, skipping.")
                continue

            # Shift coordinates so top-left of crop = (0,0)
            x_crop = x[mask] - x_min
            y_crop = y[mask] - y_min
            p_crop = p[mask]
            t_crop = t[mask]

            # Also crop & shift GT positions
            gt_mask = (gt_x >= x_min) & (gt_x < x_max) & (gt_y >= y_min) & (gt_y < y_max)
            gt_x_crop = gt_x[gt_mask] - x_min
            gt_y_crop = gt_y[gt_mask] - y_min
            gt_t_crop = gt_t[gt_mask]

            if len(gt_x_crop) == 0:
                print(f"⚠️ No GT inside crop for {event_file}, skipping.")
                continue

            # Save cropped file
            with h5py.File(cropped_path, "w") as f_out:
                f_out.create_dataset("event_x", data=x_crop, dtype="float32")
                f_out.create_dataset("event_y", data=y_crop, dtype="float32")
                f_out.create_dataset("event_p", data=p_crop, dtype="int8")
                f_out.create_dataset("event_t", data=t_crop, dtype="float32")

                f_out.create_dataset("gt_x", data=gt_x_crop, dtype="float32")
                f_out.create_dataset("gt_y", data=gt_y_crop, dtype="float32")
                f_out.create_dataset("gt_t", data=gt_t_crop, dtype="float32")

            print(f"✅ Saved cropped file: {cropped_path}")
            cropped_files.append(cropped_path)

        # Replace dataset list with cropped versions
        if len(cropped_files) > 0:
            self.event_files = cropped_files
            print(f"✅ Cropped {len(cropped_files)} files to {crop_width}×{crop_height} (saved as *_croparound_*.hdf5).")
        else:
            print("⚠️ No valid cropped files produced.")


   
    
    def visualize_events_with_gt(self, batch, sample_idx=0, save_path=None, title="Loaded Events + GT"):
        """
        Visualize events + GT box for one sample inside a DataBatch.
        """
        mask = batch.batch == sample_idx
        pos = batch.pos[mask].cpu().numpy()
        p = batch.x[mask].cpu().numpy().squeeze()

        x, y, t = pos.T
        colors = np.where(p > 0, "red", "blue")

        plt.figure(figsize=(6, 6))
        plt.scatter(x, y, c=colors, s=1, alpha=0.7)

        # bbox for this sample
        bbox = batch.bbox0[sample_idx].cpu().numpy()
        x0, y0, w, h, cat, conf = bbox
        x0 /= self.width 
        y0 /= self.height 
        w  /= self.width 
        h  /= self.height 
        rect = patches.Rectangle((x0, y0), w, h, linewidth=1,
                                edgecolor="lime", facecolor="none")
        plt.gca().add_patch(rect)

        plt.gca().invert_yaxis()
        plt.xlim([0, 1.2])
        plt.ylim([0, 1.2])
        plt.title(title)
        
        if save_path is not None:
            plt.savefig(save_path, dpi=150)
            plt.close()  # close to free memory
        else:
            plt.show()


    def visualize_events(self, batch, sample_idx=0, save_path=None, title="Event Window"):
        mask = batch.batch == sample_idx  
        pos = batch.pos[mask].cpu().numpy()  # [N, 3] -> x,y,t
        p = batch.x[mask].cpu().numpy().squeeze()  # polarity
        
        x, y, t = pos.T
        colors = np.where(p > 0, "red", "blue")
        fig, ax = plt.subplots(figsize=(6,6), dpi=700)

        #plt.figure(figsize=(6, 6))
        #plt.tight_layout()
        plt.scatter(x, y, c=colors, s=1, alpha=0.7)
        plt.gca().invert_yaxis()
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        #plt.title(title)

        #ax.set_xlabel("x", fontsize=22, fontweight='bold', fontname='Coruier')
        #ax.set_ylabel("y", fontsize=22, fontweight='bold', fontname='Coruier')

        ax.set_yticklabels(ax.get_yticklabels(), fontsize=18, fontname='Coruier', fontweight='bold')  
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=18, fontname='Coruier', fontweight='bold') 

        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_color("black")

        if save_path is not None:
            plt.savefig(save_path, dpi=150)
            print("Saved!")
            plt.close()  # close to free memory
        else:
            plt.show()


    def visualize_events_with_gt_and_centers(self, batch, sample_idx=0, save_path=None, title="Events + GT + Centers"):
        """
        Visualize events, GT box, GT center, and event centroid for one sample in DataBatch.
        """
        mask = batch.batch == sample_idx
        pos = batch.pos[mask].cpu().numpy()   # events [N,3] -> x,y,t
        p = batch.x[mask].cpu().numpy().squeeze()  # polarity

        x, y, t = pos.T
        colors = np.where(p > 0, "red", "blue")

        plt.figure(figsize=(6, 6))
        plt.scatter(x, y, c=colors, s=1, alpha=0.6)

        # --- GT bbox for this sample ---
        bbox = batch.bbox0[sample_idx].cpu().numpy()
        x0, y0, w, h, cat, conf = bbox
        x0 /= self.width 
        y0 /= self.height 
        w  /= self.width 
        h  /= self.height 

        rect = patches.Rectangle((x0, y0), w, h, linewidth=1,
                                edgecolor="lime", facecolor="none", label="GT Box")
        plt.gca().add_patch(rect)

        # --- GT center ---
        gt_cx = x0 + w / 2 
        gt_cy = y0 + h / 2 
        plt.scatter(gt_cx, gt_cy, c="lime", marker="x", s=40, label="GT Center")

        # --- Event centroid ---
        ev_cx = np.mean(x)
        ev_cy = np.mean(y)
        #plt.scatter(ev_cx, ev_cy, c="magenta", marker="+", s=80, label="Event Centroid")

        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal')
        plt.xlim([0, 1.2])
        plt.ylim([0, 1.2])
        plt.title(title)
        plt.legend(loc="upper right")

        if save_path is not None:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()
    

    def visualize_events_with_gt_and_centers_normalized(self, batch, sample_idx=0, save_path=None, title="Events + GT + Centers"):
        """
        Visualize events, GT box, GT center, and event centroid for one sample inside a DataBatch,
        all normalized to [0,1] using original width and height.
        """
        # Original image size
        width = float(batch.width[sample_idx])
        height = float(batch.height[sample_idx])

        # Select only events for this sample
        mask = batch.batch == sample_idx
        pos = batch.pos[mask].cpu().numpy()   # shape [N, 3], x,y,(t)
        p = batch.x[mask].cpu().numpy().squeeze()

        x, y, t = pos.T
        colors = np.where(p > 0, "red", "blue")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(x , y , c=colors, s=1, alpha=0.7, label="Events")

        # bbox for this sample
        bbox = batch.bbox0[sample_idx].cpu().numpy()
        gt_x, gt_y, box_w, box_h, cat, conf = bbox

        # Plot GT box normalized
        rect = patches.Rectangle(
            (gt_x / width, gt_y / height),
            box_w / width,
            box_h / height,
            linewidth=2, edgecolor="lime", facecolor="none", label="GT Box"
        )
        ax.add_patch(rect)

        # Plot GT center
        gt_center_x = gt_x / width + (box_w / (2 * width))
        gt_center_y = gt_y / height + (box_h / (2 * height))
        ax.scatter([gt_center_x], [gt_center_y], c="lime", marker="x", s=80, label="GT Center")

        # Plot event centroid (normalized)
        event_center_x = x.mean() 
        event_center_y = y.mean() 
        #ax.scatter([event_center_x], [event_center_y], c="magenta", marker="+", s=80, label="Event Centroid")

        ax.invert_yaxis()
        ax.set_xlim([0, 2])
        ax.set_ylim([0, 2])
        ax.set_title(title)
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc="upper right")

        if save_path is not None:
            plt.savefig(save_path, dpi=150)
            plt.close(fig)
        else:
            plt.show()


