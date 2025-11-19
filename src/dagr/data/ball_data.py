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
from dagr.data.augment import init_transforms

class BallDataset(Dataset):
    def __init__(self, 
                root: Path,
                height=720, 
                width=1280, 
                scale=2,
                nr_events_window=500,                         
                overlap_ratio=0,                                #consecutive windows will overlap by x%
                split= str, 
                transform: Optional[Callable]=None,
                debug=False,
                ball_diameter_min=35,
                ball_diameter_max=45,
                max_files=40,                               
                split_dir= Path,
                crop_to_dsec = False):                           

        self.root = root
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
        self.width_dsec_scaled = self.width_dsec // self.scale      #320                               
        self.height_dsec_scaled = self.height_dsec // self.scale    #215
        self.crop_to_dsec = crop_to_dsec

        if crop_to_dsec:
            self.height = self.height_dsec_scaled
            self.width = self.width_dsec_scaled

        if transform is not None and hasattr(transform, "transforms"):
            init_transforms(transform.transforms, self.height, self.width)

        self.transform = transform

        self.event_files = []
        self.gt_files = []
        self.metadata_files = []
        self.folder_maps = []                              #save normalized histogram of each folder for comparison
        self.folder_names = []

        self.sample_indices = []                           #list of tuples for each sample (which file/simulation + which event window)
        
        self._load_file_list()                             #loads event, gt, metadata files (after setting train, val, test split)
        self._calculate_sample_indices()

    def print_event_stats(self, file_data, n_samples=5):
        """
        Print basic statistics about events in a loaded file.
        file_data: dictionary returned by _load_file_data
        n_samples: number of first events to print
        """
        x, y, p, ts = file_data['x'], file_data['y'], file_data['p'], file_data['ts']
        
        print(f"Number of events: {len(x)}")
        print(f"x: min={x.min()}, max={x.max()}")
        print(f"y: min={y.min()}, max={y.max()}")
        print(f"polarity: min={p.min()}, max={p.max()}")
        print(f"timestamps: min={ts.min()}, max={ts.max()}, Δt={ts.max()-ts.min()}")
        
        print("\nFirst few events:")
        for i in range(min(n_samples, len(x))):
            print(f"Event {i}: x={x[i]}, y={y[i]}, p={p[i]}, t={ts[i]}")
        
        centroid_x = x.mean()
        centroid_y = y.mean()
        #print(f"\nEvent centroid: x={centroid_x:.2f}, y={centroid_y:.2f}")

    def print_gt_stats(self, gt_file, show_columns=None):
        gt = pd.read_csv(gt_file)

        if show_columns is None:
            # default: all numeric columns except 'frame'
            show_columns = [c for c in gt.columns if c != "frame"]

        print(f"GT file: {gt_file}")
        for c in show_columns:
            col_min = gt[c].min()
            col_max = gt[c].max()
            print(f"  {c}: min={col_min}, max={col_max}")

            if "x" in c.lower() and col_max > 617.5:
                print(f"    ⚠️ WARNING: {c} exceeds X bound (617.5) — max={col_max:.3f}")
            if "y" in c.lower() and col_max > 407.5:
                print(f"    ⚠️ WARNING: {c} exceeds Y bound (407.5) — max={col_max:.3f}")
        print("-" * 40)

    def crop_events_mask(self):
        """Crop dataset to dsec range using GT positions and update metadata with new video length.
        Cropped events are saved as new HDF5 files to allow re-use."""
        
        kept_event_files, kept_gt_files, kept_metadata_files, kept_folder_names = [], [], [], []
        frames_kept_list = []

        for event_file, gt_file, metadata_file, folder_name in zip(
            self.event_files, self.gt_files, self.metadata_files, self.folder_names
        ):

            #path to save cropped events
            folder_path = os.path.dirname(event_file)
            cropped_event_file = os.path.join(folder_path, f"{folder_name}_events_cropped_to_640x430.hdf5")

            metadata = pd.read_csv(metadata_file)
            total_frames = int(metadata["total_frames"].iloc[0])                                    # usually 500
            video_length = float(metadata["video_length"].iloc[0])
            dt_us = video_length / total_frames * 1_000_000                                                         #compute delta t = time per frame ->convert to microseconds

            gt = pd.read_csv(gt_file)                                                                              # frame,screen_x,screen_y
            box_buffer = self.box_size_max/2
            inside_mask = (gt["screen_x"] < self.width_dsec-box_buffer) & (gt["screen_y"] < self.height_dsec-box_buffer)               #crop to dsec
            valid_frames = gt.index[inside_mask].tolist()

            #save new gt file at least locally
            gt_cropped = gt[inside_mask].reset_index(drop=True)
            cropped_gt_file = gt_file.replace(".csv", "_cropped.csv")

            gt_cropped.to_csv(cropped_gt_file, index=False)
            #self.print_gt_stats(cropped_gt_file)
            
            if len(valid_frames) == 0 or len(valid_frames)<200:
                print(f"⚠️ {folder_name}: less then 200 frames inside boundaries -> skipped")
                continue
            
            #compute num of selected frames
            start_frame, end_frame = valid_frames[0], valid_frames[-1]
            frames_kept = end_frame - start_frame + 1

            frames_kept_list.append(frames_kept)    

            #compute the exact time window of selected event window to filter events by that
            start_ts, end_ts = start_frame * dt_us, (end_frame + 1) * dt_us

            #load original
            with h5py.File(event_file, "r") as f:
                x = f["events"]["x"][:]
                y = f["events"]["y"][:]
                t = f["events"]["t"][:]
                p = f["events"]["p"][:]

                mask = (t >= start_ts) & (t < end_ts) & (x < self.width_dsec) & (y < self.height_dsec)

                x_sel, y_sel, t_sel, p_sel = x[mask], y[mask], t[mask], p[mask]

                if len(x_sel) < 100_000:
                    print(f"⚠️ {folder_name}: less than 100,000 events after cropping -> skipped")
                    continue

            #save cropped
            with h5py.File(cropped_event_file, "w") as f_out:
                grp = f_out.create_group("events")
                grp.create_dataset("x", data=x_sel, dtype="int16")
                grp.create_dataset("y", data=y_sel, dtype="int16")
                grp.create_dataset("t", data=t_sel, dtype="int64")
                grp.create_dataset("p", data=p_sel, dtype="int8")

            print(f"{folder_name}: saved cropped events -> {cropped_event_file} "
               f"(kept {len(x_sel)} events)")

            #update
            ratio = frames_kept / total_frames
            new_video_length = video_length * ratio

            metadata["start_frame"] = start_frame
            metadata["end_frame"] = end_frame
            metadata["frames_kept"] = frames_kept
            metadata["new_video_length"] = new_video_length
            metadata.to_csv(metadata_file, index=False)

            #print(f"✅ {folder_name}: frames {start_frame}-{end_frame} kept "
            #   f"({frames_kept}/{total_frames}), new video length {new_video_length:.4f}")

            kept_event_files.append(cropped_event_file)  # always point to cropped file
            kept_gt_files.append(cropped_gt_file)
            kept_metadata_files.append(metadata_file)
            kept_folder_names.append(folder_name)

            """# Debug: show event stats for sanity check
            with h5py.File(cropped_event_file, "r") as f:
                x_vals = f["events"]["x"][:]
                y_vals = f["events"]["y"][:]
                n_events = len(x_vals)

                if n_events > 0:
                    print(f"{folder_name}: Events {n_events}, "
                        f"x in [{x_vals.min()}, {x_vals.max()}], "
                        f"y in [{y_vals.min()}, {y_vals.max()}]")
                else:
                    print(f"{folder_name}: !! No events left in cropped file!")"""
            
        self.event_files = kept_event_files
        self.gt_files = kept_gt_files
        self.metadata_files = kept_metadata_files
        self.folder_names = kept_folder_names

        print(f"\nFinal {self.split} dataset after cropping: {len(self.event_files)} files")

        #print global summary
        global_summary=True
        if global_summary:
            if frames_kept_list:
                frames_kept_arr = np.array(frames_kept_list)
                print("\n📊 Frames kept summary across dataset:")
                print(f"  Min frames kept: {frames_kept_arr.min()}")
                print(f"  Max frames kept: {frames_kept_arr.max()}")
                print(f"  Mean frames kept: {frames_kept_arr.mean():.2f}")
                print(f"  Median frames kept: {np.median(frames_kept_arr):.2f}")

                # Save histogram
                os.makedirs(self.split_dir, exist_ok=True)
                hist_path = os.path.join(self.split_dir, f"frames_kept_histogram_{self.split}.png")

                plt.figure(figsize=(6,4))
                plt.hist(frames_kept_arr, bins=20, edgecolor="black")
                plt.xlabel("Frames kept")
                plt.ylabel("Count")
                plt.title("Distribution of frames_kept after cropping")
                plt.tight_layout()
                plt.savefig(hist_path)
                plt.close()

                print(f"📈 Histogram saved to: {hist_path}")

    def _load_file_list(self):
        """Load list of event and GT files for the relevant split"""

        sample_folders = os.listdir(f"{self.root}/{self.split}")
        split_ratios = {"train": 0.7, "val": 0.2, "test": 0.1}
        if self.max_files is not None:
            max_to_select = int(self.max_files * split_ratios[self.split])
            if max_to_select > len(sample_folders):
                raise ValueError(f"max_to_select ({max_to_select}) exceeds number of available folders ({len(sample_folders)}) in {self.split} split")
        else:
            max_to_select = len(sample_folders)
        print("max to select:", max_to_select)
        random.seed(11)  
        selected_folders = random.sample(sample_folders, max_to_select)
        selected_indices =list(range(len(selected_folders)))                     
 
        for idx in selected_indices:
            folder_name = selected_folders[idx]                                                           #get the original folder name e.g "00001"
            folder_path = os.path.join(self.root, self.split, folder_name)
            
            if self.crop_to_dsec:
                event_file = os.path.join(folder_path, f"{folder_name}_events_cropped_to_640x430.hdf5")                       
                gt_file = os.path.join(folder_path, f"{folder_name}_ball_coords_cropped.csv")
                metadata_file = os.path.join(folder_path, f"{folder_name}_metadata.csv")
            else:   
                event_file = os.path.join(folder_path, f"{folder_name}_events.hdf5")                       
                gt_file = os.path.join(folder_path, f"{folder_name}_ball_coords.csv")
                metadata_file = os.path.join(folder_path, f"{folder_name}_metadata.csv")
            
            if os.path.exists(event_file) and os.path.exists(gt_file) and os.path.exists(metadata_file):
                self.event_files.append(event_file)
                self.gt_files.append(gt_file)
                self.metadata_files.append(metadata_file) 
                self.folder_names.append(folder_name)
        
        print(f"Loaded {len(self.event_files)} files for {self.split} split")

    def _calculate_sample_indices(self):
        """Calculate how many samples (windows with events) each file/simulation will produce after using sliding window: 
            breaks the sequence of event data in smaller, overlapping chunks
            depending on number of events per window and the overlap ratio: 1000 * (1-0.5)=500
            e.g. window 0: events 0-999
                 window 1: events 1000-1999
                 ...
            like slicing in dsec"""

        print("Calculating number of samples per file...")
        
        total_samples = 0
        #iterate through all subfolders
        for file_idx, (event_file, gt_file, metadata_file) in enumerate(zip(self.event_files, self.gt_files, self.metadata_files)):

            if not os.path.exists(event_file):
                print(f"⚠️ Event file does not exist: {event_file}")
                continue

            with h5py.File(event_file, 'r') as f:
                n_events = len(f['events']['x'])

            print(f"Number of events in file {event_file}: {n_events}")
            print(f"Events per window: {self.nr_events_window}")
            print(f"Overlap ratio: {self.overlap_ratio}")

            #step size decides how many events fit into one windwow
            step_size = int(self.nr_events_window * (1 - self.overlap_ratio))
            #calculates how many windows the subfolder will have
            n_windows = max(0, (n_events - self.nr_events_window) // step_size + 1)             #integer division: how many full steps fit into the extra events

            # registering all the sliding windows as individual samples
            for window_idx in range(n_windows):
                self.sample_indices.append((file_idx, window_idx))                              #sample index: tuple of file index (which subfolder/simulation) and which event window inside that simualtion
                total_samples += 1
            
        print(f"Total samples/ event windows: {total_samples}")
    
    def _load_file_data(self, file_idx):
        """Load and preprocess data from a single file"""

        #for cropped data: uses files after crop_mask
        event_file = self.event_files[file_idx]
        gt_file = self.gt_files[file_idx]
        metadata_file = self.metadata_files[file_idx]

        with h5py.File(event_file, 'r') as f:
            data = f['events']
            x = data['x'][:]
            y = data['y'][:]
            p = data['p'][:]
            ts = data['t'][:]

        # Sort by timestamp
        sort_idx = np.argsort(ts)
        x, y, p, ts = x[sort_idx], y[sort_idx], p[sort_idx], ts[sort_idx]

        gt = pd.read_csv(gt_file)
        gt_metadata = pd.read_csv(metadata_file)

        if self.crop_to_dsec:
            gt_max_time = gt_metadata["new_video_length"][0]
        else:
            gt_max_time = gt_metadata["video_length"][0]
        gt_times = np.linspace(0, gt_max_time, len(gt))

        # Create interpolation functions for GT coordinates (box coordinates)
        interp_x = interp1d(gt_times, gt['screen_x'].values, 
                           kind='linear', fill_value='extrapolate')
        interp_y = interp1d(gt_times, gt['screen_y'].values, 
                           kind='linear', fill_value='extrapolate')
        #for increasing ball size: scaling the box from min to max during the vidoe
        interp_box_size = interp1d(gt_times, np.linspace(self.box_size_min, self.box_size_max, len(gt)), 
                            kind='linear', fill_value='extrapolate')

        file_data = {
            'x': x,
            'y': y, 
            'p': p, 
            'ts': ts,
            'interp_x': interp_x,
            'interp_y': interp_y,
            'interp_box_size': interp_box_size,
            'gt_times': gt_times,
            'sequence': Path(event_file).stem.replace("_events", "")
        }

        return file_data
        
    def _get_window_from_file(self, file_idx, window_idx):
        """Extract a specific window from a file"""
        file_data = self._load_file_data(file_idx)

        #print("-----DEBUG inside get window from file")
        #self.print_event_stats(file_data)
        x, y, p, ts = file_data['x'], file_data['y'], file_data['p'], file_data['ts']
        gt_times=file_data['gt_times']
        interp_x, interp_y = file_data['interp_x'], file_data['interp_y']
        interp_box_size = file_data['interp_box_size']
        sequence = file_data["sequence"]

        #event-based windows / slicing
        step_size = int(self.nr_events_window * (1 - self.overlap_ratio))
        start_idx = window_idx * step_size
        end_idx = start_idx + self.nr_events_window
        
        if end_idx > len(x):
            # Pad or skip if not enough events
            end_idx = len(x)
            start_idx = max(0, end_idx - self.nr_events_window)
            
        #extract events in window
        window_x = x[start_idx:end_idx]
        window_y = y[start_idx:end_idx]
        window_p = p[start_idx:end_idx]
        window_ts = ts[start_idx:end_idx]
        
        if len(window_x) == 0:
            return None, None, None, None, None, None

        #extract ts of event windoww
        ts_0, ts_1 = window_ts[0], window_ts[-1]                    #start/end of event window   → like image_ts_0 in dsec

        #get box at the end of event window
        gt_time = window_ts[-1] / 1e6  
        gt_x = float(interp_x(gt_time))
        gt_y = float(interp_y(gt_time))
        box_size = float(interp_box_size(gt_time))

        #downscale gt_boxes
        gt_x_l = gt_x - box_size / 2 
        gt_y_l = gt_y - box_size / 2 
        gt_x_scaled = gt_x_l / self.scale  
        gt_y_scaled = gt_y_l / self.scale
        box_size_scaled = box_size / self.scale

        #bbox = matches bbox/ convert to training format of dsec --- these are the targets. position of the ball at the end of the time_window
        bbox = np.array([[
            gt_x_scaled,                                    # x (top-left) 
            gt_y_scaled,                                     # y (top-left) y increases downwards
            box_size_scaled,                                # width
            box_size_scaled,                                       # height
            0,                                              # class_id
            1                                               # confidence
        ]], dtype=np.float32)

        #events array [x, y, t, p] = matches dsec
        events = np.column_stack([window_x, window_y, window_ts, window_p]).astype(np.float32)
        
        histogram = self._generate_histogram(events)

        return events, bbox, histogram, ts_0, ts_1, sequence
   
    def _generate_histogram(self, events):
        """Generate histogram representation from events"""

        if len(events) == 0:
            return np.zeros((self.height_dsec, self.width_dsec, 2), dtype=np.float32)
            
        x, y, t, p = events.T
        x = np.clip(x.astype(int), 0, self.width_dsec - 1)
        y = np.clip(y.astype(int), 0, self.height_dsec - 1)
        
        img_pos = np.zeros((self.height_dsec * self.width_dsec,), dtype=np.float32)
        img_neg = np.zeros((self.height_dsec * self.width_dsec,), dtype=np.float32)
        
        pos_mask = p > 0
        neg_mask = p <= 0
        
        if np.any(pos_mask):
            np.add.at(img_pos, x[pos_mask] + self.width_dsec * y[pos_mask], 1)
        if np.any(neg_mask):
            np.add.at(img_neg, x[neg_mask] + self.width_dsec * y[neg_mask], 1)
        
        histogram = np.stack([img_neg, img_pos], -1).reshape((self.height_dsec, self.width_dsec, 2))

        #normalize + flatten
        H_norm = histogram / (histogram.sum() + 1e-8)   # normalize per folder to relative spatial distribution of events (prob by pixel)
        H_norm_flat=H_norm.flatten()
        
        self.folder_maps.append({ 
            "map": H_norm,                              # keep the 2D spatial map
            "vector": H_norm_flat                       # store the flattened vector
        })
        #print(f"After append: {len(self.folder_maps)} entries")

        return histogram.astype(np.float32)

    def __len__(self):
        return len(self.sample_indices)
    
    def preprocess_ball_events(self, events):
        # events: [N, 4] -> x, y, t, p
        #mask = events[:, 1] < self.height
        #events = events[mask]

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

        #for the current idx that iterates through dataset: check which sequence and which event window it belongs to
        file_idx, window_idx = self.sample_indices[idx]                                 
        events, bbox0, histogram, start_ts, end_ts, sequence = self._get_window_from_file(file_idx, window_idx)
        
        #preprocess data in the same way as dsec
        events=self.preprocess_ball_events(events)

        #make some copies
        events=events.copy()
        bbox0=bbox0.copy()
        histogram=histogram.copy()

        # loader returns one sample
        data=to_data(**events,
                     bbox0=bbox0,
                     histogram=histogram,
                     t0= start_ts,
                     t1= end_ts,
                     width=self.width,
                     height=self.height,
                     time_window=self.time_window,
                     sequence=sequence,
                     scale=self.scale
                    )
                    
        if self.transform is not None:
            data = self.transform(data)

        return data
        

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
        rect = patches.Rectangle((x0, y0), w, h, linewidth=1, edgecolor="lime", facecolor="none")
        plt.gca().add_patch(rect)

        plt.gca().invert_yaxis()
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.title(title)
        
        if save_path is not None:
            plt.savefig(save_path, dpi=150)
            plt.close() 
        else:
            plt.show()

    def visualize_events(self, batch, sample_idx=0, save_path=None, title="Event Window"):
        mask = batch.batch == sample_idx  
        pos = batch.pos[mask].cpu().numpy()  # [N, 3] -> x,y,t
        p = batch.x[mask].cpu().numpy().squeeze()  # polarity
        
        x, y, t = pos.T
        colors = np.where(p > 0, "red", "blue")
        fig, ax = plt.subplots(figsize=(6,6), dpi=700)

        plt.scatter(x, y, c=colors, s=1, alpha=0.7)
        plt.gca().invert_yaxis()
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        #plt.title(title)

        ax.set_yticklabels(ax.get_yticklabels(), fontsize=18, fontname='Coruier', fontweight='bold')  
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=18, fontname='Coruier', fontweight='bold') 

        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_color("black")

        if save_path is not None:
            plt.savefig(save_path, dpi=150)
            print("Saved!")
            plt.close() 
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

        gt_cx = x0 + w / 2 
        gt_cy = y0 + h / 2 
        plt.scatter(gt_cx, gt_cy, c="lime", marker="x", s=40, label="GT Center")

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
    




