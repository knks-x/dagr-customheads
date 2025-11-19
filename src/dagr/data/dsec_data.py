from pathlib import Path
from typing import Optional, Callable

from torch_geometric.data import Dataset

import numpy as np
import cv2

import torch
from functools import lru_cache

from dsec_det.dataset import DSECDet

from dsec_det.io import yaml_file_to_dict
from dagr.data.dsec_utils import filter_tracks, crop_tracks, rescale_tracks, compute_class_mapping, map_classes, filter_small_bboxes
from dsec_det.directory import BaseDirectory
from dagr.data.augment import init_transforms
from dagr.data.utils import to_data

from dagr.visualization.bbox_viz import draw_bbox_on_img
from dagr.visualization.event_viz import draw_events_on_image


def tracks_to_array(tracks):
    return np.stack([tracks['x'], tracks['y'], tracks['w'], tracks['h'], tracks['class_id']], axis=1)

def interpolate_tracks(detections_0, detections_1, t):

    #debugging
    if len(detections_0) != len(detections_1):
        
        ids0 = detections_0['track_id'] if 'track_id' in detections_0.dtype.names else None
        ids1 = detections_1['track_id'] if 'track_id' in detections_1.dtype.names else None
        raise AssertionError(
            f"Mismatch at t={t}: len(d0)={len(detections_0)}, len(d1)={len(detections_1)}; "
            f"track_ids0={ids0}, track_ids1={ids1}"
        )
    #previous this:
    #assert len(detections_1) == len(detections_0)
    
    if len(detections_0) == 0:
        return detections_1

    t0 = detections_0['t'][0]
    t1 = detections_1['t'][0]

    assert t0 < t1

    # need to sort detections
    detections_0 = detections_0[detections_0['track_id'].argsort()]
    detections_1 = detections_1[detections_1['track_id'].argsort()]

    r = ( t - t0 ) / ( t1 - t0 )
    detections_out = detections_0.copy()
    for k in 'xywh':
        detections_out[k] = detections_0[k] * (1 - r) + detections_1[k] * r

    return detections_out

class EventDirectory(BaseDirectory):
    @property
    @lru_cache
    def event_file(self):
        return self.root / "left/events_2x.h5"


class DSEC(Dataset):
    MAPPING = dict(pedestrian="object", rider=None, car="object", bus="object", truck="object", bicycle=None,
                   motorcycle=None, train=None)
    def __init__(self,
                 root: Path,
                 split: str,
                 transform: Optional[Callable]=None,
                 debug=False,
                 min_bbox_diag=0,
                 min_bbox_height=0,
                 scale=2,
                 cropped_height=430,
                 only_perfect_tracks=False,
                 demo=False,
                 no_eval=False,
                 sync: str = "back"):

        Dataset.__init__(self)

        split_config = None
        if not demo:
            split_config = yaml_file_to_dict(Path(__file__).parent / "dsec_split.yaml")
            assert split in split_config.keys(), f"'{split}' not in {list(split_config.keys())}"

        self.dataset = DSECDet(root=root, split=split, sync=sync, debug=debug, split_config=split_config)

        for directory in self.dataset.directories.values():
            directory.events = EventDirectory(directory.events.root)

        self.scale = scale
        self.width = self.dataset.width // scale
        self.height = cropped_height // scale
        self.classes = ("object")
        self.time_window = 1000000
        self.min_bbox_height = min_bbox_height
        self.min_bbox_diag = min_bbox_diag
        self.debug = debug
        self.num_us = -1

        self.class_remapping = compute_class_mapping(self.classes, self.dataset.classes, self.MAPPING)

        if transform is not None and hasattr(transform, "transforms"):
            init_transforms(transform.transforms, self.height, self.width)

        self.transform = transform
        self.no_eval = no_eval

        if self.no_eval:
            only_perfect_tracks = False

        #valid (consecutive) frame pairs and tracks after filtering out operation
        self.image_index_pairs, self.track_masks = filter_tracks(dataset=self.dataset, image_width=self.width,
                                                                 image_height=self.height,
                                                                 class_remapping=self.class_remapping,
                                                                 min_bbox_height=min_bbox_height,
                                                                 min_bbox_diag=min_bbox_diag,
                                                                 only_perfect_tracks=only_perfect_tracks,
                                                                 scale=scale)

    def set_num_us(self, num_us):
        self.num_us = num_us

    def visualize_debug(self, index, log_to_wandb=False):
        data = self.__getitem__(index)
        image = data.image[0].permute(1,2,0).numpy()
        p = data.x[:,0].numpy()
        x, y = data.pos.t().numpy()
        b_x, b_y, b_w, b_h, b_c = data.bbox.t().numpy()

        image = draw_events_on_image(image, x, y, p)
        image = draw_bbox_on_img(image, b_x, b_y, b_w, b_h,
                                 b_c, np.ones_like(b_c), conf=0.3, nms=0.65)

        #added this to log the visualization wandb
        if log_to_wandb:
            wandb.log({f"visualization/frame_{index}": wandb.Image(image, caption=f"Debug {index}")})
        else:
            cv2.imshow(f"Debug {index}", image)
            cv2.waitKey(0)


    def __len__(self):
        return sum(len(d) for d in self.image_index_pairs.values())

    def preprocess_detections(self, detections):
        detections = rescale_tracks(detections, self.scale)
        detections = crop_tracks(detections, self.width, self.height)
        detections['class_id'], _ = map_classes(detections['class_id'], self.class_remapping)
        return detections

    def preprocess_events(self, events):
        mask = events['y'] < self.height
        events = {k: v[mask] for k, v in events.items()}
        if len(events['t']) > 0:
            events['t'] = self.time_window + events['t'] - events['t'][-1]
        events['p'] = 2 * events['p'].reshape((-1,1)).astype("int8") - 1
        return events

    def preprocess_image(self, image):
        image = image[:self.scale * self.height]
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = image.unsqueeze(0)
        return image

    def __getitem__(self, idx):
        dataset, image_index_pairs, track_masks, idx = self.rel_index(idx)
  
        image_index_0, image_index_1 = image_index_pairs[idx]
        #TODO image_ts are not correct like this in front mode!
        image_ts_0, image_ts_1 = dataset.images.timestamps[[image_index_0, image_index_1]]

        #loads tracks for image_index_0 (relevant for front)
        detections_0 = self.dataset.get_tracks(image_index_0, mask=track_masks, directory_name=dataset.root.name)
        #loads tracks for image_index_1 (relevant for back -looking in the future)
        detections_1 = self.dataset.get_tracks(image_index_1, mask=track_masks, directory_name=dataset.root.name)

        detections_0 = self.preprocess_detections(detections_0)
        detections_1 = self.preprocess_detections(detections_1)

        #always load image_index_0 for both front and back
        image_0 = self.dataset.get_image(image_index_0, directory_name=dataset.root.name)
        image_0 = self.preprocess_image(image_0)

        #loads events correctly for front: index-1 to index)
        events = self.dataset.get_events(image_index_0, directory_name=dataset.root.name)

        
        if self.num_us >= 0:
            image_ts_1 = image_ts_0 + self.num_us
            events = {k: v[events['t'] < image_ts_1] for k, v in events.items()}
            if not self.no_eval:
                detections_1 = interpolate_tracks(detections_0, detections_1, image_ts_1)

        # here, the timestamp of the events is no longer absolute but relative to the last event
        events = self.preprocess_events(events)

        # convert to torch geometric data
        data = to_data(**events, 
                        bbox=tracks_to_array(detections_1), 
                        bbox0=tracks_to_array(detections_0), 
                        t0=image_ts_0, 
                        t1=image_ts_1,
                        width=self.width, 
                        height=self.height, 
                        time_window=self.time_window,
                        image=image_0, 
                        sequence=str(dataset.root.name))

        if self.transform is not None:
            data = self.transform(data)

        # remove bboxes if they have 0 width or height
        mask = filter_small_bboxes(data.bbox[:, 2], data.bbox[:, 3], self.min_bbox_height, self.min_bbox_diag)
        data.bbox = data.bbox[mask]
        mask = filter_small_bboxes(data.bbox0[:, 2], data.bbox0[:, 3], self.min_bbox_height, self.min_bbox_diag)
        data.bbox0 = data.bbox0[mask]

        return data

    def rel_index(self, idx, r_sequence=False):                         
    #returns sequence(dir), image pairs, tracks and local idx(in sequence) for a global idx i (=sample) that iterates through dataset.                                                                     
                                                                        #e.g.: idx=501 -> sequence B idx 1 (imagine 2000 samples in total, 4 sequences with A=500 samples, B=500samples,..)
        
        for folder in self.dataset.subsequence_directories:             #checks which sequence and which frame pair the current/loaded sample belongs to (by subtracting len(sequence) from global idx -> if idx < as amount)
            name = folder.name
            image_index_pairs = self.image_index_pairs[name]            #return only image_index_pairs for the current sequence          
            directory = self.dataset.directories[name]
            track_mask = self.track_masks[name]
            if idx < len(image_index_pairs):                            #if idx<len of sequence -> correct sequence. else substract lenght of sequence from the idx
                if not r_sequence:
                    return directory, image_index_pairs, track_mask, idx        #returned idx is idx is position of the tuple inside the sequence’s image_index_pairs array
                else:
                    return directory, image_index_pairs, track_mask, idx, name
            idx -= len(image_index_pairs)                               
        raise IndexError
    