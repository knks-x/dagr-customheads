import numpy as np
import h5py
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import csv

import torch.nn.functional as F

def to_data(**kwargs):
    # convert all tracks to correct format
    for k, v in kwargs.items():
        if k.startswith("bbox"):
            kwargs[k] = torch.from_numpy(v)

    xy = np.stack([kwargs['x'], kwargs['y']], axis=-1).astype("int16")
    xy[:, 0] = xy[:, 0] / kwargs['scale']                               #scale x and y. gt boxes are scaled when created
    xy[:, 1] = xy[:, 1] / kwargs['scale']

    t = kwargs['t'].astype("int32")
    p = kwargs['p'].reshape((-1,1))

    kwargs['x'] = torch.from_numpy(p)                                   #polarity is stored in x lol
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

def crop_tracks(tracks, width, height):
    tracks = tracks.copy()
    
    x1 = tracks[:, 0]
    y1 = tracks[:, 1]
    x2 = x1 + tracks[:, 2]
    y2 = y1 + tracks[:, 3]

    x1 = np.clip(x1, 0, width-1)
    x2 = np.clip(x2, 0, width-1)
    y1 = np.clip(y1, 0, height-1)
    y2 = np.clip(y2, 0, height-1)

    tracks[:, 0] = x1
    tracks[:, 1] = y1
    tracks[:, 2] = x2 - x1
    tracks[:, 3] = y2 - y1

    return tracks

def visualize_events_after(events, idx, title="Events after model processing", save_path=None):

    mask = events.batch == idx
    pos = events.pos[mask].cpu().numpy()
    num_events = len(pos)

    p = events.x[mask].detach().cpu().numpy().squeeze()
    x, y, t = pos.T

    if p.ndim > 1:
        p = p[:, 0]
    colors = np.where(p > 0, "red", "blue").ravel()

    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(x, y, c=colors, s=1, alpha=0.05, label="Events")

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()

def visualize_events_with_boxes(events, gt_boxes, idx,  pred_boxes=None, img_width=None, img_height=None, 
                                title="Pred Boxes + Targets", save_path=None, line_pred=None):
    
    mask = events.batch == idx
    pos = events.pos[mask].cpu().numpy()
    num_events = len(pos)

    p = events.x[mask].detach().cpu().numpy().squeeze()
    x, y, t = pos.T

    if p.ndim > 1:
        p = p[:, 0]
    colors = np.where(p > 0, "red", "blue").ravel()

    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(x, y, c=colors, s=1, alpha=0.3)
    
    gt_boxes = gt_boxes.detach().cpu().numpy()
    for b in gt_boxes:
        cx, cy, w, h = b[:4]
        
        #top-left corner for rectangle
        x0 = cx - w / 2
        y0 = cy - h / 2
        rect = patches.Rectangle(
            (x0/img_width, y0/img_height),
            w/img_width,
            h/img_height,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
            label="GT"
        )
        ax.add_patch(rect)
        #ax.scatter([cx/img_width], [cy/img_height], c="lime", marker="x", s=60)

    if pred_boxes is not None:
        pred_boxes = pred_boxes.detach().cpu().numpy()
        for b in pred_boxes:
            cx, cy, w, h = b[:4]

            #top-left corner for rectangle
            x0 = cx - w / 2
            y0 = cy - h / 2
            rect = patches.Rectangle(
                (x0/img_width, y0/img_height),
                w/img_width,
                h/img_height,
                linewidth=line_pred,  #0.7 many boxes - 1.7 one box
                edgecolor="black",
                facecolor="none",
                label="Pred"
            )
            ax.add_patch(rect)
            ax.scatter([cx/img_width], [cy/img_height], c="black", marker="+", s=10)
    
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    #ax.set_title(title)

    #ax.set_xlabel("x", fontsize=22, fontweight='bold', fontname='Coruier')
    #ax.set_ylabel("y", fontsize=22, fontweight='bold', fontname='Coruier')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=14, fontname='Coruier', fontweight='bold')  
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=14, fontname='Coruier', fontweight='bold')
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    #ax.legend(by_label.values(), by_label.keys())
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def visualize_boxes(gt_boxes, idx, pred_boxes=None, top_boxes=None, img_width=None, img_height=None, 
                    title="Pred Boxes + Targets", save_path=None, line_pred=1):
    
    fig, ax = plt.subplots(figsize=(10,8), dpi=700)
    
    gt_boxes = gt_boxes.detach().cpu().numpy()
    for b in gt_boxes:
        cx, cy, w, h = b[:4]
        x0 = cx - w / 2
        y0 = cy - h / 2
        rect = patches.Rectangle(
            (x0 / img_width, y0 / img_height),
            w / img_width,
            h / img_height,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
            label="GT"
        )
        ax.add_patch(rect)
        ax.scatter([cx / img_width], [cy / img_height], c="lime", marker="x", s=60)

    if pred_boxes is not None:
        pred_boxes = pred_boxes.detach().cpu().numpy()
        for b in pred_boxes:
            cx, cy, w, h = b[:4]
            # top-left corner for rectangle
            x0 = cx - w / 2
            y0 = cy - h / 2
            rect = patches.Rectangle(
                (x0 / img_width, y0 / img_height),
                w / img_width,
                h / img_height,
                linewidth=line_pred,                                                    #e.g. 0.7 for many boxes, 1.7 for one box
                edgecolor="black",
                facecolor="none",
                label="Pred"
            )
            ax.add_patch(rect)
            ax.scatter([cx / img_width], [cy / img_height], c="black", marker="+", s=10)

    if top_boxes is not None:
        top_boxes = top_boxes.detach().cpu().numpy()
        for b in top_boxes:
            cx, cy, w, h = b[:4]
            x0 = cx - w / 2
            y0 = cy - h / 2
            rect = patches.Rectangle(
                (x0 / img_width, y0 / img_height),
                w / img_width,
                h / img_height,
                linewidth=line_pred,
                edgecolor="red",
                facecolor="none",
                label="Top Score"
            )
            ax.add_patch(rect)
            ax.scatter([cx / img_width], [cy / img_height], c="red", marker="+", s=30)

    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    ax.set_xlabel("x", fontsize=22, fontweight='bold', fontname='Coruier')
    ax.set_ylabel("y", fontsize=22, fontweight='bold', fontname='Coruier')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=18, fontname='Coruier', fontweight='bold')  
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=18, fontname='Coruier', fontweight='bold') 

    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color("black")
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    #ax.legend(by_label.values(), by_label.keys())
    
    if save_path is not None:
        plt.savefig(save_path, dpi=600)
        plt.close(fig)
    else:
        plt.show()


def select_top_bboxes(bbox_preds, obj_preds, keep_ratio=0.25):
    assert bbox_preds.shape[0] == obj_preds.shape[0], "Batch size mismatch"
    if obj_preds.ndim == 3:
        obj_preds = obj_preds.squeeze(-1)
        
    B, N, _ = bbox_preds.shape
    k = max(1, int(N * keep_ratio))

    top_boxes = []
    top_scores = []

    for i in range(B):
        topk_scores, topk_idx = torch.topk(obj_preds[i], k, dim=0)
        top_boxes.append(bbox_preds[i][topk_idx])
        top_scores.append(topk_scores)

    top_boxes = torch.stack(top_boxes, dim=0)
    top_scores = torch.stack(top_scores, dim=0)

    return top_boxes, top_scores

def save_detections_csv(target, predictions, filepath):
    #predictions/t  are already in cx format
    with open(filepath, "w", newline="") as f:
        flat_target = target.view(-1)
        writer = csv.writer(f)
        writer.writerow(["cx", "cy", "w", "h", "conf"])
        writer.writerow([float(x) for x in flat_target.tolist()])  
        for row in predictions:
            writer.writerow(row.tolist())
