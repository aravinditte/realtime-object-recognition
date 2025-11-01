# Lightweight SORT tracker for realtime multi-object tracking
# Based on Kalman filter + IoU association (Hungarian assignment)

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

# Simple Kalman filter for bbox [cx, cy, w, h]
class KalmanBoxTracker:
    _count = 0
    def __init__(self, bbox):
        # bbox: [x1, y1, x2, y2]
        self.id = KalmanBoxTracker._count
        KalmanBoxTracker._count += 1
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w  = max(1.0, x2 - x1)
        h  = max(1.0, y2 - y1)
        # State: [cx, cy, w, h, vx, vy, vw, vh]
        self.x = np.array([[cx, cy, w, h, 0, 0, 0, 0]], dtype=np.float32).T
        self.P = np.eye(8, dtype=np.float32) * 10.0
        self.F = np.eye(8, dtype=np.float32)
        dt = 1.0
        for i in range(4):
            self.F[i, i+4] = dt
        self.H = np.zeros((4, 8), dtype=np.float32)
        self.H[0,0] = self.H[1,1] = self.H[2,2] = self.H[3,3] = 1.0
        self.R = np.eye(4, dtype=np.float32) * 1.0
        self.Q = np.eye(8, dtype=np.float32) * 0.01
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.last_bbox = bbox
        self.cls = None
        self.conf = 0.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.get_state()

    def update(self, bbox, cls=None, conf=0.0):
        # bbox: [x1,y1,x2,y2]
        x1, y1, x2, y2 = bbox
        z = np.array([[(x1+x2)/2.0, (y1+y2)/2.0, max(1.0, x2-x1), max(1.0, y2-y1)]], dtype=np.float32).T
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        I = np.eye(8, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.last_bbox = bbox
        if cls is not None:
            self.cls = cls
        self.conf = float(conf)

    def get_state(self):
        cx, cy, w, h = self.x[0,0], self.x[1,0], self.x[2,0], self.x[3,0]
        x1 = cx - w/2.0
        y1 = cy - h/2.0
        x2 = cx + w/2.0
        y2 = cy + h/2.0
        return [float(x1), float(y1), float(x2), float(y2)]


def iou(bb_test, bb_gt):
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)
    inter = w * h
    area1 = max(0., (bb_test[2]-bb_test[0]) * (bb_test[3]-bb_test[1]))
    area2 = max(0., (bb_gt[2]-bb_gt[0]) * (bb_gt[3]-bb_gt[1]))
    union = area1 + area2 - inter + 1e-6
    return inter / union


class SortTracker:
    def __init__(self, max_age=15, min_hits=2, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, detections):
        """
        detections: list of dicts with keys: bbox [x1,y1,x2,y2], cls, conf
        Returns list of tracks: dicts with id, bbox, cls, conf
        """
        self.frame_count += 1
        # Predict existing trackers
        preds = []
        for t in self.trackers:
            preds.append(t.predict())
        
        det_bboxes = [d['bbox'] for d in detections]
        N = len(self.trackers)
        M = len(det_bboxes)
        
        if N == 0 and M == 0:
            return []
        
        # Cost matrix = 1 - IoU
        cost = np.ones((N, M), dtype=np.float32)
        for i, t in enumerate(self.trackers):
            for j, bb in enumerate(det_bboxes):
                cost[i, j] = 1.0 - iou(preds[i], bb)
        
        matched_idx = []
        if N > 0 and M > 0:
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if 1.0 - cost[r, c] >= self.iou_threshold:
                    matched_idx.append((r, c))
        
        matched_dets = set([c for _, c in matched_idx])
        matched_trks = set([r for r, _ in matched_idx])
        
        # Update matched trackers
        for r, c in matched_idx:
            d = detections[c]
            self.trackers[r].update(d['bbox'], d.get('cls'), d.get('conf', 0.0))
        
        # Create new trackers for unmatched detections
        for j in range(M):
            if j not in matched_dets:
                t = KalmanBoxTracker(det_bboxes[j])
                t.update(det_bboxes[j], detections[j].get('cls'), detections[j].get('conf', 0.0))
                self.trackers.append(t)
        
        # Remove dead trackers
        alive = []
        outputs = []
        for t in self.trackers:
            if t.time_since_update <= self.max_age:
                alive.append(t)
                # Output only if it has sufficient hits or in early frames
                if (t.hits >= self.min_hits) or (self.frame_count <= self.min_hits):
                    outputs.append({
                        'id': t.id,
                        'bbox': t.get_state(),
                        'cls': t.cls,
                        'conf': t.conf
                    })
        self.trackers = alive
        return outputs
