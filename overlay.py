#!/usr/bin/env python3
"""
Realtime overlay utilities: consistent colors, drawing tracked boxes, labels, and info panels.
"""
import cv2
import random

_random = random.Random(42)

_color_cache = {}

def color_for_id(id_):
    if id_ in _color_cache:
        return _color_cache[id_]
    _random.seed(id_ * 9973)
    c = (_random.randint(64,255), _random.randint(64,255), _random.randint(64,255))
    _color_cache[id_] = (int(c[0]), int(c[1]), int(c[2]))
    return _color_cache[id_]


def draw_tracks(img, tracks, thickness=2):
    """
    tracks: list of {id, bbox [x1,y1,x2,y2], cls, conf}
    """
    for t in tracks:
        x1,y1,x2,y2 = [int(v) for v in t['bbox']]
        name = t.get('cls') or 'obj'
        conf = t.get('conf', 0.0)
        tid = t.get('id', -1)
        color = color_for_id(tid)
        cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness)
        label = f"{name}#{tid} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    return img


def draw_info_panel(img, stats):
    """
    stats: dict like {fps, counts: {cls: n}, active: k}
    draws small panel at top-left
    """
    panel = [
        f"FPS: {int(stats.get('fps',0))}",
        f"Active: {int(stats.get('active',0))}",
    ]
    y = 24
    for line in panel:
        cv2.putText(img, line, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        y += 22
    return img
