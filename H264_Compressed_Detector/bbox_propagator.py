import os
import sys
import numpy as np


# -----------------------------------------------------------------------
# HELPER: Check if an MV block overlaps with the bounding box.
#
# COORDINATE CONVENTION:
#   src_x, src_y = TOP-LEFT corner of the macroblock (mb_x*16, mb_y*16)
#   blk_w, blk_h = 16, 16
#   The block occupies pixels [src_x, src_x+blk_w) x [src_y, src_y+blk_h)
#
# bbox = (x1, y1, x2, y2) in pixels - the bbox in the PREVIOUS frame
# -----------------------------------------------------------------------
def block_overlaps_bbox(src_x, src_y, blk_w, blk_h, bbox):
    x1, y1, x2, y2 = bbox

    # Full block rectangle (top-left to bottom-right)
    bx1 = src_x
    by1 = src_y
    bx2 = src_x + blk_w
    by2 = src_y + blk_h

    # Standard axis-aligned rectangle overlap test
    return bx1 < x2 and bx2 > x1 and by1 < y2 and by2 > y1


# -----------------------------------------------------------------------
# CORE: Propagate ONE bounding box using the motion vectors of a P-frame
#
# Arguments:
#   prev_bbox : (x1, y1, x2, y2) in pixels - the bbox from the previous frame
#   mv_array  : structured numpy array from p_frame_extractor
#               fields: source, w, h, src_x, src_y, dst_x, dst_y, flags,
#                       motion_x, motion_y, motion_scale
#
# Returns:
#   new_bbox  : (x1, y1, x2, y2) in pixels - the shifted bbox for the current frame
#
# Strategy:
#   1. Collect all MV blocks whose source rectangle overlaps the bbox.
#   2. Compute pixel displacement = motion_{x,y} / motion_scale for each.
#   3. Prefer blocks with |displacement| > 0.5 px (real motion, not background
#      zero-MVs that would drive the mean to zero).
#   4. If no such blocks exist, fall back to all overlapping blocks.
#   5. If no blocks overlap at all, keep bbox frozen (person not moved / occluded).
# -----------------------------------------------------------------------
def propagate_bbox(prev_bbox, mv_array):
    x1, y1, x2, y2 = prev_bbox

    # If no motion vectors are present, keep the box where it was
    if mv_array is None or len(mv_array) == 0:
        return prev_bbox

    # Collect pixel displacements for blocks overlapping the bbox
    all_dx = []
    all_dy = []

    for mv in mv_array:
        src_x        = int(mv['src_x'])
        src_y        = int(mv['src_y'])
        blk_w        = int(mv['w'])
        blk_h        = int(mv['h'])
        motion_x     = int(mv['motion_x'])
        motion_y     = int(mv['motion_y'])
        motion_scale = int(mv['motion_scale'])

        if not block_overlaps_bbox(src_x, src_y, blk_w, blk_h, prev_bbox):
            continue
        if motion_scale == 0:
            continue

        dx = motion_x / motion_scale   # pixel displacement (sub-pixel precision)
        dy = motion_y / motion_scale
        all_dx.append(dx)
        all_dy.append(dy)

    if not all_dx:
        # No MV blocks overlap the bbox at all — keep frozen
        return prev_bbox

    # --- Filter: remove exactly zero-motion blocks (static background)
    # The moving object will have non-zero MVs. We collect only those.
    moving_dx = [dx for dx, dy in zip(all_dx, all_dy) if dx != 0 or dy != 0]
    moving_dy = [dy for dx, dy in zip(all_dx, all_dy) if dx != 0 or dy != 0]

    if moving_dx:
        # Use the median of moving blocks to resist single-block noisy outliers
        avg_dx = float(np.median(moving_dx))
        avg_dy = float(np.median(moving_dy))
    else:
        # All blocks inside the bbox are static (0,0) — keep box frozen
        return prev_bbox

    # H.264 motion vectors point BACKWARDS in time (current frame -> reference frame)
    # To propagate an object's position FORWARD in time, we must subtract the vector.
    new_x1 = x1 - avg_dx
    new_y1 = y1 - avg_dy
    new_x2 = x2 - avg_dx
    new_y2 = y2 - avg_dy

    return (new_x1, new_y1, new_x2, new_y2)


# -----------------------------------------------------------------------
# HELPER: Convert YOLO format (cx, cy, w, h) normalized to pixel bbox
# -----------------------------------------------------------------------
def yolo_to_pixel_bbox(cx_n, cy_n, w_n, h_n, frame_width, frame_height):
    cx = cx_n * frame_width
    cy = cy_n * frame_height
    w  = w_n  * frame_width
    h  = h_n  * frame_height

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return (x1, y1, x2, y2)


# -----------------------------------------------------------------------
# HELPER: Convert pixel bbox back to YOLO format (normalized cx, cy, w, h)
# -----------------------------------------------------------------------
def pixel_bbox_to_yolo(bbox, frame_width, frame_height):
    x1, y1, x2, y2 = bbox

    cx_n = ((x1 + x2) / 2) / frame_width
    cy_n = ((y1 + y2) / 2) / frame_height
    w_n  = (x2 - x1) / frame_width
    h_n  = (y2 - y1) / frame_height

    return (cx_n, cy_n, w_n, h_n)


# -----------------------------------------------------------------------
# DEMO: Propagate bboxes across all P-frames for a given video's MV folder
#
# Arguments:
#   mv_dir         : folder containing the P-frame .npy MV files
#   initial_bboxes : list of (x1, y1, x2, y2) pixel bboxes from the I-frame
#   frame_width    : width of the video in pixels
#   frame_height   : height of the video in pixels
#
# The function walks through the MV files in PTS order and propagates all
# bboxes frame by frame.
#
# Returns:
#   results : dict mapping PTS -> list of (x1, y1, x2, y2) bboxes
# -----------------------------------------------------------------------
def propagate_all_frames(mv_dir, initial_bboxes, frame_width, frame_height):
    # Get all MV npy files sorted by PTS (filename)
    npy_files = sorted(
        [f for f in os.listdir(mv_dir) if f.endswith('.npy')],
        key=lambda f: int(os.path.splitext(f)[0])
    )

    if not npy_files:
        print(f"  [bbox_propagator] No MV files found in: {mv_dir}")
        return {}

    # Start with the initial bboxes from the I-frame
    current_bboxes = list(initial_bboxes)
    results = {}

    for npy_file in npy_files:
        pts = int(os.path.splitext(npy_file)[0])
        mv_path = os.path.join(mv_dir, npy_file)

        mv_array = np.load(mv_path, allow_pickle=True)

        # Propagate each bbox independently
        new_bboxes = []
        for prev_bbox in current_bboxes:
            new_bbox = propagate_bbox(prev_bbox, mv_array)
            new_bboxes.append(new_bbox)

        results[pts] = new_bboxes
        current_bboxes = new_bboxes   # Use the newly propagated bboxes for the next frame

        print(f"  [bbox_propagator] PTS {pts}: propagated {len(new_bboxes)} bbox(es)")

    return results


# -----------------------------------------------------------------------
# Quick test / standalone demo
# -----------------------------------------------------------------------
if __name__ == "__main__":
    # Example: test with Meet1_MVS folder
    # Suppose we detected one person at this bbox in the I-frame
    # (using a 1920x1080 video, bbox in pixel coords)
    TEST_MV_DIR     = "Meet1_MVS"
    TEST_FRAME_W    = 1920
    TEST_FRAME_H    = 1080

    # A sample initial bbox from the I-frame (x1, y1, x2, y2) in pixels
    # Just a dummy example - in the real pipeline this comes from the detection model
    sample_initial_bboxes = [
        (800, 300, 1000, 900),   # Person 1
    ]

    print(f"Starting propagation from: {TEST_MV_DIR}")
    print(f"Initial bboxes: {sample_initial_bboxes}\n")

    propagated = propagate_all_frames(
        mv_dir         = TEST_MV_DIR,
        initial_bboxes = sample_initial_bboxes,
        frame_width    = TEST_FRAME_W,
        frame_height   = TEST_FRAME_H
    )

    print("\n=== Propagation Results ===")
    for pts, bboxes in propagated.items():
        print(f"  PTS {pts}: {bboxes}")
