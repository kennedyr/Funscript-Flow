#!/usr/bin/env python3

import concurrent.futures
import gc
import json
import math
import os
import threading
import time
from datetime import UTC, datetime
from itertools import repeat
from queue import Empty, Queue

import cv2
import numpy as np

from funscript_flow.async_video_reader import AsyncVideoReader
from funscript_flow.localization import STRINGS
from funscript_flow.video_reader_cv import VideoReaderCV

# ---------- Constants ----------
SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".m4v",
    ".webm",
    ".wmv",
    ".flv",
    ".mpg",
    ".mpeg",
    ".ts",
}
SUPPORTED_VIDEO_PATTERNS = " ".join(
    f"*{ext}" for ext in sorted(SUPPORTED_VIDEO_EXTENSIONS)
)


# ---------- GPU Backend Detection ----------
def get_available_backends() -> dict[str, bool]:
    """Detect available GPU acceleration backends."""
    backends = {
        "CPU": True,  # Always available
        "CUDA": False,
        "OpenCL": False,
        "DNN": False,
    }

    # Check for CUDA support
    try:
        if hasattr(cv2.cuda, "getCudaEnabledDeviceCount"):
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            backends["CUDA"] = device_count > 0
    except Exception:
        pass

    # Check for OpenCL support
    try:
        if hasattr(cv2.ocl, "haveOpenCL"):
            backends["OpenCL"] = cv2.ocl.haveOpenCL()
    except Exception:
        pass

    # Check for DNN backend support
    try:
        if hasattr(cv2, "dnn"):
            backends["DNN"] = True
    except Exception:
        pass

    return backends


# def _get_gpu_info() -> str:
#     """Get information about available GPUs and backends."""
#     info = []
#     backends = get_available_backends()

#     # CUDA info with device details
#     try:
#         if backends["CUDA"] and hasattr(cv2.cuda, "getCudaEnabledDeviceCount"):
#             count = cv2.cuda.getCudaEnabledDeviceCount()
#             if count > 0:
#                 device_names = []
#                 for i in range(count):
#                     try:
#                         name = cv2.cuda.getDeviceName(i)
#                         device_names.append(name)
#                     except Exception:
#                         device_names.append(f"Device {i}")
#                 if len(device_names) == 1:
#                     info.append(f"CUDA: {device_names[0]}")
#                 else:
#                     info.append(f"CUDA: {len(device_names)} devices")
#     except Exception:
#         pass

#     # OpenCL info (simplified)
#     if backends["OpenCL"]:
#         info.append("OpenCL supported")

#     # DNN info
#     if backends["DNN"]:
#         info.append("DNN module available")

#     if not info:
#         return "CPU only"

#     return " | ".join(info)


# def _detect_cut(pair, log_func=None, threshold=30):
#     return False
#     prev_frame, curr_frame = pair
#     diff = np.mean(
#         np.abs(curr_frame.astype(np.float32) - prev_frame.astype(np.float32))
#     )
#     if log_func != None and diff > threshold:
#         log_func("Found a cut at " + str(diff))

#     return diff > threshold


# ---------- Original flow functions (for VR mode) ------------
# def _compute_flow(pair):
#     """
#     (VR Mode) Process a pair of consecutive 512x512 grayscale frames (cropped from left half).
#     Computes optical flow on two regions (middle-center and bottom-center of a 3x3 grid)
#     and returns a tuple (avg_flow_middle, avg_flow_bottom).
#     """
#     prev_frame, curr_frame = pair
#     h, w = prev_frame.shape  # expected 512x512
#     cell_h = h // 3
#     cell_w = w // 3
#     prev_middle = prev_frame[cell_h : 2 * cell_h, cell_w : 2 * cell_w]
#     curr_middle = curr_frame[cell_h : 2 * cell_h, cell_w : 2 * cell_w]
#     prev_bottom = prev_frame[2 * cell_h : 3 * cell_h, cell_w : 2 * cell_w]
#     curr_bottom = curr_frame[2 * cell_h : 3 * cell_h, cell_w : 2 * cell_w]
#     flow_middle = cv2.calcOpticalFlowFarneback(
#         prev_middle,
#         curr_middle,
#         None,
#         pyr_scale=0.5,
#         levels=3,
#         winsize=15,
#         iterations=3,
#         poly_n=5,
#         poly_sigma=1.2,
#         flags=0,
#     )
#     flow_bottom = cv2.calcOpticalFlowFarneback(
#         prev_bottom,
#         curr_bottom,
#         None,
#         pyr_scale=0.5,
#         levels=3,
#         winsize=15,
#         iterations=3,
#         poly_n=5,
#         poly_sigma=1.2,
#         flags=0,
#     )
#     avg_middle = np.mean(flow_middle[..., 1])
#     avg_bottom = np.mean(flow_bottom[..., 1])
#     return avg_middle, avg_bottom


# def _compute_flow_nonvr_invert(pair):
#     prev_frame, curr_frame = pair
#     flow = cv2.calcOpticalFlowFarneback(
#         prev_frame,
#         curr_frame,
#         None,
#         pyr_scale=0.5,
#         levels=3,
#         winsize=15,
#         iterations=3,
#         poly_n=5,
#         poly_sigma=1.2,
#         flags=0,
#     )
#     avg_flow = np.mean(-flow[..., 0] + flow[..., 1])
#     return avg_flow


# def _compute_flow_nonvr(pair):
#     prev_frame, curr_frame = pair
#     flow = cv2.calcOpticalFlowFarneback(
#         prev_frame,
#         curr_frame,
#         None,
#         pyr_scale=0.5,
#         levels=3,
#         winsize=15,
#         iterations=3,
#         poly_n=5,
#         poly_sigma=1.2,
#         flags=0,
#     )
#     avg_flow = np.mean(flow[..., 0] + flow[..., 1])
#     return avg_flow


# --- PSO mode ---


# def _center_of_mass_variance(flow, num_cells=32):
#     """
#     Splits the optical flow into a configurable grid (num_cells x num_cells),
#     computes the variance of the optical flow in each grid cell,
#     and returns the center of mass of the variance.
#     """
#     h, w, _ = flow.shape
#     grid_h, grid_w = h // num_cells, w // num_cells

#     variance_grid = np.zeros((num_cells, num_cells))
#     y_coords, x_coords = np.meshgrid(
#         np.arange(num_cells), np.arange(num_cells), indexing="ij"
#     )

#     for i in range(num_cells):
#         for j in range(num_cells):
#             cell = flow[i * grid_h : (i + 1) * grid_h, j * grid_w : (j + 1) * grid_w]
#             magnitude = np.sqrt(cell[..., 0] ** 2 + cell[..., 1] ** 2)
#             variance_grid[i, j] = np.var(magnitude)

#     total_variance = np.sum(variance_grid)

#     if total_variance == 0:
#         return (w // 2, h // 2)  # Default to image center if no variance
#     else:
#         center_x = (
#             np.sum(x_coords * variance_grid) * grid_w / total_variance + grid_w / 2
#         )
#         center_y = (
#             np.sum(y_coords * variance_grid) * grid_h / total_variance + grid_h / 2
#         )
#         return (center_x, center_y)


def max_divergence(flow):
    """
    Computes the divergence of the optical flow over the whole image and returns
    the pixel (x, y) with the highest absolute divergence along with its value.
    """
    # No grid, just pure per-pixel divergence!
    div = np.gradient(flow[..., 0], axis=0) + np.gradient(flow[..., 1], axis=1)

    # Get the index (y, x) of the max abs divergence
    y, x = np.unravel_index(np.argmax(np.abs(div)), div.shape)
    return x, y, div[y, x]


def radial_motion_weighted(flow, center, is_cut, pov_mode=False, balance_global=True):
    """
    Computes signed radial motion: positive for outward motion, negative for inward motion.
    Closer pixels have higher weight.
    """
    if is_cut:
        return 0.0
    h, w, _ = flow.shape
    y, x = np.indices((h, w))
    dx = x - center[0]
    dy = y - center[1]

    dot = flow[..., 0] * dx + flow[..., 1] * dy

    # In POV mode, just return the mean dot product
    if pov_mode or not balance_global:
        return np.mean(dot)

    # Cancel out global motion by balancing the averages
    # multiply products to the right of the center (w-x) / w and to the left by x / w
    weighted_dot = np.where(x > center[0], dot * (w - x) / w, dot * x / w)
    # multiply products below the center (h-y) / h and above by y / h
    weighted_dot = np.where(
        y > center[1], weighted_dot * (h - y) / h, weighted_dot * y / h
    )

    return np.mean(weighted_dot)


# def _largest_cluster_center(positions, threshold=10.0):
#     """
#     BFS to find the largest cluster of swarm positions, return its centroid.
#     """
#     num_particles = len(positions)
#     adj = [[] for _ in range(num_particles)]
#     for i in range(num_particles):
#         for j in range(i + 1, num_particles):
#             if np.linalg.norm(positions[i] - positions[j]) < threshold:
#                 adj[i].append(j)
#                 adj[j].append(i)

#     visited = set()
#     clusters = []

#     def bfs(start):
#         queue, c = [start], []
#         while queue:
#             node = queue.pop()
#             if node in visited:
#                 continue
#             visited.add(node)
#             c.append(node)
#             for nei in adj[node]:
#                 if nei not in visited:
#                     queue.append(nei)
#         return c

#     for i in range(num_particles):
#         if i not in visited:
#             group = bfs(i)
#             clusters.append(group)

#     biggest = max(clusters, key=len)
#     return (np.mean(positions[biggest], axis=0), len(biggest))


# def _swarm_positions(flow, num_particles=30, iterations=50):
#     """
#     Moves 'num_particles' along 'flow' for 'iterations'. Return final positions for clustering.
#     """
#     h, w, _ = flow.shape
#     positions = np.column_stack(
#         [np.random.uniform(0, w, num_particles), np.random.uniform(0, h, num_particles)]
#     )
#     for _ in range(iterations):
#         for i in range(num_particles):
#             x_i = int(np.clip(positions[i, 0], 0, w - 1))
#             y_i = int(np.clip(positions[i, 1], 0, h - 1))
#             vx = flow[y_i, x_i, 1]
#             vy = flow[y_i, x_i, 0]
#             positions[i, 0] = np.clip(positions[i, 0] + vx, 0, w - 1)
#             positions[i, 1] = np.clip(positions[i, 1] + vy, 0, h - 1)
#     return positions


def precompute_flow_info(p0, p1, config):
    """
    Concurrency-friendly function:
      - compute Farneback flow
      - swarm normal flow => pos_center => val_pos
      - swarm negative flow => neg_center => val_neg
      - detect cut
      - pick a center for cut jumps (could just use pos_center)
    Returns a dict with everything needed for final pass.
    """
    # Try to use the selected backend
    backend = config.get("backend", "CPU")

    if backend == "CUDA":
        try:
            return precompute_flow_info_gpu(p0, p1, config.get("cut_threshold", 7))
        except Exception:
            # Fall back to CPU if GPU fails
            pass
    elif backend == "OpenCL":
        try:
            return precompute_flow_info_opencl(p0, p1, config)
        except Exception:
            # Fall back to CPU if OpenCL fails
            pass
    elif backend == "DNN":
        try:
            return precompute_flow_info_dnn(p0, p1, config)
        except Exception:
            # Fall back to CPU if DNN fails
            pass

    # CPU implementation (default)
    cut_threshold = config.get("cut_threshold", 7)

    flow_start = time.perf_counter()
    flow = cv2.calcOpticalFlowFarneback(p0, p1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_seconds = time.perf_counter() - flow_start

    center_start = time.perf_counter()
    if config.get("pov_mode"):
        # In pov mode, just use the center of the bottom edge of the frame
        max = (p0.shape[1] // 2, p0.shape[0] - 1, 0)
    else:
        max = max_divergence(flow)
    pos_center = max[0:2]
    val_pos = max[2]
    center_seconds = time.perf_counter() - center_start

    # Detect cut based on flow map
    cut_start = time.perf_counter()
    mag = cv2.magnitude(flow[..., 0], flow[..., 1])
    mean_mag = np.mean(mag)
    if mean_mag > cut_threshold:
        is_cut = True
    else:
        is_cut = False
    cut_seconds = time.perf_counter() - cut_start

    cut_center = pos_center[0]

    return {
        "flow": flow,
        "pos_center": pos_center,
        "neg_center": pos_center,
        "val_pos": val_pos,
        "val_neg": val_pos,
        "cut": is_cut,
        "cut_center": cut_center,
        "mean_mag": mean_mag,
        "timings": {
            "farneback": flow_seconds,
            "center": center_seconds,
            "cut_detection": cut_seconds,
        },
    }


def precompute_flow_info_opencl(p0, p1, config):
    """OpenCL-accelerated optical flow computation using UMat."""
    cut_threshold = config.get("cut_threshold", 7)

    # Convert to UMat for GPU processing
    gpu_p0 = cv2.UMat(p0)
    gpu_p1 = cv2.UMat(p1)

    # Compute optical flow on GPU
    flow_gpu = cv2.calcOpticalFlowFarneback(
        gpu_p0, gpu_p1, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    # Get flow back to CPU
    flow = flow_gpu.get()

    # Compute divergence on CPU (could be optimized further)
    if config.get("pov_mode"):
        max_val = (p0.shape[1] // 2, p0.shape[0] - 1, 0)
    else:
        max_val = max_divergence(flow)
    pos_center = max_val[0:2]
    val_pos = max_val[2]

    # Calculate magnitude
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mean_mag = np.mean(mag)
    is_cut = mean_mag > cut_threshold

    return {
        "flow": flow,
        "pos_center": pos_center,
        "neg_center": pos_center,
        "val_pos": val_pos,
        "val_neg": val_pos,
        "cut": is_cut,
        "cut_center": pos_center[0],
        "mean_mag": mean_mag,
    }


def precompute_flow_info_dnn(p0, p1, config):
    """DNN-accelerated optical flow using DisFlow algorithm."""
    cut_threshold = config.get("cut_threshold", 7)

    # Use DIS optical flow (faster than Farneback)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)

    # Compute flow
    flow = dis.calc(p0, p1, None)

    # Rest is same as CPU version
    if config.get("pov_mode"):
        max_val = (p0.shape[1] // 2, p0.shape[0] - 1, 0)
    else:
        max_val = max_divergence(flow)
    pos_center = max_val[0:2]
    val_pos = max_val[2]

    # Calculate magnitude
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mean_mag = np.mean(mag)
    is_cut = mean_mag > cut_threshold

    return {
        "flow": flow,
        "pos_center": pos_center,
        "neg_center": pos_center,
        "val_pos": val_pos,
        "val_neg": val_pos,
        "cut": is_cut,
        "cut_center": pos_center[0],
        "mean_mag": mean_mag,
    }


def precompute_flow_info_gpu(p0, p1, cut_threshold):
    # Upload frames to GPU—time to let your GPU do the heavy lifting!
    gpu_p0 = cv2.cuda_GpuMat()
    gpu_p1 = cv2.cuda_GpuMat()
    gpu_p0.upload(p0)
    gpu_p1.upload(p1)

    # Compute optical flow on the GPU
    fb = cv2.cuda_FarnebackOpticalFlow.create(0.5, 3, 15, 3, 5, 1.2, 0)
    flow_gpu = fb.calc(gpu_p0, gpu_p1, None)

    # Grab flow back to CPU for any CPU-based ops (like max_divergence)
    flow = flow_gpu.download()
    max_val = max_divergence(flow)
    pos_center = max_val[:2]
    val_pos = max_val[2]

    # Calculate magnitude on GPU (splitting channels)
    channels = cv2.cuda.split(flow_gpu)
    mag_gpu = cv2.cuda.magnitude(channels[0], channels[1])
    mag = mag_gpu.download()
    mean_mag = float(np.mean(mag))
    is_cut = mean_mag > cut_threshold

    cut_center = pos_center[0]

    return {
        "flow": flow,
        "pos_center": pos_center,
        "neg_center": pos_center,
        "val_pos": val_pos,
        "val_neg": val_pos,
        "cut": is_cut,
        "cut_center": cut_center,
        "mean_mag": mean_mag,
    }


def precompute_wrapper(p, params):
    return precompute_flow_info(p[0], p[1], params)


def radial_motion_wrapper(args):
    return radial_motion_weighted(*args)


# def _fetch_frames(video_path, chunk, params):
#     frames_gray = []
#     try:
#         # Use AsyncVideoReader for better performance
#         vr = AsyncVideoReader(
#             video_path,
#             num_threads=min(params["threads"], 4),  # Limit decoder threads
#             width=512 if params.get("vr_mode") else 256,
#             height=512 if params.get("vr_mode") else 256,
#             buffer_size=min(len(chunk), 100),  # Adaptive buffer size
#         )
#         batch_frames = vr.get_batch_parallel(chunk)  # Use parallel decoding
#     except Exception:
#         return frames_gray
#     vr = None
#     gc.collect()

#     for f in batch_frames:
#         if params.get("vr_mode"):
#             h, w, _ = f.shape
#             gray = cv2.cvtColor(f[h // 2 :, : w // 2], cv2.COLOR_RGB2GRAY)
#         else:
#             gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
#         frames_gray.append(gray)

#     return frames_gray


def fetch_frames_optimized(
    video_path: str, chunk: list[int], params: dict
) -> list[np.ndarray]:
    """Optimized frame fetching with direct grayscale conversion."""
    frames_gray = []

    try:
        # Don't resize if we're going to crop anyway in VR mode
        target_size = None if params.get("vr_mode") else (256, 256)

        vr = AsyncVideoReader(
            video_path,
            num_threads=min(params["threads"], 4),
            width=target_size[0] if target_size else None,
            height=target_size[1] if target_size else None,
            buffer_size=min(len(chunk), 100),
        )

        # Process in smaller sub-batches for better memory efficiency
        sub_batch_size = 50
        for i in range(0, len(chunk), sub_batch_size):
            sub_chunk = chunk[i : i + sub_batch_size]
            batch_frames = vr.get_batch_parallel(sub_chunk)

            for f in batch_frames:
                if params.get("vr_mode"):
                    # For VR mode, resize then crop and convert
                    f_resized = cv2.resize(f, (512, 512))
                    h, w = 512, 512
                    # Direct grayscale conversion of the cropped region
                    gray = cv2.cvtColor(
                        f_resized[h // 2 :, : w // 2], cv2.COLOR_RGB2GRAY
                    )
                else:
                    # Direct grayscale conversion
                    gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
                frames_gray.append(gray)

    except Exception:
        return frames_gray
    finally:
        vr = None
        gc.collect()

    return frames_gray


# ---------- Main Processing Function ----------
def process_video(
    video_path,
    params,
    log_func,
    progress_callback=None,
    cancel_flag=None,
    preview_callback=None,
):
    """
    Example usage that:
      1) Reads frames in bracketed intervals
      2) For each consecutive pair, runs precompute_flow_info() in multiple threads
      3) After concurrency finishes, applies inertia in a single-thread pass
      4) In a final concurrency step, computes radial_flow with the final center
         (and sign) for each pair.
    """
    start_time = time.time()
    error_occurred = False
    base, _ = os.path.splitext(video_path)
    output_path = base + ".funscript"
    if os.path.exists(output_path) and not params["overwrite"]:
        log_func(f"Skipping: output file exists ({output_path})")
        return error_occurred

    # Attempt to open video
    try:
        log_func(f"Processing video: {video_path}")
        vr = VideoReaderCV(
            video_path, width=1024, height=1024, num_threads=params["threads"]
        )
    except Exception as e:
        log_func(f"ERROR: Unable to open video at {video_path}: {e}")
        return True

    # Basic video properties
    try:
        total_frames = len(vr)
        fps = vr.get_avg_fps()
    except Exception as e:
        log_func(f"ERROR: Unable to read video properties: {e}")
        return True

    step = max(1, math.ceil(fps / 30.0))
    effective_fps = fps / step
    indices = list(range(0, total_frames, step))
    log_func(
        f"FPS: {fps:.2f}; downsampled to ~{effective_fps:.2f} fps; {len(indices)} frames selected."
    )
    log_func(f"Using backend: {params.get('backend', 'CPU')}")
    bracket_size = int(params.get("batch_size", 3000.0))
    norm_win = int(params["norm_window"] * effective_fps)
    if norm_win % 2 == 0:
        norm_win += 1
    half_norm = norm_win // 2

    center = None
    final_flow_list = []

    # Use a queue for prefetching instead of global variable
    prefetch_queue = Queue(maxsize=1)
    fetch_thread = None
    # We'll collect color frames for preview
    # (and grayscale for computing flow).
    final_con_list = []

    # Threads avoid serializing frame pairs and flow arrays between processes.
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=params["threads"]
    ) as thread_pool:
        for chunk_start in range(0, len(indices), bracket_size):
            if cancel_flag and cancel_flag():
                log_func("User bailed.")
                return error_occurred

            chunk = indices[chunk_start : chunk_start + bracket_size]
            frame_indices = chunk[:-1]
            if len(chunk) < 2:
                continue

            fetch_start = time.perf_counter()
            # Get frames from prefetch queue or fetch directly
            frames_gray = None
            if fetch_thread and fetch_thread.is_alive():
                try:
                    frames_gray = prefetch_queue.get(timeout=30.0)
                except Empty:
                    log_func(f"WARNING: Prefetch timeout for chunk {chunk_start}")

            if frames_gray is None:
                frames_gray = fetch_frames_optimized(video_path, chunk, params)
            log_func(
                f"Chunk {chunk_start}: frame fetching took "
                f"{time.perf_counter() - fetch_start:.2f}s."
            )
            if not frames_gray:
                log_func(
                    f"ERROR: Unable to fetch frames for chunk {chunk_start} - skipping."
                )
                continue

            # Start prefetching next batch
            if chunk_start + bracket_size < len(indices):
                next_chunk = indices[
                    chunk_start + bracket_size : chunk_start + 2 * bracket_size
                ]

                def prefetch_worker(path, chunk_data, params_data, queue):
                    try:
                        frames = fetch_frames_optimized(path, chunk_data, params_data)
                        queue.put(frames)
                    except Exception:
                        queue.put([])  # Put empty list on error

                fetch_thread = threading.Thread(
                    target=prefetch_worker,
                    args=(video_path, next_chunk, params, prefetch_queue),
                )
                fetch_thread.start()

            # Build consecutive pairs
            pairs = list(zip(frames_gray[:-1], frames_gray[1:]))

            del frames_gray

            precompute_start = time.perf_counter()
            precomputed = list(
                thread_pool.map(precompute_wrapper, pairs, repeat(params))
            )
            log_func(
                f"Chunk {chunk_start}: precompute thread map took "
                f"{time.perf_counter() - precompute_start:.2f}s."
            )
            stage_timings = {
                stage: sum(info["timings"][stage] for info in precomputed)
                for stage in ("farneback", "center", "cut_detection")
            }
            log_func(
                f"Chunk {chunk_start}: precompute summed worker time "
                f"(Farneback {stage_timings['farneback']:.2f}s, "
                f"center/divergence {stage_timings['center']:.2f}s, "
                f"cut detection {stage_timings['cut_detection']:.2f}s; "
                "summed across frame pairs, not wall time)."
            )
            del pairs

            # import matplotlib.pyplot as plt
            # mean_mags = [abs(info["val_pos"]) for info in precomputed]
            # plt.plot(mean_mags)
            # plt.show()
            # Add values of val_pos to final_con_list for preview
            final_con_list.extend([abs(info["val_pos"] * 10) for info in precomputed])

            # 2) Single-thread pass to calculate centers based on the median center of the surrounding second, discarding outliers
            final_centers = []
            final_cuts = []
            for j, info in enumerate(precomputed):
                # Use the mean center of the 6 frames in each direction, discarding outliers
                center_list = [info["pos_center"]]
                for i in range(1, 7):
                    if j - i >= 0:
                        center_list.append(precomputed[j - i]["pos_center"])
                    if j + i < len(precomputed):
                        center_list.append(precomputed[j + i]["pos_center"])
                center_list = np.array(center_list)
                # Discard outliers from center list
                center = np.mean(center_list, axis=0)
                final_centers.append(center)

                if info["cut"]:
                    if j >= half_norm:
                        window = [
                            info["mean_mag"]
                            for info in precomputed[j - half_norm : j + half_norm + 1]
                        ]
                        window_max = max(window)
                        is_local_max = info["mean_mag"] >= window_max
                        print(f"Index {j} (val={info['mean_mag']} cut={is_local_max})")
                        final_cuts.append(is_local_max)
                    else:
                        final_cuts.append(True)
                else:
                    final_cuts.append(False)

                # Progress update during center computation
                if progress_callback and j % 24 == 0:  # Update every 24 frames
                    frames_processed = (
                        chunk_start + j * 0.5
                    )  # Center computation is ~50% of processing
                    prog = min(100, int(100 * frames_processed / len(indices)))
                    progress_callback(prog)

                # Show preview with the new center on the "next" color frame
                # e.g. color_pairs[j][1] is the "current" next frame
                # preview_frame = pairs[j][1].copy()
                # cv2.circle(preview_frame, (int(center[0]), int(center[1])), 6, (0,255,0), -1)
                # cv2.imshow("preview", preview_frame)
                # cv2.waitKey(30)

            # 3) Concurrency to compute final weighted dot products with actual final center

            dot_args = [
                (
                    info["flow"],
                    final_centers[j],
                    final_cuts[j],
                    params.get("pov_mode", False),
                    params.get("balance_global", True),
                )
                for j, info in enumerate(precomputed)
            ]
            radial_start = time.perf_counter()
            dot_vals = list(thread_pool.map(radial_motion_wrapper, dot_args))
            log_func(
                f"Chunk {chunk_start}: radial-motion thread map took "
                f"{time.perf_counter() - radial_start:.2f}s."
            )

            for j, dot_val in enumerate(dot_vals):
                # flow_val = dot_val * signs[j]
                is_cut = final_cuts[j]
                final_flow_list.append((dot_val, is_cut, frame_indices[j]))

                # More frequent progress updates within chunk processing
                if progress_callback and j % 24 == 0:  # Update every 24 frames
                    frames_processed = chunk_start + j + 1
                    prog = min(100, int(100 * frames_processed / len(indices)))
                    progress_callback(prog)

            # Explicitly release heavy flow arrays before next chunk
            del precomputed, dot_args, dot_vals

            # Final progress update for this chunk
            if progress_callback:
                prog = min(100, int(100 * (chunk_start + len(chunk)) / len(indices)))
                progress_callback(prog)

    # Clean up prefetch thread if still running
    if fetch_thread and fetch_thread.is_alive():
        fetch_thread.join(timeout=5.0)

    # Clear the prefetch queue
    while not prefetch_queue.empty():
        try:
            prefetch_queue.get_nowait()
        except Empty:
            break

    post_processing_start = time.perf_counter()
    # --- Piecewise Integration and Timestamping
    cum_flow = [0]
    time_stamps = [final_flow_list[0][2]]
    cut_timestamps = []

    for i in range(1, len(final_flow_list)):
        flow_prev, _cut_prev, _t_prev = final_flow_list[i - 1]
        flow_curr, cut_curr, t_curr = final_flow_list[i]

        if cut_curr:
            cum_flow.append(0)
            cut_timestamps.append(t_curr)
        else:
            # Midpoint integration to reduce phase lag
            mid_flow = (flow_prev + flow_curr) / 2
            cum_flow.append(cum_flow[-1] + mid_flow)

        time_stamps.append(t_curr)

    # Optional: Shift the result back by half a time step to correct for residual phase offset
    cum_flow = [
        (cum_flow[i] + cum_flow[i - 1]) / 2 if i > 0 else cum_flow[i]
        for i in range(len(cum_flow))
    ]

    # --- Detrending & Normalization ---
    detrend_win = int(params["detrend_window"] * effective_fps)
    disc_threshold = (
        1000  # float(params.get("discontinuity_threshold", 0.1))  # tweak as needed
    )

    detrended_data = np.zeros_like(cum_flow)
    weight_sum = np.zeros_like(cum_flow)

    # Find indices where a jump occurs
    disc_indices = np.where(np.abs(np.diff(cum_flow)) > disc_threshold)[0] + 1
    # Break data into continuous segments
    segment_boundaries = [0] + list(disc_indices) + [len(cum_flow)]

    overlap = detrend_win // 2

    for i in range(len(segment_boundaries) - 1):
        seg_start = segment_boundaries[i]
        seg_end = segment_boundaries[i + 1]
        seg_length = seg_end - seg_start

        # Just subtract the average  segments with too few points to converge with polypit
        if seg_length < 5:
            detrended_data[seg_start:seg_end] = cum_flow[seg_start:seg_end] - np.mean(
                cum_flow[seg_start:seg_end]
            )
            continue
        if seg_length <= detrend_win:
            # If segment is too short, process it in one go
            segment = cum_flow[seg_start:seg_end]
            x = np.arange(len(segment))
            trend = np.polyfit(x, segment, 1)
            detrended_segment = segment - np.polyval(trend, x)
            weights = np.hanning(len(segment))
            detrended_data[seg_start:seg_end] += detrended_segment * weights
            weight_sum[seg_start:seg_end] += weights
        else:
            # Process long segments in overlapping windows
            for start in range(seg_start, seg_end - overlap, overlap):
                end = min(start + detrend_win, seg_end)
                segment = cum_flow[start:end]
                x = np.arange(len(segment))
                trend = np.polyfit(x, segment, 1)
                detrended_segment = segment - np.polyval(trend, x)
                weights = np.hanning(len(segment))
                detrended_data[start:end] += detrended_segment * weights
                weight_sum[start:end] += weights

    # Normalize by weight sum to blend overlapping windows
    detrended_data /= np.maximum(weight_sum, 1e-6)

    smoothed_data = np.convolve(
        detrended_data, [1 / 16, 1 / 4, 3 / 8, 1 / 4, 1 / 16], mode="same"
    )
    # Normalize each window in normalization_window to 0-100
    norm_rolling = np.empty_like(smoothed_data)
    for i in range(len(smoothed_data)):
        start_idx = max(0, i - half_norm)
        end_idx = min(len(smoothed_data), i + half_norm + 1)
        local_window = smoothed_data[start_idx:end_idx]
        local_min = local_window.min()
        local_max = local_window.max()
        if local_max - local_min == 0:
            norm_rolling[i] = 50
        else:
            norm_rolling[i] = (
                (smoothed_data[i] - local_min) / (local_max - local_min) * 100
            )

    # Sine fit (aborted experiment, left here for reference)
    # sine = sine_fit(norm_rolling)

    # Plot sine against norm_rolling
    # import matplotlib.pyplot as plt
    # plt.plot(norm_rolling)
    # #Smooth final con
    # smoothed_final_con = np.convolve(final_con_list, [1/16, 1/4, 3/8, 1/4, 1/16], mode='same')
    # plt.plot(smoothed_final_con)
    # plt.show()

    # TEST: Raw data
    # norm_rolling = sine

    # 3. Keyframe Reduction. Just use slope inversions for now.
    if params["keyframe_reduction"]:
        key_indices = [0]
        for i in range(1, len(norm_rolling) - 1):
            d1 = norm_rolling[i] - norm_rolling[i - 1]
            d2 = norm_rolling[i + 1] - norm_rolling[i]

            if (d1 < 0) != (d2 < 0):
                key_indices.append(i)
        key_indices.append(len(norm_rolling) - 1)
    else:
        key_indices = range(len(norm_rolling))
    actions = []
    for ki in key_indices:
        try:
            timestamp_ms = int(((time_stamps[ki]) / fps) * 1000)
            pos = round(norm_rolling[ki])
            actions.append({"at": timestamp_ms, "pos": 100 - pos})
        except Exception as e:
            log_func(f"Error computing action at segment index {ki}: {e}")
            error_occurred = True
    final_actions = actions

    log_func(f"Keyframe reduction: {len(final_actions)} actions computed.")
    actions = final_actions

    bookmarks = []
    prev_cut = None
    for cut in cut_timestamps:
        try:
            timestamp_ms = int(((cut) / fps) * 1000)
            if prev_cut is not None and timestamp_ms - prev_cut < 1000:
                log_func(f"ignoring duplicate cut at {timestamp_ms}")
                continue  # Skip duplicate cut timestamps
            timestamp_s = timestamp_ms / 1000.0
            bookmarks.append(
                {
                    "name": "",
                    "time": datetime.fromtimestamp(timestamp_s, tz=UTC).strftime(
                        "%H:%M:%S.%f"
                    )[:-3],
                }
            )
            prev_cut = timestamp_ms
        except Exception as e:
            log_func(f"Error computing bookmark at timestamp {cut}: {e}")
            error_occurred = True

    funscript = {
        "version": "1.0",
        "actions": actions,
        "metadata": {
            "bookmarks": bookmarks,
        },
    }
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(funscript, f, indent=2)
        log_func(STRINGS["funscript_saved"].format(output_path=output_path))
    except Exception as e:
        log_func(STRINGS["log_error"].format(error=str(e)))
        error_occurred = True
    log_func(
        f"Post-processing took {time.perf_counter() - post_processing_start:.2f}s."
    )
    # Log processing time
    elapsed_time = time.time() - start_time
    log_func(f"Processing time: {elapsed_time:.2f} seconds")

    return error_occurred


# ---------- Sine Fit ------------


# def _sine_fit(data, error_threshold=5000.0, gain=2.0, min_points=3, max_points=30):
#     """
#     Fits half-wave sine segments (center=50) to `data` by testing candidate endpoints
#     from min_points to max_points ahead. After segmentation, if two consecutive segments
#     have the same sign, we split them with an inserted corrective half-wave (with inverted amplitude)
#     to help catch missed alternations.

#     Returns the fitted array.
#     """
#     n = len(data)
#     segments = []  # each segment is a dict with {'start', 'end', 'A'}
#     start = 0

#     # --- First pass: Segment the data ---
#     while start < n - 1:
#         best_err = np.inf
#         best_end = None
#         best_A = 0.0

#         for seg_len in range(min_points, max_points + 1):
#             end = start + seg_len
#             if end >= n:
#                 break

#             T = seg_len  # segment length (points between endpoints)
#             x = np.arange(T + 1)
#             model = np.sin(np.pi * x / T)
#             segment = data[start : end + 1]
#             denom = np.sum(model**2)
#             if denom == 0:
#                 continue
#             # Linear LS solution for amplitude A.
#             A = np.sum(model * (segment - 50)) / denom
#             fit = 50 + A * model
#             err = np.sqrt(np.mean((segment - fit) ** 2))

#             if err < best_err:
#                 best_err = err
#                 best_end = end
#                 best_A = A

#         if best_end is None:
#             break

#         # Error correction: if error too high, flatten the segment.
#         if best_err > error_threshold:
#             best_A = 0.0
#         # Boost low amplitude segments (because sometimes they're just shy).
#         # best_A = np.sign(best_A) * (abs(best_A) ** (1.0 / gain))

#         segments.append({"start": start, "end": best_end, "A": best_A})
#         start = best_end

#     # --- Second pass: Correction for consecutive segments with the same sign ---
#     corrected_segments = []
#     i = 0
#     while i < len(segments):
#         # If the next segment exists and both segments have nonzero, same-signed amplitude...
#         if (
#             i < len(segments) - 1
#             and segments[i]["A"] != 0
#             and segments[i + 1]["A"] != 0
#             and np.sign(segments[i]["A"]) == np.sign(segments[i + 1]["A"])
#         ):
#             combined_start = segments[i]["start"]
#             combined_end = segments[i + 1]["end"]
#             if (combined_end - combined_start) >= min_points * 2:
#                 L = combined_end - combined_start
#                 # Split the combined region into three parts.
#                 mid1 = combined_start + L // 3
#                 mid2 = combined_start + 2 * L // 3

#                 # Re-fit first sub-segment.
#                 T1 = mid1 - combined_start
#                 if T1 < 2:
#                     T1 = 2
#                     mid1 = combined_start + T1
#                 x1 = np.arange(T1 + 1)
#                 model1 = np.sin(np.pi * x1 / T1)
#                 seg1 = data[combined_start : mid1 + 1]
#                 denom1 = np.sum(model1**2)
#                 A1 = np.sum(model1 * (seg1 - 50)) / denom1 if denom1 != 0 else 0

#                 # Re-fit third sub-segment.
#                 T3 = combined_end - mid2
#                 if T3 < 2:
#                     T3 = 2
#                     mid2 = combined_end - T3
#                 x3 = np.arange(T3 + 1)
#                 model3 = np.sin(np.pi * x3 / T3)
#                 seg3 = data[mid2 : combined_end + 1]
#                 denom3 = np.sum(model3**2)
#                 A3 = np.sum(model3 * (seg3 - 50)) / denom3 if denom3 != 0 else 0

#                 # Corrective (middle) segment: force amplitude opposite in sign.
#                 A2 = -np.sign(segments[i]["A"]) * (0.5 * (abs(A1) + abs(A3)))

#                 corrected_segments.append(
#                     {"start": combined_start, "end": mid1, "A": A1}
#                 )
#                 corrected_segments.append({"start": mid1, "end": mid2, "A": A2})
#                 corrected_segments.append({"start": mid2, "end": combined_end, "A": A3})
#                 i += 2  # skip the next segment; we've merged it
#                 continue
#             else:
#                 # Comvine them into one segment
#                 combined_A = segments[i]["A"] + segments[i + 1]["A"]
#                 combined_start = segments[i]["start"]
#                 combined_end = segments[i + 1]["end"]
#                 corrected_segments.append(
#                     {"start": combined_start, "end": combined_end, "A": combined_A}
#                 )
#                 i += 2
#                 continue

#         corrected_segments.append(segments[i])
#         i += 1

#     # --- Third pass: Detect and fix missed periods ---
#     final_segments = []
#     for j in range(len(corrected_segments)):
#         if j > 0 and j < len(corrected_segments) - 1:
#             prev_L = (
#                 corrected_segments[j - 1]["end"] - corrected_segments[j - 1]["start"]
#             )
#             curr_L = corrected_segments[j]["end"] - corrected_segments[j]["start"]
#             next_L = (
#                 corrected_segments[j + 1]["end"] - corrected_segments[j + 1]["start"]
#             )

#             if curr_L > prev_L + next_L:
#                 # Split into a number of segments depending on the calculated number of missed periods
#                 missed_periods = round(curr_L / (prev_L + next_L))

#                 segment_splits = np.linspace(
#                     corrected_segments[j]["start"],
#                     corrected_segments[j]["end"],
#                     2 * missed_periods + 1,
#                     dtype=int,
#                 )
#                 invert = False
#                 for split_idx in range(len(segment_splits) - 1):
#                     split_segment = {
#                         "start": segment_splits[split_idx],
#                         "end": segment_splits[split_idx + 1],
#                         "A": corrected_segments[j]["A"] * (-1 if invert else 1),
#                     }
#                     invert = not invert
#                     final_segments.append(split_segment)
#                 continue
#         final_segments.append(corrected_segments[j])
#     # plot the rolling variance of segment lengths, with outliers flagged
#     segment_lengths = [seg["end"] - seg["start"] for seg in final_segments]
#     # Calculate the rolling variance of segment lengths in a window of 5 segments
#     rolling_var = np.full(len(segment_lengths), np.nan)
#     for i in range(2, len(segment_lengths) - 2):
#         rolling_var[i] = np.var(segment_lengths[i - 2 : i + 3])
#     # Flag outliers (variance > 1.5 * median variance)
#     var_threshold = 1.5 * np.nanmedian(rolling_var)
#     for i in range(len(rolling_var)):
#         if rolling_var[i] > var_threshold:
#             final_segments[i]["outlier"] = True

#     # plt.figure(figsize=(10, 5))
#     # plt.plot(rolling_var, label='Variance', marker='o')
#     # plt.axhline(y=np.mean(rolling_var), color='r', linestyle='--', label='Mean Segment Length')
#     # plt.axhline(y=var_threshold, color='g', linestyle=':', label='Variance Threshold', xmin=0, xmax=len(segment_lengths)-1)
#     # plt.title('Rolling Variance of Segment Lengths with Outliers Flagged')
#     # plt.xlabel('Segment Index')
#     # plt.ylabel('Length')
#     # plt.legend()
#     # plt.show()

#     # --- Build the fitted curve from the corrected segments ---
#     fitted = np.full(n, 50.0)
#     for seg in final_segments:
#         s, e = seg["start"], seg["end"]
#         T = e - s
#         if T < 1:
#             continue
#         x_seg = np.arange(T + 1)
#         fitted[s : e + 1] = 50 + seg["A"] * np.sin(np.pi * x_seg / T)

#     return fitted


# ---------- Preview Helper ----------
# def _convert_frame_to_photo(frame):
#     try:
#         rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
#         retval, buffer = cv2.imencode(".png", rgb)
#         if not retval:
#             return None
#         img_data = buffer.tobytes()
#         return tk.PhotoImage(data=img_data)
#     except Exception:
#         return None


# ---------- Headless Mode ----------
def run_headless(input_path, settings):
    log_filename = "run.log"
    try:
        logf = open(log_filename, "w", encoding="utf-8")
    except Exception as e:
        print(f"Error opening log file: {e}")
        return

    def log_func(msg):
        logf.write(msg + "\n")
        logf.flush()
        print(msg)

    if os.path.isdir(input_path):
        files = []
        for root, dirs, files_in in os.walk(input_path):
            for f in files_in:
                ext = os.path.splitext(f)[1].lower()
                if ext in SUPPORTED_VIDEO_EXTENSIONS:
                    files.append(os.path.join(root, f))
    else:
        files = [input_path]
    if not files:
        print("No video files found.")
        logf.write("No video files found.\n")
        logf.close()
        return
    total_files = len(files)
    log_func(STRINGS["found_files"].format(n=total_files))
    for idx, video in enumerate(files):
        log_func(
            STRINGS["processing_file"].format(
                current=idx + 1, total=total_files, video_path=video
            )
        )
        process_video(
            video,
            settings,
            log_func,
            progress_callback=lambda prog: print(f"Video progress: {prog}%"),
        )
    log_func(STRINGS["batch_processing_complete"])
    logf.close()
    print("Done. See run.log for details.")
