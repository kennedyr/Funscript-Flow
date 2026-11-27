#!/usr/bin/env python3

import cv2
import numpy as np


# ---------- OpenCV VideoReader wrapper to replace decord ----------
class VideoReaderCV:
    """OpenCV VideoReader wrapper to replace decord."""

    def __init__(self, video_path, width=None, height=None, num_threads=None):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise Exception(f"Cannot open video: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = width
        self.height = height

        # Get original dimensions
        self.orig_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __len__(self):
        return self.total_frames

    def get_avg_fps(self):
        return self.fps

    def get_batch(self, indices):
        frames: list[np.ndarray] = []
        for idx in indices:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize if needed
                if self.width is not None and self.height is not None:
                    frame = cv2.resize(frame, (self.width, self.height))
                frames.append(frame)
            else:
                # If frame reading fails, create a black frame
                if self.width is not None and self.height is not None:
                    frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                else:
                    frame = np.zeros(
                        (self.orig_height, self.orig_width, 3), dtype=np.uint8
                    )
                frames.append(frame)

        # Return as numpy array to mimic decord's asnumpy() behavior
        return np.array(frames)

    def __del__(self):
        if hasattr(self, "cap"):
            self.cap.release()
