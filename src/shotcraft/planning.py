"""Plan bounded windows without propagating target identity across shot cuts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Window:
    shot: int
    start: int
    stop: int
    padded_frames: int

    @property
    def valid_frames(self):
        return self.stop - self.start


def plan_windows(frame_count, cuts=(), capacity=49, overlap=9):
    """Use half-open ranges; pad the final window and crop padding after generation.

    This schedules independent backend calls. It does not implement or claim
    VideoPainter's upstream target-ID resampling algorithm.
    """
    if type(frame_count) is not int or frame_count < 1:
        raise ValueError("frame_count must be a positive integer")
    if type(capacity) is not int or capacity < 5 or (capacity - 1) % 4:
        raise ValueError("capacity must follow the video VAE 4k+1 convention")
    if type(overlap) is not int or not 0 <= overlap < capacity:
        raise ValueError("overlap must be smaller than capacity")
    cuts = tuple(cuts)
    if any(type(c) is not int or not 0 < c < frame_count for c in cuts):
        raise ValueError("cuts must be internal integer frame boundaries")
    if tuple(sorted(set(cuts))) != cuts:
        raise ValueError("cuts must be unique and strictly increasing")
    boundaries = (0, *cuts, frame_count)
    windows = []
    for shot, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
        start = begin
        while start < end:
            stop = min(start + capacity, end)
            windows.append(Window(shot, start, stop, capacity))
            if stop == end:
                break
            start = stop - overlap
    return windows
