"""Process a folder of videos in parallel on the available GPUs (or CPU cores).

The ``if __name__ == "__main__"`` guard is required on Windows and macOS, where
worker processes are started with ``spawn``.
"""

import sys

import tachymetry as tm

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "videos/"
    report = tm.track_many(folder, out="runs/batch", devices="auto", render=False)
    print(report)
    for job in report.failed:
        print(job["video"], job["error"].splitlines()[-1])
