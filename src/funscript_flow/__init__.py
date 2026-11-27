import argparse
import os
import signal
import sys

from PySide6.QtCore import QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from funscript_flow.app import App
from funscript_flow.funscript_flow import run_headless


def main() -> None:
    parser = argparse.ArgumentParser(description="Optical Flow to Funscript")
    parser.add_argument("input", nargs="?", help="Input video file or folder")
    parser.add_argument(
        "--threads", type=int, default=8, help="Number of threads (default: 8)"
    )
    parser.add_argument(
        "--detrend_window",
        type=float,
        default=2.0,
        help="Detrend window in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--norm_window",
        type=float,
        default=3.0,
        help="Normalization window in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=3000,
        help="Batch size in frames (default: 3000)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output files"
    )
    parser.add_argument(
        "--vr_mode",
        action="store_true",
        help="Enable VR Mode (if not set, non‑VR mode is used)",
    )
    parser.add_argument(
        "--pov_mode",
        action="store_true",
        help="Enable POV Mode (improves stability for POV videos)",
    )
    parser.add_argument(
        "--balance_global",
        action="store_true",
        help="Balance Global Motion (cancel out camera motion)",
    )
    parser.add_argument(
        "--disable_keyframe_reduction",
        action="store_false",
        help="Disable keyframe reduction",
    )
    parser.add_argument(
        "--backend",
        choices=["CPU", "CUDA", "OpenCL", "DNN"],
        default="CPU",
        help="Processing backend (default: CPU)",
    )
    args = parser.parse_args()
    settings = {
        "threads": args.threads,
        "detrend_window": args.detrend_window,
        "norm_window": args.norm_window,
        "batch_size": args.batch_size,
        "overwrite": args.overwrite,
        "vr_mode": args.vr_mode,
        "pov_mode": args.pov_mode,
        "balance_global": args.balance_global,
        "keyframe_reduction": not args.disable_keyframe_reduction,
        "backend": args.backend,
        "cut_threshold": 7,
    }
    if args.input:
        run_headless(args.input, settings)
    else:
        print("starting app")
        app = QApplication(sys.argv)

        # Set application properties
        app.setApplicationName("Funscript Flow")
        app.setApplicationDisplayName("Funscript Flow")
        app.setOrganizationName("Funscript Flow")

        # Set application icon
        script_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(script_dir, "icon.png")
        if os.path.exists(icon_path):
            app.setWindowIcon(QIcon(icon_path))

        # Install signal handler for graceful shutdown
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        # Create timer to allow Python to process signals
        timer = QTimer()
        timer.timeout.connect(lambda: None)
        timer.start(100)  # Process signals every 100ms

        window = App()
        window.show()
        sys.exit(app.exec())
