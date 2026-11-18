"""Process start-up shared by every node of the package."""

from __future__ import annotations

import signal

import rclpy
from rclpy.signals import SignalHandlerOptions


def _raise_interrupt(*_) -> None:
    raise KeyboardInterrupt


def init_ros(args=None) -> None:
    """``rclpy.init`` with orderly shutdown on SIGINT and SIGTERM.

    rclpy's own handler invalidates the context the moment a signal arrives, so
    a node could no longer publish its final zero-velocity command or finish
    writing a video/CSV file, and background threads crash mid-publish. Here
    both signals raise KeyboardInterrupt in the main thread instead; each
    node's ``finally`` block cleans up and then calls ``rclpy.try_shutdown()``.
    """
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    # Install SIGINT explicitly too: processes started in the background by a
    # non-interactive shell inherit SIGINT as *ignored*, and Python then never
    # installs its KeyboardInterrupt handler.
    signal.signal(signal.SIGINT, _raise_interrupt)
    signal.signal(signal.SIGTERM, _raise_interrupt)


def ignore_further_interrupts() -> None:
    """Call at the start of cleanup: ``ros2 launch`` may forward a second SIGINT."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
