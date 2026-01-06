"""ANSI color utilities for terminal output."""

import sys


class Colors:
    """ANSI color codes for terminal output."""

    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    WHITE = "\033[37m"
    RED = "\033[31m"
    RESET = "\033[0m"

    _enabled = True

    @classmethod
    def disable(cls):
        """Disable all color codes (for non-TTY output)."""
        cls._enabled = False
        cls.BOLD = ""
        cls.DIM = ""
        cls.CYAN = ""
        cls.GREEN = ""
        cls.YELLOW = ""
        cls.BLUE = ""
        cls.MAGENTA = ""
        cls.WHITE = ""
        cls.RED = ""
        cls.RESET = ""

    @classmethod
    def enable(cls):
        """Re-enable color codes."""
        if not cls._enabled:
            cls._enabled = True
            cls.BOLD = "\033[1m"
            cls.DIM = "\033[2m"
            cls.CYAN = "\033[36m"
            cls.GREEN = "\033[32m"
            cls.YELLOW = "\033[33m"
            cls.BLUE = "\033[34m"
            cls.MAGENTA = "\033[35m"
            cls.WHITE = "\033[37m"
            cls.RED = "\033[31m"
            cls.RESET = "\033[0m"

    @classmethod
    def auto_detect(cls):
        """Auto-detect TTY and disable colors if not a terminal."""
        if not sys.stdout.isatty():
            cls.disable()


# Auto-detect on import
Colors.auto_detect()
