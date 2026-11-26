"""Logging for the command-line entry points.

Library code logs through ``logging.getLogger(__name__)`` and never configures
handlers: a program importing clipmaker decides where messages go. Each CLI
``main`` calls :func:`setup_logging` once, so progress and warnings reach stderr
and stdout carries only what the command was asked to produce (a plan, a list
of hits, a report path), which stays safe to pipe or redirect.
"""
from __future__ import annotations

import argparse
import logging
import sys

_HANDLER_NAME = "clipmaker-cli"


class _CliFormatter(logging.Formatter):
    """Plain messages for INFO; a level prefix for warnings and errors; with
    --verbose, the logger name on every line to show which module spoke."""

    def __init__(self, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if self.verbose:
            message = f"{record.levelname:<7} {record.name}: {message}"
        elif record.levelno >= logging.WARNING:
            message = f"{record.levelname}: {message}"
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
        return message


class _StderrHandler(logging.StreamHandler):
    """Writes to whatever sys.stderr is at emit time, not the one at setup: a
    replaced stream (pytest capture, a redirecting caller) is never written
    after it was closed."""

    def __init__(self) -> None:
        super().__init__(sys.stderr)

    @property
    def stream(self):
        return sys.stderr

    @stream.setter
    def stream(self, value) -> None:
        pass


def add_verbose_flag(parser: argparse.ArgumentParser) -> None:
    """Add ``-v/--verbose`` (debug messages) to a command's parser."""
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show debug messages and the module each line comes from")


def setup_logging(verbose: bool = False) -> None:
    """Send clipmaker's log records to stderr; DEBUG with verbose, else INFO.

    Only the ``clipmaker`` logger is configured, so --verbose does not turn on
    the debug chatter of transformers or urllib3. Calling it again (tests run
    several mains in one process) replaces the handler instead of stacking one.
    """
    logger = logging.getLogger("clipmaker")
    for handler in list(logger.handlers):
        if handler.get_name() == _HANDLER_NAME:
            logger.removeHandler(handler)
    handler = _StderrHandler()
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(_CliFormatter(verbose))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
