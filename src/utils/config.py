"""Helpers to layer YAML configs and CLI arguments correctly.

The convention used across this project:

  * YAML config supplies defaults for hyperparameters
  * CLI flags override the YAML

This matches the de-facto behavior expected from a research codebase: a YAML
preset captures a known-good configuration, and the CLI exists so you can
nudge one or two hyperparameters at the command line without editing the
preset (e.g. ``--max-steps 100`` for a smoke test).

There are two entry points:

  * ``apply_yaml_defaults(parser, argv, section=...)`` — call BEFORE
    ``parser.parse_args(argv)``. Pre-scans ``argv`` for ``--config <path>``,
    loads the YAML, and sets the values from the requested section as
    defaults on the parser. The subsequent ``parse_args`` will then layer
    CLI flags on top of those defaults.

  * ``apply_config_overrides(args, config_path, section=...)`` — legacy
    behavior kept for the original ``project3`` code paths. New scripts in
    ``jazz_piano`` should use ``apply_yaml_defaults`` instead.
"""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Sequence

import yaml


def _read_yaml_section(config_path: str | Path, section: str | None) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file {config_path} not found")
    data: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    section_data = data.get(section, data) if section else data
    if not isinstance(section_data, dict):
        raise ValueError(f"Config section {section!r} must be a mapping")
    return section_data


def _peek_config_path(argv: Sequence[str] | None) -> str | None:
    """Find ``--config <path>`` (or ``--config=path``) in argv without consuming it."""
    if argv is None:
        import sys
        argv = sys.argv[1:]
    it = iter(argv)
    for tok in it:
        if tok == "--config":
            return next(it, None)
        if tok.startswith("--config="):
            return tok.split("=", 1)[1]
    return None


def apply_yaml_defaults(
    parser: ArgumentParser,
    argv: Sequence[str] | None = None,
    section: str | None = None,
) -> None:
    """Pre-load YAML values into the parser as defaults so CLI flags win.

    Call this BEFORE ``parser.parse_args(argv)``. Idempotent.
    """
    config_path = _peek_config_path(argv)
    if not config_path:
        return
    section_data = _read_yaml_section(config_path, section)
    # Only set defaults for arguments that the parser actually knows about,
    # so a typo in YAML raises later in parse rather than silently passing.
    known = {action.dest for action in parser._actions}
    unknown = [k for k in section_data if k not in known]
    if unknown:
        raise ValueError(
            f"YAML keys not recognized by parser (section={section!r}): {unknown}"
        )
    parser.set_defaults(**section_data)


def apply_config_overrides(args: Namespace, config_path: str | None, section: str | None = None) -> Namespace:
    """Legacy overrider — YAML wins over CLI. Kept for backward compatibility
    with the original project3 scripts. New code should prefer
    ``apply_yaml_defaults`` instead.
    """
    if not config_path:
        return args
    section_data = _read_yaml_section(config_path, section)
    for key, value in section_data.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


__all__ = ["apply_yaml_defaults", "apply_config_overrides"]
