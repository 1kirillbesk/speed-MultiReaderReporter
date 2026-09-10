# speed_MultiReaderReporter/core/family.py
"""Per-cell-family config resolution.

One run can mix cells from different test families (e.g. SPEED_JGNE and
SPEED_LWHK) whose checkups use different procedure names, step ids and voltage
windows. The optional ``families`` list in the config carries those overrides;
anything a family omits falls back to the top-level value, so configs without a
``families`` key behave exactly as before.
"""
from __future__ import annotations
import copy
import logging
from typing import Any, Iterable

_LOG = logging.getLogger(__name__)

def as_list(value: Any) -> list[str]:
    """Accept either a single string or a list of strings from config."""
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [str(value)]
    if isinstance(value, Iterable):
        return [str(v) for v in value]
    return [str(value)]

def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out

def resolve_family_cfg(cfg: dict, cell: str) -> dict:
    """Return cfg with the first family whose 'match' hits `cell` merged on top."""
    families = (cfg or {}).get("families") or []
    if not families:
        return cfg

    cell_lower = (cell or "").lower()
    for family in families:
        if not isinstance(family, dict):
            continue
        needles = [n.lower() for n in as_list(family.get("match")) if n]
        if not needles or not any(n in cell_lower for n in needles):
            continue
        overrides = {k: v for k, v in family.items() if k not in ("match", "name")}
        merged = _deep_merge(copy.deepcopy(cfg), overrides)
        merged.pop("families", None)
        merged["_family"] = family.get("name", "?")
        return merged

    _LOG.warning("no family matched cell %s; falling back to top-level config", cell)
    return cfg
