from __future__ import annotations

import runpy
import sys
from pathlib import Path

from .workflow_specs import WorkflowSpec


def run_script_path(script_path: Path | str, script_args: list[str] | tuple[str, ...] | None = None) -> None:
    resolved_path = Path(script_path)
    if not resolved_path.is_absolute():
        project_root = Path(__file__).resolve().parents[1]
        resolved_path = project_root / resolved_path
    resolved_path = resolved_path.resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Workflow script not found: {resolved_path}")

    original_argv = list(sys.argv)
    script_parent = str(resolved_path.parent)
    original_sys_path = list(sys.path)
    try:
        if script_parent not in sys.path:
            sys.path.insert(0, script_parent)
        configured_args = [str(arg) for arg in (script_args or [])]
        sys.argv = [str(resolved_path), *configured_args]
        runpy.run_path(str(resolved_path), run_name="__main__")
    finally:
        sys.argv = original_argv
        sys.path[:] = original_sys_path


def run_legacy_workflow(workflow: WorkflowSpec, script_args: list[str] | tuple[str, ...] | None = None) -> None:
    run_script_path(workflow.legacy_path, script_args=script_args)
