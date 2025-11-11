"""Background execution utilities for the data pipeline."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration provided by the UI when launching the pipeline."""

    categories: Sequence[str]
    search_terms: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    max_results: int
    query: Optional[str]
    embed_mode: str
    steps: Sequence[str]


@dataclass
class PipelineJob:
    """State container for a single pipeline execution."""

    job_id: str
    config: PipelineConfig
    status: str = "pending"
    logs: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.utcnow())
    updated_at: datetime = field(default_factory=lambda: datetime.utcnow())
    current_step: Optional[str] = None
    completed_steps: int = 0
    total_steps: int = 0
    error: Optional[str] = None


@dataclass
class StepDefinition:
    """Metadata for a pipeline step."""

    key: str
    name: str

    def build_command(self, config: PipelineConfig) -> List[str]:
        raise NotImplementedError

    def build_env(self, config: PipelineConfig) -> Mapping[str, str]:
        return {}


class FetchStep(StepDefinition):
    def build_command(self, config: PipelineConfig) -> List[str]:
        cmd = [
            sys.executable,
            "data_scripts/1_fetch_papers.py",
            "--max-results",
            str(config.max_results),
        ]
        categories = ",".join(sorted({c.strip() for c in config.categories if c.strip()}))
        if categories:
            cmd.extend(["--categories", categories])
        if config.search_terms:
            cmd.extend(["--search", config.search_terms])
        if config.start_date:
            cmd.extend(["--start-date", config.start_date])
        if config.end_date:
            cmd.extend(["--end-date", config.end_date])
        if config.query:
            cmd.extend(["--query", config.query])
        return cmd


class EmbedStep(StepDefinition):
    def build_command(self, config: PipelineConfig) -> List[str]:
        return [sys.executable, "data_scripts/2_embed_papers.py"]

    def build_env(self, config: PipelineConfig) -> Mapping[str, str]:
        return {"EMBED_MODE": config.embed_mode}


class SimpleScriptStep(StepDefinition):
    """A step backed by a single Python script without customisation."""

    def __init__(self, key: str, name: str, script_path: str) -> None:
        super().__init__(key=key, name=name)
        self._script_path = script_path

    def build_command(self, config: PipelineConfig) -> List[str]:  # noqa: ARG002
        return [sys.executable, self._script_path]


class PipelineRunner:
    """Executes the data pipeline in a background thread."""

    DEFAULT_STEPS: Sequence[str] = (
        "fetch",
        "embed",
        "analyze",
        "run_analysis",
        "deep_analysis",
    )

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self._jobs: Dict[str, PipelineJob] = {}
        self._lock = threading.Lock()
        self._active_job: Optional[str] = None
        self._step_registry: Dict[str, StepDefinition] = {
            "fetch": FetchStep(key="fetch", name="Fetch latest papers"),
            "embed": EmbedStep(key="embed", name="Generate embeddings"),
            "analyze": SimpleScriptStep(
                key="analyze",
                name="Build hybrid layouts",
                script_path="data_scripts/3_analyze_embeddings.py",
            ),
            "run_analysis": SimpleScriptStep(
                key="run_analysis",
                name="Compute clustering metrics",
                script_path="data_scripts/4_run_analysis.py",
            ),
            "deep_analysis": SimpleScriptStep(
                key="deep_analysis",
                name="Derive deep-dive diagnostics",
                script_path="data_scripts/5_deep_analysis.py",
            ),
        }

    # ------------------------------------------------------------------
    # Job lifecycle helpers
    # ------------------------------------------------------------------
    def start_job(self, config: PipelineConfig) -> PipelineJob:
        steps = self._resolve_steps(config.steps)
        if not steps:
            raise ValueError("No pipeline steps were selected.")

        job_id = uuid.uuid4().hex
        job = PipelineJob(job_id=job_id, config=config, total_steps=len(steps))

        with self._lock:
            if self._active_job and self._jobs[self._active_job].status in {"pending", "running"}:
                raise RuntimeError("Another pipeline run is already in progress.")
            self._jobs[job_id] = job
            self._active_job = job_id

        thread = threading.Thread(target=self._run_job, args=(job_id, steps), daemon=True)
        thread.start()
        return job

    def get_job(self, job_id: str) -> Optional[PipelineJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> List[PipelineJob]:
        with self._lock:
            return list(self._jobs.values())

    @property
    def available_steps(self) -> Dict[str, str]:
        return {key: step.name for key, step in self._step_registry.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_steps(self, keys: Sequence[str]) -> List[StepDefinition]:
        resolved: List[StepDefinition] = []
        seen = set()
        for key in keys or self.DEFAULT_STEPS:
            if key in seen:
                continue
            step = self._step_registry.get(key)
            if not step:
                raise ValueError(f"Unknown pipeline step '{key}'.")
            resolved.append(step)
            seen.add(key)
        return resolved

    def _append_log(self, job: PipelineJob, message: str) -> None:
        with self._lock:
            job.logs.append(message)
            job.updated_at = datetime.utcnow()

    def _run_job(self, job_id: str, steps: Sequence[StepDefinition]) -> None:
        job = self.get_job(job_id)
        if job is None:
            return

        job.status = "running"
        self._append_log(job, "Pipeline execution started.")

        try:
            for step_index, step in enumerate(steps, start=1):
                job.current_step = step.name
                self._append_log(job, f"→ {step.name}")
                self._execute_step(job, step)
                job.completed_steps = step_index
                self._append_log(job, f"✓ Completed {step.name}")

            job.status = "completed"
            job.current_step = None
            self._append_log(job, "Pipeline finished successfully.")
        except Exception as exc:  # pragma: no cover - background thread
            job.status = "failed"
            job.error = str(exc)
            self._append_log(job, f"✗ {exc}")
        finally:
            with self._lock:
                if self._active_job == job_id:
                    self._active_job = None

    def _execute_step(self, job: PipelineJob, step: StepDefinition) -> None:
        command = step.build_command(job.config)
        env = os.environ.copy()
        env.update(step.build_env(job.config))

        process = subprocess.Popen(  # noqa: PLW1510 - stream logs in real time
            command,
            cwd=str(self.repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )

        assert process.stdout is not None  # narrow type for mypy
        for line in iter(process.stdout.readline, ""):
            cleaned = line.rstrip()
            if cleaned:
                self._append_log(job, cleaned)
        exit_code = process.wait()
        if exit_code != 0:
            raise RuntimeError(f"Step '{step.name}' failed with exit code {exit_code}.")


__all__ = ["PipelineRunner", "PipelineConfig", "PipelineJob"]
