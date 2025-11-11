"""FastAPI application bridging the UI with the data pipeline."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, root_validator, validator

from .pipeline_runner import PipelineConfig, PipelineRunner

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.NE"]


class PipelineRequestModel(BaseModel):
    """Request payload for launching the pipeline."""

    categories: List[str] = Field(
        default_factory=lambda: DEFAULT_CATEGORIES.copy(),
        description="Selected arXiv categories.",
    )
    search_terms: Optional[str] = Field(
        None,
        description="Additional keyword filters appended to the query.",
    )
    start_date: Optional[str] = Field(
        None,
        description="Inclusive start date (YYYY-MM-DD).",
        regex=r"^\d{4}-\d{2}-\d{2}$",
    )
    end_date: Optional[str] = Field(
        None,
        description="Inclusive end date (YYYY-MM-DD).",
        regex=r"^\d{4}-\d{2}-\d{2}$",
    )
    max_results: int = Field(500, ge=1, le=2000)
    query: Optional[str] = Field(
        None,
        description="Full arXiv query string. Overrides other filters if provided.",
    )
    embed_mode: str = Field("abstracts", regex=r"^(abstracts|sections)$")
    steps: Optional[List[str]] = Field(
        None,
        description="Specific pipeline steps to execute (defaults to full run).",
    )

    @validator("categories", pre=True, always=True)
    def _clean_categories(cls, value: Optional[List[str]]) -> List[str]:
        if not value:
            return []
        return [item.strip() for item in value if item and item.strip()]

    @root_validator
    def _validate_dates(cls, values: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
        start = values.get("start_date")
        end = values.get("end_date")
        if start and end and start > end:
            raise ValueError("start_date must be earlier than or equal to end_date")
        return values


class JobResponseModel(BaseModel):
    job_id: str
    status: str
    current_step: Optional[str]
    completed_steps: int
    total_steps: int
    error: Optional[str]
    created_at: datetime
    updated_at: datetime
    logs: List[str]


app = FastAPI(title="EmbedCompare Pipeline API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

runner = PipelineRunner(REPO_ROOT)


def _serialise_job(job) -> Dict[str, object]:  # noqa: ANN001 - FastAPI response helper
    return {
        "job_id": job.job_id,
        "status": job.status,
        "current_step": job.current_step,
        "completed_steps": job.completed_steps,
        "total_steps": job.total_steps,
        "error": job.error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "logs": job.logs,
    }


@app.get("/api/pipeline/options")
def get_pipeline_options() -> JSONResponse:
    """Return UI metadata such as available categories and steps."""

    payload = {
        "default_categories": DEFAULT_CATEGORIES,
        "available_steps": runner.available_steps,
        "default_steps": list(PipelineRunner.DEFAULT_STEPS),
        "default_max_results": 500,
        "default_embed_mode": "abstracts",
    }
    return JSONResponse(payload)


@app.post("/api/pipeline/run", response_model=JobResponseModel)
def run_pipeline(request: PipelineRequestModel) -> JobResponseModel:
    """Launch the pipeline in the background."""

    config = PipelineConfig(
        categories=request.categories,
        search_terms=request.search_terms,
        start_date=request.start_date,
        end_date=request.end_date,
        max_results=request.max_results,
        query=request.query,
        embed_mode=request.embed_mode,
        steps=tuple(request.steps or PipelineRunner.DEFAULT_STEPS),
    )
    try:
        job = runner.start_job(config)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JobResponseModel(**_serialise_job(job))


@app.get("/api/jobs", response_model=List[JobResponseModel])
def list_jobs() -> List[JobResponseModel]:
    return [JobResponseModel(**_serialise_job(job)) for job in runner.list_jobs()]


@app.get("/api/jobs/{job_id}", response_model=JobResponseModel)
def get_job(job_id: str) -> JobResponseModel:
    job = runner.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponseModel(**_serialise_job(job))


# Serve the static visualization assets.
app.mount("/", StaticFiles(directory=str(REPO_ROOT), html=True), name="static")
