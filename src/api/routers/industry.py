"""Industry dictionary endpoints."""

import os

from fastapi import APIRouter, HTTPException
from rich.console import Console

from src.api.schemas import IndustryBuildRequest, IndustryBuildResponse
from src.industry import DEFAULT_MODEL, build_industry_dict

router = APIRouter()

_console = Console()


@router.post("/build-dict", response_model=IndustryBuildResponse)
def build_dict(body: IndustryBuildRequest = IndustryBuildRequest()) -> IndustryBuildResponse:
    """Build hierarchical ETF industry dictionary via ZhipuAI.

    Requires the ZHIPU_API_KEY environment variable to be set.
    This operation calls the LLM and may take a while.
    """
    api_key = os.environ.get("ZHIPU_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="Environment variable ZHIPU_API_KEY is not set.")

    try:
        build_industry_dict(api_key=api_key, console=_console, model=body.model)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return IndustryBuildResponse(status="success", message="Industry dictionary built successfully.")
