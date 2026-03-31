"""Industry dictionary endpoints."""

from fastapi import APIRouter, HTTPException
from rich.console import Console

from src.api.schemas import IndustryBuildRequest, IndustryBuildResponse
from src.industry import DEFAULT_MODEL, build_industry_dict
from src.utils.llm_client import get_llm_client

router = APIRouter()

_console = Console()


@router.post("/build-dict", response_model=IndustryBuildResponse)
def build_dict(body: IndustryBuildRequest = IndustryBuildRequest()) -> IndustryBuildResponse:
    """Build hierarchical ETF industry dictionary via ZhipuAI.

    Requires the ZHIPU_API_KEY environment variable to be set.
    This operation calls the LLM and may take a while.
    """
    try:
        client = get_llm_client(body.model or DEFAULT_MODEL)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        build_industry_dict(client=client, console=_console, model=body.model or DEFAULT_MODEL)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return IndustryBuildResponse(status="success", message="Industry dictionary built successfully.")
