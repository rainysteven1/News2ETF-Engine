"""Config read-only endpoints."""

from fastapi import APIRouter, HTTPException

from src.api.schemas import ConfigResponse
from src.experiment.manager import ExperimentManager

router = APIRouter()


@router.get("", response_model=list[ConfigResponse])
def list_configs(type: str | None = None) -> list[ConfigResponse]:
    with ExperimentManager() as mgr:
        configs = mgr.list_configs(task_type=type)
        return [ConfigResponse.model_validate(c) for c in configs]


@router.get("/{config_id}", response_model=ConfigResponse)
def get_config(config_id: int) -> ConfigResponse:
    with ExperimentManager() as mgr:
        cfg = mgr.get_config(config_id)
        if cfg is None:
            raise HTTPException(status_code=404, detail=f"Config not found: {config_id}")
        return ConfigResponse.model_validate(cfg)
