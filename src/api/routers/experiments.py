"""Experiment CRUD endpoints."""

from fastapi import APIRouter, HTTPException

from src.api.schemas import ExperimentCreate, ExperimentResponse
from src.experiment.manager import ExperimentManager

router = APIRouter()


@router.post("", response_model=ExperimentResponse, status_code=201)
def create_experiment(body: ExperimentCreate) -> ExperimentResponse:
    with ExperimentManager() as mgr:
        try:
            exp = mgr.create_experiment(body.name, body.description, body.task_type)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        return ExperimentResponse.model_validate(exp)


@router.get("", response_model=list[ExperimentResponse])
def list_experiments() -> list[ExperimentResponse]:
    with ExperimentManager() as mgr:
        exps = mgr.list_experiments()
        return [ExperimentResponse.model_validate(e) for e in exps]


@router.get("/{name}", response_model=ExperimentResponse)
def get_experiment(name: str) -> ExperimentResponse:
    with ExperimentManager() as mgr:
        exp = mgr.get_experiment(name)
        if exp is None:
            for e in mgr.list_experiments():
                if str(e.experiment_id).startswith(name):
                    exp = e
                    break
        if exp is None:
            raise HTTPException(status_code=404, detail=f"Experiment not found: {name}")
        return ExperimentResponse.model_validate(exp)


@router.delete("/{name}", status_code=204)
def delete_experiment(name: str) -> None:
    with ExperimentManager() as mgr:
        exp = mgr.get_experiment(name)
        if exp is None:
            raise HTTPException(status_code=404, detail=f"Experiment not found: {name}")
        mgr.session.delete(exp)
        mgr.session.commit()
