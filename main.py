from pathlib import Path

import typer
import uvicorn
from dotenv import load_dotenv

if (Path(__file__).parent / ".env").exists():
    load_dotenv()

cli = typer.Typer()


@cli.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind address"),
    port: int = typer.Option(8040, help="Bind port"),
    migrate: bool = typer.Option(False, "--migrate/--no-migrate", help="Run alembic upgrade head on startup"),
    sync_apifox: bool = typer.Option(
        False, "--sync-apifox/--no-sync-apifox", help="Push OpenAPI schema to Apifox on startup"
    ),
    reload: bool = typer.Option(False, "--reload/--no-reload", help="Enable hot reload (development only)"),
):
    """Start the News2ETF-Engine API server."""
    import src.api.app as app_module

    app_module.RUN_MIGRATE = migrate
    app_module.SYNC_APIFOX = sync_apifox

    uvicorn.run("src.api.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    cli()
