import typer

app = typer.Typer(
    help="Main application for command-line interface.",
    pretty_exceptions_show_locals=False,
)


def main() -> None:
    from src.cli import industry, labeling

    app.add_typer(industry.app, name="industry", help="Build industry classification dictionary")
    app.add_typer(labeling.app, name="labeling", help="Label index names with industry categories")

    app()
