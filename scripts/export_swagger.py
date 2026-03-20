import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.app import create_app


def export_swagger():
    app = create_app()
    openapi_schema = app.openapi()

    with open("news2etf_swagger.json", "w", encoding="utf-8") as f:
        json.dump(openapi_schema, f, indent=2, ensure_ascii=False)

    print("Swagger JSON 已成功导出到 news2etf_swagger.json")


if __name__ == "__main__":
    export_swagger()
