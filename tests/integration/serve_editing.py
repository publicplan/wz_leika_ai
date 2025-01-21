from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI


def wz_additions() -> List[Dict[str, Any]]:
    """Example data for wz editing system."""
    correct1 = {
        "code": "01.12.0",
        "name": "Test",
        "section_name": "Test",
        "explanation": "Test",
        "keywords": ["Test"]
    }
    correct2 = {
        "code": "01.11.0",
        "keywords": ["keyword1", "keyword2", "keyword3"]
    }

    malformed = {
        "code": "abc",
        "name": "Test",
        "section_name": "Test",
        "explanation": "Test",
        "keywords": ["Test"]
    }
    return [correct1, correct2, malformed]


# pylint: disable=unused-variable
def _build_app() -> FastAPI:
    app = FastAPI()

    @app.get("/editing")
    def editing():
        """Serve sample editing data."""
        return wz_additions()

    return app


def _main():
    host = "0.0.0.0"
    port = 8000

    app = _build_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    _main()
