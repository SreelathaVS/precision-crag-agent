import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.agent import crag_agent

app = FastAPI(
    title="Precision CRAG Agent",
    description="Corrective Retrieval-Augmented Generation Agent API",
    version="1.0.0",
)

# Mount static folder if present (for HTML/JS/CSS assets)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)


class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: int
    query_used: str


@app.get("/", include_in_schema=False)
def root():
    """Serves the frontend UI if it exists, otherwise falls back to Swagger /docs."""
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return RedirectResponse(url="/docs")


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/api/v1/query", response_model=QueryResponse)
def run_query(payload: QueryRequest):
    try:
        result = crag_agent.invoke({"question": payload.question, "loop_count": 0})
        return QueryResponse(
            answer=result["generation"],
            retrieved_chunks=len(result.get("documents", [])),
            query_used=result.get("question", payload.question),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))