from fastapi import FastAPI
from openai import BaseModel

from rag import answer_question, ingest_video


app = FastAPI(
  title="RAG API",
  description="A simple API for RAG",
  version="0.1.0"
)

@app.get("/")
async def root():
  return {"message": "Hello World"}

class IngestRequest(BaseModel):
  url_video: str
  video_id: str
class RagRequest(BaseModel):
  video_id: str
  question: str

@app.post("/api/v1/ingest")
async def ingest(request: IngestRequest):
  ingest_video(request.url_video, request.video_id)
  return {"Status": "success"}


@app.post("/api/v1/rag")
async def rag(request: RagRequest):
  answer = answer_question(request.question, request.video_id)
  return {"answer": answer}