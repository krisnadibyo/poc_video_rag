from fastapi import FastAPI
from openai import BaseModel

from rag import rag_video


app = FastAPI(
  title="RAG API",
  description="A simple API for RAG",
  version="0.1.0"
)

@app.get("/")
async def root():
  return {"message": "Hello World"}

class RAGRequest(BaseModel):
  question: str
  url_video: str  

@app.post("/api/v1/rag")
async def rag(request: RAGRequest):
  answer = rag_video(request.question, request.url_video)
  return {"answer": answer}