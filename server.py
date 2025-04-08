from fastapi import Depends, FastAPI, HTTPException
from openai import BaseModel
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from rag import answer_question, ingest_video
from dotenv import load_dotenv
import os

load_dotenv()

USERNAME = os.getenv("ADMIN_USERNAME")
PASSWORD = os.getenv("ADMIN_PASSWORD")

app = FastAPI(
  title="RAG API",
  description="A simple API for RAG",
  version="0.1.0"
)
security = HTTPBasic()

class IngestRequest(BaseModel):
  url_video: str
  video_id: str
class RagRequest(BaseModel):
  video_id: str
  question: str


def is_client_valid(credentials: HTTPBasicCredentials = Depends(security)):
  if credentials.username != USERNAME or credentials.password != PASSWORD:
    return False
  return True

@app.post("/api/v1/ingest")
async def ingest(request: IngestRequest, is_client_valid: bool = Depends(is_client_valid)):
  if not is_client_valid:
    raise HTTPException(
      status_code=401,
      detail="Invalid Authentication credentials"
    )
  ingest_video(request.url_video, request.video_id)
  return {"Status": "success"}


@app.post("/api/v1/rag")
async def rag(request: RagRequest, is_client_valid: bool = Depends(is_client_valid)):
  if not is_client_valid:
    raise HTTPException(
      status_code=401,
      detail="Invalid Authentication credentials"
    )
  answer = answer_question(request.question, request.video_id)
  return {"answer": answer}