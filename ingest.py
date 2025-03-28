from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chat_models import init_chat_model
from pytubefix import YouTube
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from langchain import hub
from langchain.document_loaders import TextLoader
from dotenv import load_dotenv

import whisper
import os

load_dotenv()

def download_video(url: str):
  yt = YouTube(url)
  file_name = f"{yt.title.replace(' ', '_')}.mp3"
  output_path = "audio"
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  yt.streams.filter(only_audio=True).first().download(output_path=output_path, filename=file_name)
  return f"{output_path}/{file_name}"

def get_chat_model():
  if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set")
  llm = init_chat_model(model="gpt-4o-mini", model_provider="openai")
  return llm

def get_embeddings():
  if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set")
  embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
  return embeddings

def get_vector_store(embeddings):
  vector_store = InMemoryVectorStore(embeddings)
  return vector_store

def transcribe_video(file_path: str):
  model = whisper.load_model("tiny", device="cpu")
  result = model.transcribe(file_path)
  text = result["text"]
  output_path = "transcripts"
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  file_name = file_path.split('/')[-1].split('.')[0]
  with open(f"{output_path}/{file_name}.txt", "w") as f:
    f.write(text)

def load_transcript(file_path: str):
  loader = TextLoader(file_path)
  docs = loader.load()
  return docs
  
def split_transcript(transcript: str):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  all_splits = text_splitter.split_documents(transcript)
  return all_splits

class State(TypedDict):
  question: str
  answer: str 
  context: List[Document]

def retreive(vector_store, question: str):
  retrieve_docs = vector_store.similarity_search(question)
  return retrieve_docs

def generate_answer(llm, question: str, docs: List[Document]):
  docs_content = "\n\n".join(doc.page_content for doc in docs)
  prompt = hub.pull("rlm/rag-prompt")
  prompt = prompt.invoke({"question": question, "context": docs_content})
  answer = llm.invoke(prompt)
  return answer.content


if __name__ == "__main__":
  # download and transcribe the video
  # input_url = input("Enter the URL of the video to download: ")
  # transcribe_video(download_video(input_url))

  # load the transcript
  transcript = load_transcript("transcripts/transcript_2.txt")

  # split the transcript into chunks
  all_splits = split_transcript(transcript)

  # Index the chunks
  vector_store = get_vector_store(get_embeddings())
  _ = vector_store.add_documents(all_splits)

  question = input("Enter a question: ")
  retrieve_docs = retreive(vector_store, question)
  response = generate_answer(get_chat_model(), question, retrieve_docs)
  print(response)