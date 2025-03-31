import time
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
llm = init_chat_model(model="gpt-4o-mini", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)

def download_video(url: str, video_id: str):
  print(f"Downloading video...")
  yt = YouTube(url)
  file_name = f"{video_id}.mp3"
  output_path = "audio"
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  yt.streams.filter(only_audio=True).first().download(output_path=output_path, filename=file_name)
  return f"{output_path}/{file_name}"

# def get_chat_model():
#   if not os.environ.get("OPENAI_API_KEY"):
#     raise ValueError("OPENAI_API_KEY is not set")
#   llm = init_chat_model(model="gpt-4o-mini", model_provider="openai")
#   return llm

# def get_embeddings():
#   if not os.environ.get("OPENAI_API_KEY"):
#     raise ValueError("OPENAI_API_KEY is not set")
#   embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
#   return embeddings

# def get_vector_store(embeddings):
#   vector_store = InMemoryVectorStore(embeddings)
#   return vector_store

def transcribe_video(file_path: str):
  print(f"Transcribing video...")
  model = whisper.load_model("tiny", device="cpu")
  result = model.transcribe(file_path)
  text = result["text"]
  output_path = "transcripts"
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  file_name = file_path.split('/')[-1].split('.')[0]
  with open(f"{output_path}/{file_name}.txt", "w") as f:
    f.write(text)
  # delete the audio file
  os.remove(file_path)
  return f"{output_path}/{file_name}.txt"

def load_transcript(file_path: str):
  loader = TextLoader(file_path)
  docs = loader.load()
  return docs
  
def split_transcript(transcript: str, video_id: str):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
  all_splits = text_splitter.split_documents(transcript)
  print(len(all_splits))
  for doc in all_splits:
    doc.metadata.update({"video_id": video_id})
  return all_splits

def retreive(vector_store, question: str, video_id: str):
  def _filter_document(doc: Document) -> bool:
    return doc.metadata.get("video_id", None) == video_id
  retrieve_docs = vector_store.similarity_search(question, k= 1, filter=_filter_document)
  print(len(retrieve_docs))
  return retrieve_docs

def generate_answer(llm, question: str, docs: List[Document]):
  docs_content = "\n\n".join(doc.page_content for doc in docs)
  prompt = hub.pull("rlm/rag-prompt")
  prompt = prompt.invoke({"question": question, "context": docs_content})
  answer = llm.invoke(prompt)
  return answer.content

def ingest_video(url_video: str, video_id: str):
  file_name = download_video(url_video, video_id)
  transcript = transcribe_video(file_name)
  transcript = load_transcript(transcript)
  all_splits = split_transcript(transcript, video_id)
  vector_store.add_documents(all_splits)

def answer_question(question: str, video_id: str):
  retrieve_docs = retreive(vector_store, question, video_id)
  response = generate_answer(llm, question, retrieve_docs)
  return response

# def rag_video(question: str, url_video: str):
#   file_name = download_video(url_video)
#   transcript = transcribe_video(file_name)
#   transcript = load_transcript(transcript)
#   all_splits = split_transcript(transcript)
#   vector_store.add_documents(all_splits)
#   retrieve_docs = retreive(vector_store, question)
#   response = generate_answer(llm, question, retrieve_docs)
#   return response

if __name__ == "__main__":
  # get the question and url from the user
  input_url = input("Enter the URL of the video to download: ")
  question = input("Enter a question: ")

  # download and transcribe the video
  file_name = transcribe_video(download_video(input_url))

  # load the transcript
  transcript = load_transcript(file_name)

  # split the transcript into chunks
  print(f"Splitting transcript into chunks...")
  all_splits = split_transcript(transcript)

  # Index the chunks
  print(f"Indexing chunks...")
  vector_store.add_documents(all_splits)


  # retrieve the most relevant documents
  question = question + " -- This is a question from a video transcript as RAG context"
  print(f"Retrieving most relevant documents...")
  retrieve_docs = retreive(vector_store, question)

  # generate the answer
  print(f"Generating answer...")
  response = generate_answer(llm, question, retrieve_docs)
  print("Answer: ", response)