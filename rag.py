import re
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chat_models import init_chat_model
from pytubefix import YouTube
from typing_extensions import List
from langchain_core.documents import Document
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

def transcribe_video(video_id: str):
  print(f"Transcribing video...")
  file_path = f"audio/{video_id}.mp3"
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

def download_captions(url: str, video_id: str):
  print(f"Downloading captions...")
  yt = YouTube(url)
  print(yt.captions)
  caption = yt.captions.get_by_language_code('en')
  if caption is None:
    caption = yt.captions.get_by_language_code('a.en')
  if caption is None:
    caption = yt.captions.get_by_language_code('a.id')
  if caption is None:
    # fallback to download video with audio
    download_video(url, video_id)
    transcribe_video(video_id)
    return
  caption_text = caption.generate_srt_captions()
  caption_text = re.sub(r'\d+\n', '', caption_text)
  # remove the timestamps
  caption_text = re.sub(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},', '', caption_text)
  caption_text = re.sub(r'\n', ' ', caption_text)
  
  with open(f"transcripts/{video_id}.txt", "w") as f:
    f.write(caption_text)
  
def split_transcript(video_id: str):
  loader = TextLoader(f"transcripts/{video_id}.txt")
  docs = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
  all_splits = text_splitter.split_documents(docs)
  print(len(all_splits))
  url = f"https://www.youtube.com/watch?v={video_id}"
  yt = YouTube(url)
  # adding metadata to the documents
  for doc in all_splits:
    doc.metadata.update({
      "video_id": video_id,
      "video_author": yt.author,
      "video_title": yt.title,
      "video_description": yt.description,
    })
  return all_splits

def retrieve(vector_store, question: str, video_id: str):
  def _filter_document(doc: Document) -> bool:
    return doc.metadata.get("video_id", None) == video_id
  retrieve_docs = vector_store.similarity_search(question, k= 3, filter=_filter_document)
  print(len(retrieve_docs))
  return retrieve_docs

def generate_answer(question: str, docs: List[Document]):
  docs_content = "\n\n".join(doc.page_content for doc in docs)
  prompt = hub.pull("rlm/rag-prompt")
  prompt = prompt.invoke({"question": question, "context": docs_content})
  answer = llm.invoke(prompt)
  return answer.content

def ingest_video(url_video: str, video_id: str):
  download_captions(url_video, video_id)
  all_splits = split_transcript(video_id)
  vector_store.add_documents(all_splits)

def answer_question(question: str, video_id: str):
  retrieve_docs = retrieve(vector_store, question, video_id)
  response = generate_answer(question, retrieve_docs)
  return response

if __name__ == "__main__":
  # get the question and url from the user
  input_url = input("Enter the URL of the video to download: ")
  download_captions(input_url, "2")

  all_splits = split_transcript("2")
  vector_store.add_documents(all_splits)
  question = ""

  while question != "exit":
    question = input("Enter a question: ")
    response = answer_question(question, "2")
    print(response)