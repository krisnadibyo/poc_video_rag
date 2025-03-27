import os
from pytubefix import YouTube
from pydub import AudioSegment
from pydub.silence import split_on_silence
import whisper


def download_video(url: str):
  yt = YouTube(url)
  yt.streams.filter(only_audio=True).first().download(output_path="audio", filename=f"{yt.title.replace(' ', '_')}.mp3")

def read_audio(file_path: str):
  audio = AudioSegment.from_mp3(file_path)
  return audio


if __name__ == "__main__":
  file_path = "audio/sample.mp3"
  # audio = read_audio(file_path)

  # chunks = split_on_silence(audio)

  # if not os.path.isdir("chunks"):
  #   os.makedirs("chunks")
  
  # print(f"Splitting audio into {len(chunks)} chunks")
  # for i, chunk in enumerate(chunks):
  #   chunk.export(f"chunks/chunk{i}.mp3", format="mp3")
  #   model = whisper.load_model("base")
  #   result = model.transcribe(f"chunks/chunk{i}.mp3")
  #   text = result["text"]
  #   with open(f"chunks/chunk{i}.txt", "w") as f:
  #     f.write(text)
  model = whisper.load_model("tiny", device="cpu")
  
  result = model.transcribe(file_path)
  text = result["text"]
  with open("transcript.txt", "w") as f:
    f.write(text)