from pytubefix import YouTube
import whisper
import os

def download_video(url: str):
  yt = YouTube(url)
  file_name = f"{yt.title.replace(' ', '_')}.mp3"
  output_path = "audio"
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  yt.streams.filter(only_audio=True).first().download(output_path=output_path, filename=file_name)
  return f"{output_path}/{file_name}"

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

if __name__ == "__main__":
  # download and transcribe the video
  input_url = input("Enter the URL of the video to download: ")
  transcribe_video(download_video(input_url))