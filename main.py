import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from fastapi import FastAPI
from pydantic import BaseModel
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import torch
import random

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

gauth = GoogleAuth()           
drive = GoogleDrive(gauth)  

class Prompt(BaseModel):
    prompt: str

@app.post("/generate-audio")
async def generate_audio(prompt: Prompt):
    model = MusicGen.get_pretrained('large')
    model.set_generation_params(duration=8)  # generate 8 seconds.

    descriptions = [prompt.prompt]
    wav = model.generate(descriptions)  # generates 1 sample.

    # # Convert the audio sample to a byte stream
    file = f'{random.randint(1, 100)}.wav'

    audio_write(file, wav[0].cpu(), model.sample_rate, strategy="loudness")

    file_path = 'C:/Users/Ashan/Desktop/MusicGenApi/' + file + ".wav"

    # Upload the audio file to Google Drive
    file_drive = drive.CreateFile({'title': file})
    file_drive.SetContentFile(file_path)
    file_drive.Upload()

    # Get the file ID of the uploaded file
    file_id = file_drive['id']

    # Construct the streaming URL using the file ID
    streaming_url = f"https://drive.google.com/uc?id={file_id}"

    return {"streaming_url": streaming_url}