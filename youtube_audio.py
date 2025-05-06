import os
import yt_dlp
import ffmpeg
import time
from datetime import datetime

def download_audio(youtube_url):
    os.makedirs("audios", exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    filename = f"audios/audio_{timestamp}.wav"

    time.sleep(2)

    # Download the audio using yt_dlp
    print('Downloading audio from youtube...')
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': filename.replace(".wav", ".%(ext)s"),  # Save as a temporary file
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])


    with open("filename_audio.txt", "w") as text_file:
        text_file.write(f"{filename}")

    print(f"Audio downloaded and saved as {filename}")

    return filename  
