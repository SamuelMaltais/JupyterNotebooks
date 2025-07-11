import youtube_dl as yt_dlp
import ffmpeg
import pandas as pd
import numpy as np
import csv
import threading
from tqdm import tqdm
from os.path import exists

def download_audio(YTID: str, path: str) -> None:
    """
    Créez une fonction qui télécharge l'audio de la vidéo Youtube avec un identifiant donné
    et l'enregistre dans le dossier donné par `path`. Téléchargez-le en mp3. S'il y a un problème lors du téléchargement du fichier, gérez l'exception. Si il y a déjà un fichier à `path`, la fonction devrait retourner sans tenter de le télécharger à nouveau.

    ** Utilisez la librairie youtube_dl : https://github.com/ytdl-org/youtube-dl/ **
    
    Arguments :
    - YTID : Contient l'identifiant youtube, la vidéo youtube correspondante peut être trouvée sur
    'https://www.youtube.com/watch?v='+YTID
    - path : Le chemin d'accès au fichier où l'audio sera enregistré
    """
    if exists(path):
        return

    youtube_url = f'https://www.youtube.com/watch?v={YTID}'


    ydl_opts = {
        'format': 'bestaudio/best',  
        'outtmpl': path,          
        'postprocessors': [{
            'key': 'FFmpegExtractAudio', 
            'preferredcodec': 'mp3',    
            'preferredquality': '192',   
        }],
        'quiet': True,
        'no_warnings': True 
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
    except Exception as e:
        print(f"Erreur {e}")


def cut_audio(in_path: str, out_path: str, start: float, end: float) -> None:
    """
    Créez une fonction qui coupe l'audio de in_path pour n'inclure que le segment de start à end et l'enregistre dans out_path.

    ** Utilisez la bibliothèque ffmpeg : https://github.com/kkroening/ffmpeg-python
    Arguments :
    - in_path : Chemin du fichier audio à couper
    - out_path : Chemin du fichier pour enregistrer l'audio coupé
    - start : Indique le début de la séquence (en secondes)
    - end : Indique la fin de la séquence (en secondes)
    """
    try:

        (
            ffmpeg
            .input(in_path, ss=start, to=end)
            .output(out_path, vframes=1)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        pass