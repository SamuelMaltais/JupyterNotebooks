import re
import os
import pandas as pd
from tqdm import tqdm
from q2 import download_audio, cut_audio
from typing import List

def filter_df(csv_path: str, label: str) -> List[str]:
    """
    Écrivez une fonction qui prend le path vers le csv traité (dans la partie notebook de q1) et renvoie un df avec seulement les rangées qui contiennent l'étiquette `label`.

    Par exemple:
    get_ids("audio_segments_clean.csv", "Speech") ne doit renvoyer que les lignes où l'un des libellés est "Speech"
    """
    df = pd.read_csv(csv_path)
    regex = fr'(^|\|){label}(\||$)'
    filtree = df[df['label_names'].str.contains(regex, na=False)]
    return filtree

def data_pipeline(csv_path: str, label: str) -> None:
    """
    En utilisant vos fonctions précédemment créées, écrivez une fonction qui prend un csv traité et pour chaque vidéo avec l'étiquette donnée:
    1. Le télécharge à <label>_raw/<ID>.mp3
    2. Le coupe au segment approprié
    3. L'enregistre dans <label>_cut/<ID>.mp3
    (n'oubliez pas de créer le dossier audio/ et le dossier label associé !).

    Il est recommandé d'itérer sur les rangées de filter_df().
    Utilisez tqdm pour suivre la progression du processus de téléchargement (https://tqdm.github.io/)

    Malheureusement, il est possible que certaines vidéos ne peuvent pas être téléchargées. Dans de tels cas, votre pipeline doit gérer l'échec en passant à la vidéo suivante avec l'étiquette.
    """
    raw_dir = f"{label}_raw"
    cut_dir = f"{label}_cut"
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(cut_dir, exist_ok=True)

    filtered_df = filter_df(csv_path=csv_path, label=label)

    filtered_df.columns = filtered_df.columns.str.strip()

    for index, row in tqdm(filtered_df.iterrows(), total=filtered_df.shape[0]):
        video_id = row['# YTID']
        start = row['start_seconds']
        end = row['end_seconds']

        try:
            raw_audio_path = os.path.join("audio",raw_dir, f"{video_id}.mp3")
            # On assume que la video est deja downloaded.
            #download_audio(video_id, raw_audio_path)

            cut_audio_path = os.path.join("audio",cut_dir, f"{video_id}.mp3")
            cut_audio(raw_audio_path, cut_audio_path, start, end)

        except Exception as e:
            continue

def rename_files(path_cut: str, csv_path: str) -> None:
    """
    Supposons que nous voulons maintenant renommer les fichiers que nous avons téléchargés dans `path_cut` pour inclure les heures de début et de fin ainsi que la longueur du segment. Alors que
    cela aurait pu être fait dans la fonction data_pipeline(), supposons que nous avons oublié et que nous ne voulons pas tout télécharger à nouveau.

    Écrivez une fonction qui, en utilisant regex (c'est-à-dire la bibliothèque `re`), renomme les fichiers existants de "<ID>.mp3" -> "<ID>_<start_seconds_int>_<end_seconds_int>_<length_int>.mp3"
    dans path_cut. csv_path est le chemin vers le csv traité à partir de q1. `path_cut` est un chemin vers le dossier avec l'audio coupé.

    Par exemple
    "--BfvyPmVMo.mp3" -> "--BfvyPmVMo_20_30_10.mp3"

    ## ATTENTION : supposez que l'YTID peut contenir des caractères spéciaux tels que '.' ou même '.mp3' ##
    """
    df = pd.read_csv(csv_path)

    df.columns = df.columns.str.strip()

    for filename in os.listdir(path_cut):
        match = re.match(r'([^.]+)', filename)  # extraire l'ID du fichier sans l'extension
   
        if match:
            video_id = match.group(1)
            row = df[df['# YTID'] == video_id]

            if not row.empty:
                start = int(row['start_seconds'].values[0])
                end = int(row['end_seconds'].values[0])
                length = end - start

                new_filename = f"{video_id}_{start}_{end}_{length}.mp3"
                old_path = os.path.join(path_cut, filename)
                new_path = os.path.join(path_cut, new_filename)

                os.rename(old_path, new_path)


if __name__ == "__main__":
    print(filter_df("audio_segments_clean.csv", "Laughter"))
    #cut_audio('./Hammer_raw/0GNNFBrRz1E.mp3', './Hammer_cut/0GNNFBrRz1E.mp3', 40, 50)
    data_pipeline("audio_segments_clean.csv", "Hammer")
    rename_files("Hammer_cut", "audio_segments_clean.csv")
