from huggingface_hub import hf_hub_download
from tqdm import tqdm
import os
save_dir="data"
repo_id="roomtour3d/room_tour_video_3fps"
with open("tools/fns","r") as f:
    fns=f.readlines()
fns=[fn.strip() for fn in fns]
with tqdm(total=len(fns)) as pbar:
    for fn in fns:
        if not os.path.exists(os.path.join(save_dir,fn)):
            hf_hub_download(repo_id=repo_id, filename="fleurs.py", repo_type="dataset")
        pbar.update(1)
    