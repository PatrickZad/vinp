import os
import zipfile
from tqdm import tqdm

data_dir="data"
fns=os.listdir(data_dir)
with tqdm(total=len(fns)) as pbar:
    for fn in fns:
        if fn.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(data_dir,fn), 'r') as zip_ref:
                zip_ref.extractall(data_dir)
        pbar.update(1)
        
