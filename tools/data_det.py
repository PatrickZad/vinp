import os
from fire import Fire

from PIL import Image

import torch
import torchvision.transforms as tvF
from torchvision.ops import box_convert
from torch.utils.data import Dataset,DataLoader

from ram.models import ram_plus
from ram import inference_ram
import groundingdino.util.inference as gdino_inf # load_model, load_image, predict, annotate

root_dir="data"
save_annos=[]
class Rrt3dFrames(Dataset):
    def __init__(self,n_p,i_p,transforms):
        v_list=[ vf for vf in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,vf))]
        v_list=sorted(v_list)
        n_samps=len(v_list)//n_p
        if len(v_list)%n_p > i_p:
            n_samps+=1
        v_list=[v_list[i*n_p+i_p] for i in range(n_samps)]
        self.imgns=[]
        for vn in v_list:
            imgns=[ imgf for imgf in os.listdir(os.path.join(root_dir,vn,"imgs_3fps_360p")) if imgf.endswith(".png")]
            for imgn in imgns:
                self.imgns.append(os.path.join(root_dir,vn,"imgs_3fps_360p",imgn))
        self.transforms=transforms
    def __len__(self):
        return len(self.imgns)
    def __getitem__(self, index):
        fpath=self.imgns[index]
        items=fpath.split("/")
        img=Image.open(os.path.join(root_dir,fpath))
        img=self.transforms(img)
        return img,items[0],items[2]

device = torch.device("cuda")    
# ram model

ram_model = ram_plus(pretrained="model_zoo/ram_plus_swin_large_14m.pth",
                             image_size=384,
                             vit='swin_l').eval().to(device)
ram_transform=tvF.Compose([
    tvF.Pad(padding=[0,0,0,24]), # 360 -> 384
    tvF.ToTensor(),
    tvF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# gdino model

gdino_model =gdino_inf.load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "model_zoo/groundingdino_swint_ogc.pth")

BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25



def annotate(n_p,i_p):
    dataset=Rrt3dFrames(n_p,i_p)
    for img,vidn,imgn in dataset:
        img=img.to(device)
        img=img.unsqueeze(0)
        cates = inference_ram(img, ram_model)[0].split(" | ")
        
        TEXT_PROMPT=" . ".join(cates)+" ."
        gdino_inf.annotate
        boxes, logits, phrases = gdino_inf.predict(
            model=gdino_model,
            image=img[0],
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )

        h, w, = img.shape[-2:]
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        labels = phrases


if __name__ == "__main__":
    Fire(annotate)