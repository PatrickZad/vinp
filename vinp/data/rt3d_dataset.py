import os
import json
import math

import numpy as np
from detectron2.data.detection_utils import read_image

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import detectron2.data.transforms as d2t

from .transforms import RandomAffine,MixUp,RandomInstanceErasing

class Rt3dVideoDataset(Dataset):
    def __init__(self,cfg,root_dir="data"):
        super().__init__()
        self.setup(root_dir)

        data_cfg=cfg.TRAIN.DATA
        self.base_fps=3
        self.resample_fps=min(data_cfg.RESAMPLE_FPS,self.base_fps)
        self.clip_len=data_cfg.CLIP_LEN

        self.rand_affine= RandomAffine() if data_cfg.RAND_AFFINE else None
        self.mix_up=MixUp(img_scale=(640,360)) if data_cfg.MIX_UP else None
        self.rea=RandomInstanceErasing(p=data_cfg.REA_P) if data_cfg.REA else None

        self.transforms=d2t.AugmentationList([d2t.ResizeScale(min_scale=data_cfg.RESIZE[0],max_scale=data_cfg.RESIZE[1]),d2t.RandomFlip(prob=0.5,horizontal=True,vertical=False)])
        self.totensor=ToTensor()
        self.input_format=cfg.MODEL.INPUT.INPUT_FORMAT
    def setup(self,root_dir):
        # read image filenames and annotations
        # filter empty ones
        v_list=[ vf for vf in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,vf))]
        self.video_list=[]
        self.ann_list=[]
        for vname in v_list:
            ann_fn=os.path.join(root_dir,vname+".json")
            if os.path.exists(ann_fn):
                with open(ann_fn,"r") as f:
                    ann=json.load(f)
                fns=sorted(list(ann.keys()))
                frames=[]
                frame_anns=[]
                for fn in fns :
                    f_ann=ann[fn]
                    if len(f_ann)>0:
                        frames.append(os.path.join(root_dir,vname,"imgs_3fps_360p",fn))
                        frame_anns.append(f_ann)
                self.video_list.append(frames)
                self.ann_list.append(frame_anns)
    def __len__(self):
        return len(self.video_list)
    def __getitem__(self, index):
        vid_frames=self.video_list[index]
        vid_anns=self.ann_list[index]
        n_f=len(vid_frames)
        # resample video
        if self.resample_fps<self.base_fps:
            resamp_base=self.base_fps/self.resample_fps
            resample_start=np.random.randint(0,math.ceil(resamp_base))
            resample_indices=np.linspace(start=resample_start,stop=n_f,num=max(int(n_f/resamp_base),self.clip_len)).astype(np.int32).tolist()
            resample_indices=sorted(list(set(resample_indices)))
            vid_frames=vid_frames[resample_indices]
            vid_anns=vid_anns[resample_indices]
            n_f=len(vid_frames)
        # random clip
        clip_start=np.random.choice(n_f-self.clip_len+1)
        images=[]
        obj_boxes=[]
        obj_labels=[]
        
        for i in range(self.clip_len):
            img=read_image(vid_frames[clip_start+i])
            boxes=[]
            labels=[]
            for ann in vid_anns[clip_start+i]:
                boxes.append(ann[1:])
                labels.append(ann[0])
            boxes=np.array(boxes,dtype=np.float32)
            if self.rand_affine is not None:
                img,boxes,labels=self.rand_affine(img,boxes,labels)
            if self.mix_up is not None:
                mixup_vid=np.random.randint(0,len(self))
                mixup_vid_frames=self.video_list[mixup_vid]
                mixup_vid_anns=self.ann_list[mixup_vid]
                mixup_n_f=len(mixup_vid_frames)
                mixup_fid=np.random.randint(0,mixup_n_f)
                mixup_img=read_image(mixup_vid_frames[mixup_fid])
                mixup_boxes=[]
                mixup_labels=[]
                for ann in mixup_vid_anns[mixup_fid]:
                    mixup_boxes.append(ann[1:])
                    mixup_labels.append(ann[0])
                mixup_boxes=np.array(mixup_boxes,dtype=np.float32)
                img,boxes,labels=self.mix_up(img,boxes,labels,mixup_img,mixup_boxes,mixup_labels)
            if self.rea is not None:
                img=self.rea(img,boxes)
            aug_input = d2t.AugInput(image=img.copy(), boxes=boxes.copy())
            img = aug_input.image
            boxes = aug_input.boxes
            img= self.totensor(img.copy())
            images.append(img)
            obj_boxes.append(boxes)
            obj_labels.append(labels)
        return images,obj_boxes,obj_labels
