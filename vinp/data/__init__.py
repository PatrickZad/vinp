from torch.utils.data import DataLoader,Dataset

class VideoClipDataset(Dataset):
    def __init__(self,base_fps,samp_fps,clip_len,video_list,transforms,video_sampling=False,is_train=False):
        super().__init__()
        self.base_fps=base_fps
        self.samp_fps=samp_fps
        self.video_list=video_list
        self.video_sampling=video_sampling
        self.transforms=transforms
        if self.video_sampling:
            clip_list=[]
            
            
    def __len__(self):
        return len(self.video_list)
    def __getitem__(self,idx):
        pass


def build_dataloader(cfg):
    pass
def build_dataset(cfg):
    pass