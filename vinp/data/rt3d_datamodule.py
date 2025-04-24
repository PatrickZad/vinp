from torch.utils.data import DataLoader
import lightning as L

from .rt3d_dataset import Rt3dVideoDataset

class Rt3dVideo(L.LightningDataModule):
    def __init__(self,cfg,root_dir="data"):
        super().__init__()
        self.cfg=cfg
        self.batch_size=cfg.TRAIN.OPTIM.BATCH_SIZE
        self.root_dir=root_dir

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.rt3d_train_set=Rt3dVideoDataset(self.cfg,self.root_dir)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            # TODO
            pass


    def train_dataloader(self):
        return DataLoader(self.rt3d_train_set, batch_size=self.cfg.TRAIN.OPTIM.BATCH_SIZE)

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        # TODO
        pass