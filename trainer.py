from argparse import ArgumentParser
import os

import lightning as L
import lightning.pytorch.callbacks as callbacks

from vinp.model import model_registry
from vinp.config import get_cfg

def main(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()

    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    model=model_registry[cfg.MODEL.NAME](cfg)

    trainer=L.Trainer(accelerator="gpu",devices=args.ngpu,num_nodes=1,callbacks=[callbacks.DeviceStatsMonitor(),callbacks.LearningRateMonitor(),callbacks.ModelCheckpoint(dirpath=cfg.OUTPUT_DIR,monitor="sr",save_top_k=5,mode="max"),callbacks.ModelSummary()])

    trainer.fit()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cfg",type=str)
    parser.add_argument("--ngpu",type=int)
    args=parser.parse_args()

    main(args)
