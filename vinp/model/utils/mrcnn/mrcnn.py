from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
def build_mrcnn(cfg):
    mrcnn=build_model(cfg)
    mrcnn.eval()
    checkpointer = DetectionCheckpointer(mrcnn)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    return mrcnn