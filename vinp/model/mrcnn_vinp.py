from .base import VideoInstanceNavEncoder

from vinp.model.utils.mrcnn import mrcnn_model_registry

class MrcnnVinp(VideoInstanceNavEncoder):
    def _build_encoder_model(cfg):
        return mrcnn_model_registry[cfg.MODEL.MRCNN.MDOEL]()
    def visual_encoder(self,ibatch):
        if self.input_format == "RGB":
            ibatch=ibatch[:,::-1] # to BGR
        return self.encoder_model.backbone(ibatch)