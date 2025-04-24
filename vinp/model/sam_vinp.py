from .base import VideoInstanceNavEncoder

from vinp.model.utils.sam import sam_model_registry



class SamVinp(VideoInstanceNavEncoder):
    def _build_encoder_model(cfg):
        return sam_model_registry[cfg.MODEL.SAM.MDOEL](cfg.MODEL.SAM.CHECKPOINT)

    