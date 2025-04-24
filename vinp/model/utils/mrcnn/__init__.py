from .mrcnn import build_mrcnn
from detectron2.config import get_cfg

MODEL_WEIGHT_REGISTRY={
    "R50FPN":"detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl",
    "R101FPN":"detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl",
    "R50FPNv2":"detectron2://new_baselines/mask_rcnn_R_50_FPN_400ep_LSJ/42019571/model_final_14d201.pkl",
    "R101FPNv2":"detectron2://new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ/42073830/model_final_f96b26.pkl",
}
MODEL_CONFIG_REGISTRY={
    "R50FPN":"vinp/model/utils/mrcnn/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    "R101FPN":"vinp/model/utils/mrcnn/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
    "R50FPNv2":"vinp/model/utils/mrcnn/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    "R101FPNv2":"vinp/model/utils/mrcnn/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
}

def mrcnn_r50_fpn():
    return _mrcnn_model("R50FPN")

def mrcnn_r101_fpn():
    return _mrcnn_model("R101FPN")

def mrcnn_r50_fpn_v2():
    return _mrcnn_model("R50FPNv2")

def mrcnn_r101_fpn_v2():
    return _mrcnn_model("R101FPNv2")

def _mrcnn_model(name):
    cfg_file=MODEL_CONFIG_REGISTRY[name]
    weight_file=MODEL_WEIGHT_REGISTRY[name]
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    cfg.merge_from_list(["MODEL.WEIGHTS",weight_file])
    cfg.freeze()
    return build_mrcnn(cfg)

mrcnn_model_registry={
    "R50FPN":mrcnn_r50_fpn,
    "R101FPN":mrcnn_r101_fpn,
    "R50FPNv2":mrcnn_r50_fpn_v2,
    "R101FPNv2":mrcnn_r101_fpn_v2,
}