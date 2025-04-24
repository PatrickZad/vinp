import warnings
import copy

import torch
import torch.optim as toptim
import torch.optim.lr_scheduler as tsched

import lightning as L


from vinp.model.iadapter import iadapter_registry
from vinp.model.ihead import ihead_registry


class VideoInstanceNavEncoder(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.encoder_model=self._build_encoder_model(cfg)
        self.iadapter=iadapter_registry[cfg.MODEL.IADAPTER.MODEL](cfg)
        self.ihead=ihead_registry[cfg.MODEL.IHEAD.MODEL](cfg)
        self.register_buffer(
            "pixel_mean", torch.Tensor(cfg.TRAIN.DATA.RGB_MEAN).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.Tensor(cfg.TRAIN.DATA.RGB_STD).view(-1, 1, 1), False)
        self.input_format=cfg.TRAIN.DATA.INPUT_FORMAT

    def train(self, mode = True):
        rst=super().train(mode)
        self.encoder_model.eval()
        return rst
    def _build_encoder_model(cfg):
        raise NotImplementedError
    def _build_iadapter(cfg):
        return iadapter_registry[cfg.MODEL.IADAPTER.MODEL](cfg)
    def _build_ihead(cfg):
        return ihead_registry[cfg.MODEL.IADAPTER.MODEL](cfg)
        
    def visual_encoder(self,ibatch):
        raise NotImplementedError
    def training_step(self, *args, **kwargs):
        return super().training_step(*args, **kwargs)
    def configure_optimizers(self):
        optim_cfg=self.hparams.cfg.TRAIN.OPTIM
        learnable_weights_iadpater=[]
        learnable_bias_iadpater=[]
        learnable_weights_ihead=[]
        learnable_bias_ihead=[]
        for n,p in self.named_parameters():
            if p.requires_grad:
                if n.startswith("iadapter"):
                    if n.endswith("bias"):
                        learnable_bias_iadpater.append(p)
                    else:
                        learnable_weights_iadpater.append(p)
                else:
                    assert n.startswith("ihead"), "Unexpected parameter {}".format(n)
                    if n.endswith("bias"):
                        learnable_bias_ihead.append(p)
                    else:
                        learnable_weights_ihead.append(p)
        optim_param_iadapter=[
            {"params":learnable_weights_iadpater,"weight_decay":optim_cfg.WEIGHT_DECAY_IADAPTER,"lr":optim_cfg.INIT_LR_IADAPTER},
            {"params":learnable_bias_iadpater,"weight_decay":optim_cfg.WEIGHT_DECAY_BIAS_IADAPTER,"lr":optim_cfg.INIT_LR_IADAPTER},]
        optim_param_ihead=[
            {"params":learnable_weights_ihead,"weight_decay":optim_cfg.WEIGHT_DECAY_IHEAD,"lr":optim_cfg.INIT_LR_IHEAD},
            {"params":learnable_bias_ihead,"weight_decay":optim_cfg.WEIGHT_DECAY_BIAS_IHEAD,"lr":optim_cfg.INIT_LR_IHEAD},
        ]
        if optim_cfg.OPT.lower() == "sgd":
            optimizers=[toptim.sgd.SGD(optim_param_iadapter,momentum=optim_cfg.MOMENTUM),toptim.sgd.SGD(optim_param_ihead,momentum=optim_cfg.MOMENTUM)]
        elif optim_cfg.OPT.lower() == "adam":
            optimizers = [toptim.adam.Adam(optim_param_iadapter),toptim.adam.Adam(optim_param_ihead)]
        elif optim_cfg.OPT.lower() == "adamw":
            optimizers=[toptim.adamw.AdamW(optim_param_iadapter),toptim.adamw.AdamW(optim_param_ihead)]
        else:
            raise KeyError("Unrecognized optimizer {}".format(optim_cfg.OPT))
        warnings.warn("Lr WarmUp not supported yet !")
        default_sched_cfg={"interval" : "step", "frequency" : 50, "strict" : False}
        if optim_cfg.SCHED.lower() == "step":
            schedulers=[
                {
                    "scheduler":tsched.MultiStepLR(optimizers[0],milestones=optim_cfg.SCHED_STEPS,gamma=optim_cfg.SCHED_STEP_GAMMA),
                    "name":"iadapter"
                },
                {"scheduler":tsched.MultiStepLR(optimizers[1],milestones=optim_cfg.SCHED_STEPS,gamma=optim_cfg.SCHED_STEP_GAMMA),"name":"ihead"},
                ]
        elif optim_cfg.SCHED.lower() == "cosine":
            schedulers=[{"scheduler":tsched.CosineAnnealingLR(optimizers[0],T_max=optim_cfg._EPOCHS,eta_min=optim_cfg.MIN_LR_IADAPTER),"name":"iadapter"},{"scheduler":tsched.CosineAnnealingLR(optimizers[1],T_max=optim_cfg._EPOCHS,eta_min=optim_cfg.MIN_LR_IHEAD),"name":"ihead"}]
        scheduler_cfgs=[]
        for sched in schedulers:
            scfg=copy.deepcopy(default_sched_cfg)
            scfg.update(sched)
            scheduler_cfgs.append(scfg)
        return optimizers,scheduler_cfgs