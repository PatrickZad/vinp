import os
from fire import Fire
import json

from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as tvF
import torchvision.transforms.functional as tvFf
from torchvision.ops import box_convert
from torch.utils.data import Dataset,DataLoader

from ram.models.ram_plus import *
from ram import inference_ram
import groundingdino.util.inference as gdino_inf # load_model, load_image, predict, annotate

root_dir="data"
save_annos={}
class Rrt3dFrames(Dataset):
    def __init__(self,n_p,i_p,transforms_reg,transforms_det):
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
        self.transforms_reg=transforms_reg
        self.transforms_det=transforms_det
    def __len__(self):
        return len(self.imgns)
    def __getitem__(self, index):
        fpath=self.imgns[index]
        items=fpath.split("/")
        img=Image.open(fpath)
        return self.transforms_reg(img),self.transforms_det(img),items[1],items[3]

class RAM_plus_size(RAM_plus):
    def __init__(self,
                 med_config=f'{CONFIG_PATH}/configs/med_config.json',
                 image_size=(384,768),
                 text_encoder_type='bert-base-uncased',
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 threshold=0.68,
                 delete_tag_index=[],
                 tag_list=f'{CONFIG_PATH}/data/ram_tag_list.txt',
                 tag_list_chinese=f'{CONFIG_PATH}/data/ram_tag_list_chinese.txt',
                 stage='eval'):
        r""" The Recognize Anything Plus Model (RAM++) inference module.
        RAM++ is a strong image tagging model, which can recognize any category with high accuracy using tag categories.
        Described in the paper "Open-Set Image Tagging with Multi-Grained Text Supervision" https://arxiv.org/abs/2310.15200

        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
            threshold (int): tagging threshold
            delete_tag_index (list): delete some tags that may disturb captioning
        """
        super(RAM_plus,self).__init__()
        image_size,image_size_2=image_size
        # create image encoder
        if vit == 'swin_b':
            if image_size == 224:
                vision_config_path = f'{CONFIG_PATH}/configs/swin/config_swinB_224.json'
            elif image_size == 384:
                vision_config_path = f'{CONFIG_PATH}/configs/swin/config_swinB_384.json'
            vision_config = read_json(vision_config_path)
            assert image_size == vision_config['image_res']
            # assert config['patch_size'] == 32
            vision_width = vision_config['vision_width']

            self.visual_encoder = SwinTransformer(
                img_size=vision_config['image_res'],
                patch_size=4,
                in_chans=3,
                embed_dim=vision_config['embed_dim'],
                depths=vision_config['depths'],
                num_heads=vision_config['num_heads'],
                window_size=vision_config['window_size'],
                mlp_ratio=4.,
                qkv_bias=True,
                drop_rate=0.0,
                drop_path_rate=0.1,
                ape=False,
                patch_norm=True,
                use_checkpoint=False)

            if stage == 'train_from_scratch':
                # download from https://github.com/microsoft/Swin-Transformer
                state_dict = torch.load(vision_config['ckpt'], map_location="cpu")['model']

                for k in list(state_dict.keys()):
                    if 'relative_position_bias_table' in k:
                        dst_num_pos = (2 * vision_config['window_size'] - 1) ** 2
                        state_dict[k] = interpolate_relative_pos_embed(state_dict[k], dst_num_pos, param_name=k)
                    elif ('relative_position_index' in k) or ('attn_mask' in k):
                        del state_dict[k]

                print("### Load Vision Backbone", vit)
                msg = self.visual_encoder.load_state_dict(state_dict, strict = False)
                print("missing_keys: ", msg.missing_keys)
                print("unexpected_keys: ", msg.unexpected_keys)

        elif vit == 'swin_l':
            if image_size == 224:
                vision_config_path = f'{CONFIG_PATH}/configs/swin/config_swinL_224.json'
            elif image_size == 384:
                vision_config_path = f'{CONFIG_PATH}/configs/swin/config_swinL_384.json'
            vision_config = read_json(vision_config_path)
            assert image_size == vision_config['image_res']
            # assert config['patch_size'] == 32
            vision_width = vision_config['vision_width']

            self.visual_encoder = SwinTransformer(
                img_size=(image_size,image_size_2),
                patch_size=4,
                in_chans=3,
                embed_dim=vision_config['embed_dim'],
                depths=vision_config['depths'],
                num_heads=vision_config['num_heads'],
                window_size=vision_config['window_size'],
                mlp_ratio=4.,
                qkv_bias=True,
                drop_rate=0.0,
                drop_path_rate=0.1,
                ape=False,
                patch_norm=True,
                use_checkpoint=False)

            if stage == 'train_from_scratch':
                # download from https://github.com/microsoft/Swin-Transformer
                state_dict = torch.load(vision_config['ckpt'], map_location="cpu")['model']

                for k in list(state_dict.keys()):
                    if 'relative_position_bias_table' in k:
                        dst_num_pos = (2 * vision_config['window_size'] - 1) ** 2
                        state_dict[k] = interpolate_relative_pos_embed(state_dict[k], dst_num_pos, param_name=k)
                    elif ('relative_position_index' in k) or ('attn_mask' in k):
                        del state_dict[k]

                print("### Load Vision Backbone", vit)
                msg = self.visual_encoder.load_state_dict(state_dict, strict = False)
                print("missing_keys: ", msg.missing_keys)
                print("unexpected_keys: ", msg.unexpected_keys)

        else:
            self.visual_encoder, vision_width = create_vit(
                vit, image_size, vit_grad_ckpt, vit_ckpt_layer)

        # create tokenzier
        self.tokenizer = init_tokenizer(text_encoder_type)

        self.delete_tag_index = delete_tag_index

        # load tag list
        self.tag_list = self.load_tag_list(tag_list)
        self.tag_list_chinese = self.load_tag_list(tag_list_chinese)

        # create image-tag recognition decoder
        self.threshold = threshold
        self.num_class = len(self.tag_list)
        q2l_config = BertConfig.from_json_file(f'{CONFIG_PATH}/configs/q2l_config.json')
        q2l_config.encoder_width = 512
        self.tagging_head = BertModel(config=q2l_config,
                                      add_pooling_layer=False)
        self.tagging_head.resize_token_embeddings(len(self.tokenizer))

        if stage == 'train_from_scratch':
            self.label_embed = nn.Parameter(torch.load(f'{CONFIG_PATH}/data/frozen_tag_embedding/ram_plus_tag_embedding_class_4585_des_51.pth',map_location='cpu').float())
        else:
            # when eval with pretrained RAM++ model, directly load from ram_plus_swin_large_14m.pth
            self.label_embed = nn.Parameter(torch.zeros(self.num_class * 51, q2l_config.encoder_width))

        if q2l_config.hidden_size != 512:
            self.wordvec_proj = nn.Linear(512, q2l_config.hidden_size)
        else:
            self.wordvec_proj = nn.Identity()

        self.fc = nn.Linear(q2l_config.hidden_size, 1)

        self.del_selfattention()

        self.image_proj = nn.Linear(vision_width, 512)

        # adjust thresholds for some tags
        self.class_threshold = torch.ones(self.num_class) * self.threshold
        ram_class_threshold_path = f'{CONFIG_PATH}/data/ram_tag_list_threshold.txt'
        with open(ram_class_threshold_path, 'r', encoding='utf-8') as f:
            ram_class_threshold = [float(s.strip()) for s in f]
        for key,value in enumerate(ram_class_threshold):
            self.class_threshold[key] = value

        self.reweight_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.tagging_loss_function = AsymmetricLoss(gamma_neg=7,
                                                    gamma_pos=0,
                                                    clip=0.05)

        self.text_alignment_loss_function = AsymmetricLoss(gamma_neg=4,
                                                    gamma_pos=0,
                                                    clip=0.05)


def ram_plus_size(pretrained='', **kwargs):
    model = RAM_plus_size(**kwargs)
    kwargs['image_size'] = kwargs['image_size'][0]
    if pretrained:
        if kwargs['vit'] == 'swin_b':
            model, msg = load_checkpoint_swinbase(model, pretrained, kwargs)
        elif kwargs['vit'] == 'swin_l':
            model, msg = load_checkpoint_swinlarge(model, pretrained, kwargs)
        else:
            model, msg = load_checkpoint(model, pretrained)
        print('vit:', kwargs['vit'])
#         print('msg', msg)
    return model
device = torch.device("cuda")    
# ram model

ram_model = ram_plus_size(pretrained="model_zoo/ram_plus_swin_large_14m.pth",
                             image_size=(384,768),
                             vit='swin_l').eval().to(device)

ram_transform=tvF.Compose([
    tvF.Resize(384), # 360 -> 384
    tvF.CenterCrop((384,768)),
    tvF.ToTensor(),
    tvF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# gdino model

gdino_model =gdino_inf.load_model("vinp/model/utils/gdino/GroundingDINO_SwinT_OGC.py", "model_zoo/groundingdino_swint_ogc.pth")

BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

dino_transform=tvF.Compose([
    tvF.Resize(448), # 360 -> 448
    tvF.Pad(padding=[0,0,100,0]), # 796 -> 896
    tvF.ToTensor(),
    tvF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



def annotate(n_p,i_p):
    dataset=Rrt3dFrames(n_p,i_p,ram_transform,dino_transform)
    dataloader=DataLoader(dataset,batch_size=1,shuffle=False,collate_fn=lambda x:x,num_workers=4)
    with open("tools/cates","r") as cf:
        kept_cates=cf.read().split("\n")
    kept_cates=[cate.lower() for cate in kept_cates]
    with tqdm(total=len(dataset)) as pbar:
        for data_b in dataloader:
            img_reg,img_det,vidn,imgn=data_b[0]
            img=img_reg.to(device)
            img=img.unsqueeze(0)
            cates = inference_ram(img, ram_model)[0].split(" | ")

            valid_cates=[]
            for cate in cates:
                if cate.lower() in kept_cates:
                    valid_cates.append(cate)
            cates=valid_cates
            if len(cates)>0:
                TEXT_PROMPT=" . ".join(cates)+" ."
                img=img_det.to(device)
                boxes, logits, phrases = gdino_inf.predict(
                    model=gdino_model,
                    image=img,
                    caption=TEXT_PROMPT,
                    box_threshold=BOX_TRESHOLD,
                    text_threshold=TEXT_TRESHOLD,
                    remove_combined=True
                )

                h, w, = img.shape[-2:]
                boxes = boxes * torch.Tensor([w, h, w, h])
                xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")*360/448
                xyxy=xyxy.numpy().tolist()
                labels = phrases
            else:
                xyxy=[]
                labels=[]

            if vidn not in save_annos:
                save_annos[vidn]={}
            if imgn not in save_annos[vidn]:
                save_annos[vidn][imgn]=[]
            for coord,cate in zip(xyxy,labels):
                save_annos[vidn][imgn].append([cate]+coord)
            pbar.update(1)
    for vidn,annos in save_annos.items():
        with open("{}/{}.json".format(root_dir,vidn),"w") as af:
            json.dump(annos,af)


if __name__ == "__main__":
    Fire(annotate)