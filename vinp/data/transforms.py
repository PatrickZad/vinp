from typing import List, Optional, Sequence, Tuple, Union
import random
import math
from functools import reduce
import numbers
import warnings

import PIL.Image as pilimg
import numpy as np
import cv2

class RandomAffine:
    """Random affine transform data augmentation.

    This operation randomly generates affine transform matrix which including
    rotation, translation, shear and scaling transforms.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)

    Args:
        max_rotate_degree (float): Maximum degrees of rotation transform.
            Defaults to 10.
        max_translate_ratio (float): Maximum ratio of translation.
            Defaults to 0.1.
        scaling_ratio_range (tuple[float]): Min and max ratio of
            scaling transform. Defaults to (0.5, 1.5).
        max_shear_degree (float): Maximum degrees of shear
            transform. Defaults to 2.
        border (tuple[int]): Distance from width and height sides of input
            image to adjust output shape. Only used in mosaic dataset.
            Defaults to (0, 0).
        border_val (tuple[int]): Border padding values of 3 channels.
            Defaults to (114, 114, 114).
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
    """

    def __init__(
        self,
        max_rotate_degree: float = 10.0,
        max_translate_ratio: float = 0.1,
        scaling_ratio_range: Tuple[float, float] = (0.5, 1.5),
        max_shear_degree: float = 2.0,
        border: Tuple[int, int] = (0, 0),
        border_val: Tuple[int, int, int] = (114, 114, 114),
        bbox_clip_border: bool = True,
    ) -> None:
        assert 0 <= max_translate_ratio <= 1
        assert scaling_ratio_range[0] <= scaling_ratio_range[1]
        assert scaling_ratio_range[0] > 0
        self.max_rotate_degree = max_rotate_degree
        self.max_translate_ratio = max_translate_ratio
        self.scaling_ratio_range = scaling_ratio_range
        self.max_shear_degree = max_shear_degree
        self.border = border
        self.border_val = border_val
        self.bbox_clip_border = bbox_clip_border

    def _get_random_homography_matrix(self, height, width):
        # Rotation
        rotation_degree = random.uniform(
            -self.max_rotate_degree, self.max_rotate_degree
        )
        rotation_matrix = self._get_rotation_matrix(rotation_degree)

        # Scaling
        scaling_ratio = random.uniform(
            self.scaling_ratio_range[0], self.scaling_ratio_range[1]
        )
        scaling_matrix = self._get_scaling_matrix(scaling_ratio)

        # Shear
        x_degree = random.uniform(-self.max_shear_degree, self.max_shear_degree)
        y_degree = random.uniform(-self.max_shear_degree, self.max_shear_degree)
        shear_matrix = self._get_shear_matrix(x_degree, y_degree)

        # Translation
        trans_x = (
            random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * width
        )
        trans_y = (
            random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * height
        )
        translate_matrix = self._get_translation_matrix(trans_x, trans_y)

        warp_matrix = translate_matrix @ shear_matrix @ rotation_matrix @ scaling_matrix
        return warp_matrix

    def __call__(self, img,bboxes,labels) -> dict:
        convert_pillow=False
        if isinstance(img,pilimg.Image):
            img=np.array(img).astype(np.uint8)
            img=img[::-1].copy()
            convert_pillow=True
        height = img.shape[0] + self.border[1] * 2
        width = img.shape[1] + self.border[0] * 2

        warp_matrix = self._get_random_homography_matrix(height, width)

        img = cv2.warpPerspective(
            img, warp_matrix, dsize=(width, height), borderValue=self.border_val
        )

        num_bboxes = len(bboxes)
        if num_bboxes:
            bboxes.project_(warp_matrix)
            if self.bbox_clip_border:
                bboxes.clip_([height, width])
            # remove outside bbox
            valid_index = bboxes.is_inside([height, width]).numpy()
            bboxes=bboxes[valid_index]
            labels=labels[valid_index]
        
        if convert_pillow:
            img=pilimg.fromarray(img[::-1].copy())
        
        return img,bboxes,labels

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(max_rotate_degree={self.max_rotate_degree}, "
        repr_str += f"max_translate_ratio={self.max_translate_ratio}, "
        repr_str += f"scaling_ratio_range={self.scaling_ratio_range}, "
        repr_str += f"max_shear_degree={self.max_shear_degree}, "
        repr_str += f"border={self.border}, "
        repr_str += f"border_val={self.border_val}, "
        repr_str += f"bbox_clip_border={self.bbox_clip_border})"
        return repr_str

    @staticmethod
    def _get_rotation_matrix(rotate_degrees: float) -> np.ndarray:
        radian = math.radians(rotate_degrees)
        rotation_matrix = np.array(
            [
                [np.cos(radian), -np.sin(radian), 0.0],
                [np.sin(radian), np.cos(radian), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return rotation_matrix

    @staticmethod
    def _get_scaling_matrix(scale_ratio: float) -> np.ndarray:
        scaling_matrix = np.array(
            [[scale_ratio, 0.0, 0.0], [0.0, scale_ratio, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        return scaling_matrix

    @staticmethod
    def _get_shear_matrix(x_shear_degrees: float, y_shear_degrees: float) -> np.ndarray:
        x_radian = math.radians(x_shear_degrees)
        y_radian = math.radians(y_shear_degrees)
        shear_matrix = np.array(
            [[1, np.tan(x_radian), 0.0], [np.tan(y_radian), 1, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        return shear_matrix

    @staticmethod
    def _get_translation_matrix(x: float, y: float) -> np.ndarray:
        translation_matrix = np.array(
            [[1, 0.0, x], [0.0, 1, y], [0.0, 0.0, 1.0]], dtype=np.float32
        )
        return translation_matrix
    @staticmethod
    def box_project(self, boxes, homography_matrix):
        corners = self._hbox2corner(boxes)
        corners = np.concatenate(
            [corners, np.ones((*corners.shape[:-1], 1))], axis=-1)
        corners_T = np.transpose(corners, -1, -2)
        corners_T = np.matmul(homography_matrix, corners_T)
        corners = np.transpose(corners_T, -1, -2)
        # Convert to homogeneous coordinates by normalization
        corners = corners[..., :2] / corners[..., 2:3]
        return self._corner2hbox(corners)
    @staticmethod
    def _hbox2corner(boxes):
        """Convert box coordinates from (x1, y1, x2, y2) to corners ((x1, y1),
        (x2, y1), (x1, y2), (x2, y2)).

        Args:
            boxes: Horizontal box array with shape of (..., 4).

        Returns:
            Tensor: Corner array with shape of (..., 4, 2).
        """
        x1, y1, x2, y2 = np.split(boxes, 1, axis=-1)
        corners = np.concatenate([x1, y1, x2, y1, x1, y2, x2, y2], axis=-1)
        return corners.reshape((*corners.shape[:-1],4,2))
    
    @staticmethod
    def _corner2hbox(corners):
        """Convert box coordinates from corners ((x1, y1), (x2, y1), (x1, y2),
        (x2, y2)) to (x1, y1, x2, y2).

        Args:
            corners: Corner tensor with shape of (..., 4, 2).

        Returns:
            Horizontal box with shape of (..., 4).
        """
        if corners.size == 0:
            return np.zeros((0, 4))
        min_xy = corners.min(axis=-2)[0]
        max_xy = corners.max(axis=-2)[0]
        return np.concatenate([min_xy, max_xy], axis=-1)


class MixUp:
    """MixUp data augmentation.

    .. code:: text

                         mixup transform
                +------------------------------+
                | mixup image   |              |
                |      +--------|--------+     |
                |      |        |        |     |
                |---------------+        |     |
                |      |                 |     |
                |      |      image      |     |
                |      |                 |     |
                |      |                 |     |
                |      |-----------------+     |
                |             pad              |
                +------------------------------+

     The mixup transform steps are as follows:

        1. Another random image is picked by dataset and embedded in
           the top left patch(after padding and resizing)
        2. The target of mixup transform is the weighted average of mixup
           image and origin image.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - mix_results (List[dict])


    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)


    Args:
        img_scale (Sequence[int]): Image output size after mixup pipeline.
            The shape order should be (width, height). Defaults to (640, 640).
        ratio_range (Sequence[float]): Scale ratio of mixup image.
            Defaults to (0.5, 1.5).
        flip_ratio (float): Horizontal flip ratio of mixup image.
            Defaults to 0.5.
        pad_val (int): Pad value. Defaults to 114.
        max_iters (int): The maximum number of iterations. If the number of
            iterations is greater than `max_iters`, but gt_bbox is still
            empty, then the iteration is terminated. Defaults to 15.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
    """

    def __init__(
        self,
        img_scale: Tuple[int, int] = (640, 360),
        ratio_range: Tuple[float, float] = (0.5, 1.5),
        flip_ratio: float = 0.5,
        pad_val: float = 114.0,
        max_iters: int = 15,
        bbox_clip_border: bool = True,
    ) -> None:
        assert isinstance(img_scale, tuple)
        self.dynamic_scale = img_scale
        self.ratio_range = ratio_range
        self.flip_ratio = flip_ratio
        self.pad_val = pad_val
        self.max_iters = max_iters
        self.bbox_clip_border = bbox_clip_border


    def __call__(self,img,bboxes,labels,mixup_img,mixup_bboxes,mixup_labels) -> dict:
        """MixUp transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        convert_pillow=False
        if isinstance(img,pilimg.Image):
            img=np.array(img).astype(np.uint8)
            img=img[::-1].copy()
            mixup_img=np.array(mixup_img).astype(np.uint8)
            mixup_img=mixup_img[::-1].copy()
            convert_pillow=True

        retrieve_img = mixup_img

        jit_factor = random.uniform(*self.ratio_range)
        is_flip = random.uniform(0, 1) > self.flip_ratio

        out_img = (
                np.ones(
                    (self.dynamic_scale[1], self.dynamic_scale[0], 3),
                    dtype=retrieve_img.dtype,
                )
                * self.pad_val
            )

        # 1. keep_ratio resize
        scale_ratio = min(
            self.dynamic_scale[1] / retrieve_img.shape[0],
            self.dynamic_scale[0] / retrieve_img.shape[1],
        )
        retrieve_img = cv2.resize(
            retrieve_img,
            (
                int(retrieve_img.shape[1] * scale_ratio),
                int(retrieve_img.shape[0] * scale_ratio),
            ),
        )

        # 2. paste
        out_img[: retrieve_img.shape[0], : retrieve_img.shape[1]] = retrieve_img

        # 3. scale jit
        scale_ratio *= jit_factor
        out_img = cv2.resize(
            out_img,
            (int(out_img.shape[1] * jit_factor), int(out_img.shape[0] * jit_factor)),
        )

        # 4. flip
        if is_flip:
            out_img = out_img[:, ::-1, :]

        # 5. random crop
        ori_img = img
        origin_h, origin_w = out_img.shape[:2]
        target_h, target_w = ori_img.shape[:2]
        padded_img = (
            np.ones((max(origin_h, target_h), max(origin_w, target_w), 3))
            * self.pad_val
        )
        padded_img = padded_img.astype(np.uint8)
        padded_img[:origin_h, :origin_w] = out_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w)
        padded_cropped_img = padded_img[
            y_offset : y_offset + target_h, x_offset : x_offset + target_w
        ]

        # 6. adjust bbox
        retrieve_gt_bboxes = mixup_bboxes
        retrieve_gt_bboxes=self._box_rescale(retrieve_gt_bboxes,[scale_ratio, scale_ratio])
        if self.bbox_clip_border:
            retrieve_gt_bboxes=self._box_clip(retrieve_gt_bboxes,[origin_h, origin_w])

        if is_flip:
            retrieve_gt_bboxes=self._box_flip(retrieve_gt_bboxes,[origin_h, origin_w], direction="h")

        # 7. filter
        cp_retrieve_gt_bboxes = retrieve_gt_bboxes.copy()
        cp_retrieve_gt_bboxes=self._box_translate_(cp_retrieve_gt_bboxes,[-x_offset, -y_offset])
        if self.bbox_clip_border:
            cp_retrieve_gt_bboxes=self._box_clip(cp_retrieve_gt_bboxes,[target_h, target_w])

        # 8. mix up
        ori_img = ori_img.astype(np.float32)
        mixup_img = 0.5 * ori_img + 0.5 * padded_cropped_img.astype(np.float32)

        retrieve_gt_bboxes_labels = mixup_labels

        # cat labels and boxes

        mixup_gt_bboxes =np.concatenate([cp_retrieve_gt_bboxes,bboxes],axis=0)
        mixup_gt_bboxes_labels = np.concatenate(
            [labels, retrieve_gt_bboxes_labels], axis=0
        )

        # remove outside bbox
        inside_mask = self._box_inside(mixup_gt_bboxes,[target_h, target_w])
        mixup_gt_bboxes = mixup_gt_bboxes[inside_mask]
        mixup_gt_bboxes_labels = mixup_gt_bboxes_labels[inside_mask]

        if convert_pillow:
            mixup_img=pilimg.fromarray(mixup_img[::-1].copy())

        return mixup_img,mixup_gt_bboxes,mixup_gt_bboxes_labels

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(dynamic_scale={self.dynamic_scale}, "
        repr_str += f"ratio_range={self.ratio_range}, "
        repr_str += f"flip_ratio={self.flip_ratio}, "
        repr_str += f"pad_val={self.pad_val}, "
        repr_str += f"max_iters={self.max_iters}, "
        repr_str += f"bbox_clip_border={self.bbox_clip_border})"
        return repr_str
    @staticmethod
    def _box_rescale(bboxes,scales):
        scales=np.array([[scales[0],scales[1],scales[0],scales[1]]])
        return bboxes*scales
    @staticmethod
    def _box_clip(bboxes,max_hw):
        max_values=np.array([[max_hw[0],max_hw[1],max_hw[0],max_hw[1]]])
        min_values=np.zeros_like(max_values)
        return np.clip(bboxes,a_min=min_values,a_max=max_values)
    @staticmethod
    def _box_translate(bboxes,xy_offsets):
        offset_values=np.array([[xy_offsets[0],xy_offsets[1],xy_offsets[0],xy_offsets[1]]])
        return bboxes+offset_values
    @staticmethod
    def _box_flip(bboxes,img_hw,direction="h"):
        assert direction in ["h","v"]
        if direction == "h":
            new_boxes=np.stack([img_hw[1]-bboxes[:,0],bboxes[:,1],img_hw[1]-bboxes[:,2],bboxes[:,3]],axis=1,dtype=bboxes.dtype)
        else:
            new_boxes=np.stack([bboxes[:,0],img_hw[0]-bboxes[:,1],bboxes[:,2],img_hw[0]-bboxes[:,3]],axis=1,dtype=bboxes.dtype)
        return new_boxes
    @staticmethod
    def _box_inside(bboxes,img_hw):
        masks=[bboxes[:,0]<img_hw[1],bboxes[:,1]<img_hw[0],bboxes[:,2]<img_hw[1],bboxes[:,3]<img_hw[0]]
        return reduce(np.logical_and,masks)




class RandomInstanceErasing:
    """Randomly selects a rectangle region in a person and erases its pixels.
    This transform does not support PIL Image.
    'Random Erasing Data Augmentation' by Zhong et al. See https://arxiv.org/abs/1708.04896

    Args:
         p: probability that the random erasing operation will be performed.
         scale: range of proportion of erased area against bounding box.
         ratio: range of aspect ratio of erased area.
         value: erasing value. Default is 0. If a single int, it is used to
            erase all pixels. If a tuple of length 3, it is used to erase
            R, G, B channels respectively.
            If a str of 'random', erasing each pixel with random values.

    Returns:
        Erased Image.

    """

    def __init__(
        self,
        p=0.5,
        scale=(0.02, 0.33),
        ratio=(0.3, 3.3),
        value=[123.675,116.28,103.53 ], # NOTE RGB
    ):
        if not isinstance(value, (numbers.Number, str, tuple, list)):
            raise TypeError(
                "Argument value should be either a number or str or a sequence"
            )
        if isinstance(value, str) and value != "random":
            raise ValueError("If value is str, it should be 'random'")
        if not isinstance(scale, (tuple, list)):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, (tuple, list)):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("Scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("Random erasing probability should be between 0 and 1")
        self.p = p
        self.scale = np.array(scale)
        self.ratio = np.array(ratio)
        self.value = np.array(value)

    def get_params(self, box, channels, scale, ratio, value=None):
        """Get parameters for ``erase`` for a random erasing."""
        box = np.around(box).astype(np.int32)
        img_c, img_h, img_w = channels, box[3] - box[1], box[2] - box[0]
        area = img_h * img_w

        log_ratio = np.log(ratio)
        for _ in range(10):

            erase_area = (
                area * (np.random.rand() * (scale[1] - scale[0]) + scale[0]).item()
            )
            aspect_ratio = np.exp(
                (np.random.rand() * (log_ratio[1] - log_ratio[0]) + log_ratio[0])
            ).item()

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            if not (h < img_h and w < img_w):
                continue

            if value is None:
                v = np.random.randn(img_c, h, w)
            else:
                v = value.view(img_c, 1, 1)

            i = np.random.randint(0, img_w - w + 1)
            j = np.random.randint(0, img_h - h + 1)
            return i + box[0], j + box[1], h, w, v

        # Return original image
        return box[0], box[1], 0, 0, 0

    def __call__(self, img, boxes):
        convert_pillow=False
        if isinstance(img,pilimg.Image):
            img=np.array(img).astype(np.uint8)
            img=img[::-1].copy()
            convert_pillow=True
        vals = np.random.rand(boxes.shape[0])
        for i in range(boxes.shape[0]):
            if vals[i] < self.p:
                x, y, h, w, v = self.get_params(
                    boxes[i], 3, scale=self.scale, ratio=self.ratio, value=self.value
                )
                self.erase(img, x, y, h, w, v)
        if convert_pillow:
            img=pilimg.fromarray(img[::-1].copy())
        return img

    def erase(self, img, i, j, h, w, v):

        img[j : j + h, i : i + w] = v

