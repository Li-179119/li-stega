import numpy as np
from PIL import Image
from typing import List, Optional
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

def segment_image_with_sam(mask_generator, image_path: str, palette: List[np.ndarray],
                           crop_n_points_downscale_factor: int = 1,
                           point_grids: Optional[List[np.ndarray]] = None,
                           min_mask_region_area: int = 0,
                           output_mode: str = "binary_mask") -> np.ndarray:
    """
    使用 SAM 模型对给定路径的图片进行分割，并返回颜色分割后的图片数组。

    :param mask_generator: SAM 模型的掩码生成器
    :param image_path: 图片路径
    :param palette: 颜色调色板
    :param crop_n_points_downscale_factor: 裁剪点的缩放因子
    :param point_grids: 点网格（可选）
    :param min_mask_region_area: 最小掩码区域面积
    :param output_mode: 输出模式
    :return: 颜色分割后的图片数组
    """

    # 读入图片
    image = Image.open(image_path).convert("RGB")  # 自动加载为 RGB
    image_np = np.array(image)  # 直接转为 NumPy 数组（RGB）

    # 图像预处理（这里假设不需要额外的预处理步骤）

    # 调用 SAM 进行分割
    masks = mask_generator.generate(image_np)

    color_seg = np.zeros((image_np.shape[0], image_np.shape[1], 3), dtype=np.uint8)  # height, width, 3
    for i, mask in enumerate(masks):
        # 假设每个掩码对应一个唯一的 label
        label = i % len(palette)  # 避免 label 超出调色板长度
        # 填充颜色
        color_seg[mask["segmentation"]] = palette[label]
    mask_image = Image.fromarray(color_seg.astype(np.uint8), mode='RGB')
    return mask_image
