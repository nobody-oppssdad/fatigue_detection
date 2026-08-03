import torch
from PIL import Image
import numpy as np


# 自定义Compose（支持同时处理图像和标注）
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


# 自定义Resize（适配640×480原始尺寸，同步缩放标注框，防止越界）
class Resize(object):
    def __init__(self, size=(480, 640)):  # 改为原始图像尺寸 (h, w) = (480, 640)
        self.size = size  # (height, width) 对应480×640

    def __call__(self, image, target):
        # 处理PIL Image（主要场景）
        if isinstance(image, Image.Image):
            orig_width, orig_height = image.size  # PIL: (宽, 高) → 640, 480
            # 缩放图像（保持原始比例，无拉伸）
            image = image.resize((self.size[1], self.size[0]), Image.BILINEAR)
            new_width, new_height = self.size[1], self.size[0]
        # 处理Tensor（备用）
        elif isinstance(image, torch.Tensor):
            orig_height, orig_width = image.shape[1], image.shape[2]  # Tensor: (C, H, W)
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False
            ).squeeze(0)
            new_height, new_width = self.size[0], self.size[1]
        else:
            raise TypeError(f"不支持的图像类型：{type(image)}")

        # 缩放标注框（核心：同步调整坐标+防止越界）
        if target is not None and len(target["boxes"]) > 0:
            # 计算缩放比例
            scale_x = new_width / orig_width
            scale_y = new_height / orig_height

            # 调整每个标注框的坐标
            boxes = target["boxes"].cpu().numpy()
            boxes[:, 0] = boxes[:, 0] * scale_x  # x1
            boxes[:, 1] = boxes[:, 1] * scale_y  # y1
            boxes[:, 2] = boxes[:, 2] * scale_x  # x2
            boxes[:, 3] = boxes[:, 3] * scale_y  # y2

            # 防止标注框越界（关键修复）
            boxes[:, 0] = np.clip(boxes[:, 0], 0, new_width)  # x1 ≥ 0
            boxes[:, 1] = np.clip(boxes[:, 1], 0, new_height)  # y1 ≥ 0
            boxes[:, 2] = np.clip(boxes[:, 2], 0, new_width)  # x2 ≤ 宽度
            boxes[:, 3] = np.clip(boxes[:, 3], 0, new_height)  # y2 ≤ 高度

            # 确保x2 > x1, y2 > y1（避免无效框）
            boxes[:, 2] = np.maximum(boxes[:, 2], boxes[:, 0] + 1)
            boxes[:, 3] = np.maximum(boxes[:, 3], boxes[:, 1] + 1)

            target["boxes"] = torch.tensor(boxes, dtype=torch.float32)

        return image, target


# 自定义ToTensor（转换PIL Image为Tensor，标准化到0-1）
class ToTensor(object):
    def __call__(self, image, target):
        if isinstance(image, Image.Image):
            # PIL Image转Tensor: (H, W, C) → (C, H, W)，并归一化到0-1
            image_np = np.array(image, dtype=np.float32) / 255.0
            image = torch.from_numpy(image_np).permute(2, 0, 1)
        return image, target


# 自定义RandomHorizontalFlip（训练增强，同步翻转标注框）
class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if np.random.random() < self.prob:
            # 翻转图像
            if isinstance(image, Image.Image):
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                width = image.size[0]
            elif isinstance(image, torch.Tensor):
                image = torch.flip(image, dims=[2])
                width = image.shape[2]

            # 翻转标注框
            if target is not None and len(target["boxes"]) > 0:
                boxes = target["boxes"].cpu().numpy()
                # 水平翻转公式：x1' = 宽度 - x2, x2' = 宽度 - x1
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                # 再次检查越界
                boxes[:, 0] = np.clip(boxes[:, 0], 0, width)
                boxes[:, 2] = np.clip(boxes[:, 2], 0, width)
                target["boxes"] = torch.tensor(boxes, dtype=torch.float32)

        return image, target


# 新增：随机框偏移增强（提升定位鲁棒性）
class RandomBoxOffset(object):
    def __init__(self, offset_range=5):
        """
        随机偏移标注框坐标
        :param offset_range: 最大偏移像素（正负）
        """
        self.offset_range = offset_range

    def __call__(self, image, target):
        if target is not None and len(target["boxes"]) > 0:
            # 获取图像尺寸
            if isinstance(image, Image.Image):
                img_w, img_h = image.size
            elif isinstance(image, torch.Tensor):
                img_h, img_w = image.shape[1], image.shape[2]
            else:
                return image, target

            # 生成随机偏移量（每个框的四个坐标都有随机偏移）
            boxes = target["boxes"].cpu().numpy()
            offsets = np.random.randint(
                -self.offset_range, self.offset_range + 1,
                size=boxes.shape
            )
            boxes += offsets

            # 防止偏移后越界
            boxes[:, 0] = np.clip(boxes[:, 0], 0, img_w)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, img_h)
            boxes[:, 2] = np.clip(boxes[:, 2], 0, img_w)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, img_h)

            # 确保框的有效性
            boxes[:, 2] = np.maximum(boxes[:, 2], boxes[:, 0] + 1)
            boxes[:, 3] = np.maximum(boxes[:, 3], boxes[:, 1] + 1)

            target["boxes"] = torch.tensor(boxes, dtype=torch.float32)

        return image, target


# 训练集预处理（含数据增强）
def get_train_transform():
    return Compose([
        Resize((480, 640)),  # 匹配原始图像尺寸，无拉伸
        RandomHorizontalFlip(0.5),  # 水平翻转
        RandomBoxOffset(offset_range=3),  # 小幅度框偏移，提升定位鲁棒性
        ToTensor()  # 最后转Tensor
    ])


# 验证集预处理（无增强，保持和训练一致的Resize）
def get_val_transform():
    return Compose([
        Resize((480, 640)),  # 与训练集尺寸完全一致
        ToTensor()
    ])

