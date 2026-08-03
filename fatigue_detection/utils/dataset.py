import os
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from PIL import Image
import torch


def validate_targets(targets, img_size):
    """校验并修复标注数据，防止训练中出现NaN"""
    h, w = img_size
    boxes = targets["boxes"]

    # 1. 确保boxes是torch.Tensor（防止格式异常）
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes, dtype=torch.float32)
        targets["boxes"] = boxes

    # 2. 过滤无效框（宽/高<=0的框）
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid]
    targets["boxes"] = boxes
    targets["labels"] = targets["labels"][valid]

    # 3. 限制框坐标在图像范围内（防止越界）
    boxes[:, 0] = torch.clamp(boxes[:, 0], 0, w)  # x1 >= 0
    boxes[:, 1] = torch.clamp(boxes[:, 1], 0, h)  # y1 >= 0
    boxes[:, 2] = torch.clamp(boxes[:, 2], 0, w)  # x2 <= 图像宽度
    boxes[:, 3] = torch.clamp(boxes[:, 3], 0, h)  # y2 <= 图像高度

    # 4. 空标签处理（避免无目标时损失计算异常）
    if len(boxes) == 0:
        targets["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
        targets["labels"] = torch.zeros(0, dtype=torch.int64)

    return targets


class VOCDataset(Dataset):
    def __init__(self, root, split="train", transforms=None):
        self.root = root
        self.split = split
        self.transforms = transforms

        # 1. 定义核心路径（必须存在）
        self.img_dir = os.path.join(root, "JPEGImages")
        self.anno_dir = os.path.join(root, "Annotations")
        self.split_file = os.path.join(root, "ImageSets", "Main", f"{split}.txt")

        # 2. 读取图像ID列表（容错：文件不存在则报错）
        if not os.path.exists(self.split_file):
            raise FileNotFoundError(f"分割文件不存在：{self.split_file}")
        with open(self.split_file, "r") as f:
            self.img_ids = [line.strip() for line in f if line.strip()]

        # 3. 强制生成images属性（评估脚本必须）
        self.images = []
        for img_id in self.img_ids:
            img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
            if os.path.exists(img_path):
                self.images.append(img_path)
            else:
                # 容错：跳过不存在的图像
                print(f"⚠️  图像不存在，跳过：{img_path}")

        # 4. 强制生成targets属性（评估脚本必须）
        self.targets = self._load_all_targets()

    def _load_all_targets(self):
        """预加载所有标注，生成targets列表（兼容评估脚本）"""
        targets = []
        for img_id in self.img_ids:
            anno_path = os.path.join(self.anno_dir, f"{img_id}.xml")
            if not os.path.exists(anno_path):
                # 空标注
                targets.append({"boxes": torch.empty((0, 4)), "labels": torch.empty(0, dtype=torch.int64)})
                continue
            # 解析XML
            tree = ET.parse(anno_path)
            root = tree.getroot()
            boxes = []
            labels = []
            # 类别映射（必须与get_classes()一致）
            cls_map = {"closed_eye": 1, "open_eye": 2, "closed_mouth": 3, "open_mouth": 4}

            for obj in root.findall("object"):
                cls_name = obj.find("name").text.strip()
                if cls_name not in cls_map:
                    continue
                # 提取边界框
                bndbox = obj.find("bndbox")
                x1 = float(bndbox.find("xmin").text)
                y1 = float(bndbox.find("ymin").text)
                x2 = float(bndbox.find("xmax").text)
                y2 = float(bndbox.find("ymax").text)
                boxes.append([x1, y1, x2, y2])
                labels.append(cls_map[cls_name])

            # 转换为tensor
            targets.append({
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64)
            })
        return targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. 加载原始图像和标注
        img = Image.open(self.images[idx]).convert("RGB")
        target = self.targets[idx]

        # 2. 数据校验（关键修复：放在transforms之前）
        img_size = (img.height, img.width)  # 获取原始图像尺寸 (高, 宽)
        target = validate_targets(target, img_size)  # 修复标注数据

        # 3. 执行数据增强/变换
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # 4. 返回处理后的图像和标注
        return img, target