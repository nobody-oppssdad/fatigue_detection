import torch
import os
import shutil
from tqdm import tqdm

def save_checkpoint(state, is_best, checkpoint_dir='checkpoints'):
    """保存模型 checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = os.path.join(checkpoint_dir, 'last.pth')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_dir, 'best.pth'))

def collate_fn(batch):
    """数据加载的批处理函数"""
    return tuple(zip(*batch))

def get_classes(root):
    """获取类别列表"""
    # 指定 encoding="utf-8" 读取文件
    with open(os.path.join(root, 'classes.txt'), 'r', encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]  # 过滤空行