import os
import pandas as pd
import numpy as np
from copy import deepcopy
import torch
import random
from torch.utils.data import Dataset
import torchaudio
import torch.nn.functional as F

# 雷达信号数据集根目录配置
from config import radar_root
from PIL import Image

# -------------------------------
# STFT处理（方案②：生成正方形时频图）
# -------------------------------
def compute_spectrogram(signal, target_size=224):
    """
    对输入一维信号进行STFT转换，生成正方形时频图
    Args:
        signal: 1D Tensor, shape: (T,)
        target_size: 目标正方形尺寸 (H=W=target_size)
    """
    # 计算满足正方形时频图的STFT参数
    n_fft = 2 * (target_size - 1)  # 确保频率维度为target_size
    win_length = n_fft              # 窗长等于n_fft以获得最佳频率分辨率
    signal_length = signal.shape[0]
    
    # 计算所需的hop_length以确保时间维度为target_size
    # 公式: time_steps = (signal_length - win_length) // hop_length + 1 = target_size
    # 推导得: hop_length = (signal_length - win_length) // (target_size - 1)
    hop_length = max(1, (signal_length - win_length) // (target_size - 1))
    
    # 信号长度调整（确保能生成完整的target_size时间步）
    required_length = win_length + (target_size - 1) * hop_length
    if signal_length < required_length:
        pad_size = required_length - signal_length
        signal = torch.cat([signal, torch.zeros(pad_size, device=signal.device)])
    else:
        signal = signal[:required_length]
    
    # 添加批次维度
    if signal.dim() == 1:
        signal = signal.unsqueeze(0)
    
    # 计算STFT
    window = torch.hann_window(win_length, device=signal.device)
    spec = torch.stft(
        signal, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        win_length=win_length,
        window=window, 
        return_complex=True
    )
    
    # 转换为幅度谱并进行log处理
    spec = torch.abs(spec)
    spec = torch.log1p(spec)  # shape: (1, target_size, target_size)
    
    # 扩展为3通道以适应ViT输入（复制单通道到3个通道）
    spec = spec.repeat(3, 1, 1)  # shape: (3, target_size, target_size)
    return spec


class CustomRadarDataset(Dataset):
    """雷达信号数据集，适配广义类别发现任务"""
    def __init__(self, root, train=True, transform=None, target_transform=None, 
                 download=False, selected_labels=None, augmentation=True, target_size=224):
        self.root = os.path.expanduser(root)
        self.transform = transform  # 预留的外部变换接口
        self.target_transform = target_transform
        self.train = train
        self.augmentation = augmentation
        self.target_size = target_size
        self.selected_labels = selected_labels
        
        # 数据增强工具
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=15)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
        
        # 如果需要下载数据，可以在这里实现
        if download:
            self._download()
            
        # 加载元数据
        self._load_metadata()
        
        # 唯一索引
        self.uq_idxs = np.array(range(len(self)))

    def _load_metadata(self):
        """加载数据集元数据，包括文件路径和标签"""
        # 确定CSV文件路径
        if self.train:
            csv_path = os.path.join(self.root+'dataset/DC_comp/individual_recognition_split/', 'train_in4model.csv')
        else:
            csv_path = os.path.join(self.root+'dataset/DC_comp/individual_recognition_split/', 'test_in4model.csv')
            
        # 读取CSV文件，格式为：文件路径, 标签（无表头）
        data = pd.read_csv(csv_path, header=None)
        
        # 筛选选定的类别
        if self.selected_labels is not None:
            data = data[data.iloc[:, 1].isin(self.selected_labels)].reset_index(drop=True)
        
        # 提取文件路径和标签
        self.file_paths = data.iloc[1:, 0].tolist()
        self.labels = data.iloc[1:, 1].astype(int).to_numpy()

    def _check_integrity(self):
        """检查数据集是否完整"""
        try:
            for path in self.file_paths:
                filepath = os.path.join(self.root, path)
                if not os.path.isfile(filepath):
                    print(f"缺失文件: {filepath}")
                    return False
            return True
        except Exception:
            return False

    def _download(self):
        """下载数据集（根据实际情况实现）"""
        if self._check_integrity():
            print('数据集已存在且完整')
            return
        
        # 这里添加实际的下载逻辑
        print("数据集下载功能未实现，请手动放置数据集到指定目录")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """获取样本：信号时频图、标签和唯一索引"""
        # 读取信号文件
        file_path = os.path.join(self.root, self.file_paths[idx])
        with open(file_path, 'r', encoding='utf-8') as file:
            data_final = []
            for line in file.read().split("\n"):
                if line.strip():  # 跳过空行
                    data_final.append(float(line.strip()))
        
        # 转换为Tensor并计算时频图
        signal = torch.from_numpy(np.array(data_final)).float()
        spec = compute_spectrogram(signal, self.target_size)
        
        # 数据增强
        spec_mask = spec
        # if self.augmentation:
        #     # 转换为适合mask的形状 (1, H, W)
        #     spec_mask = spec.unsqueeze(0)
        #     if random.random() < 0.5:
        #         spec_mask = self.time_mask(spec_mask)
        #     if random.random() < 0.5:
        #         spec_mask = self.freq_mask(spec_mask)
        #     spec_mask = spec_mask.squeeze(0)  # 移除批次维度，形状变为 (C, H, W) 或 (H, W)
            
        # 转换为numpy数组
        spec_mask = spec_mask.numpy()
        
        # 处理维度
        if spec_mask.ndim == 3:  # 有通道维度 (C, H, W)
            # 转为 (H, W, C)
            spec_mask = spec_mask.transpose(1, 2, 0)
        elif spec_mask.ndim == 2:  # 无通道维度 (H, W)，保持不变
            pass
        else:
            raise ValueError(f"不支持的数组维度: {spec_mask.ndim}")
    
        # 归一化并转换为PIL图像
        spec_mask = (spec_mask - spec_mask.min()) / (spec_mask.max() - spec_mask.min() + 1e-8) * 255
        spec = Image.fromarray(spec_mask.astype(np.uint8))
        
        # 应用外部变换（如果有）
        if self.transform is not None:
            spec = self.transform(spec)
        
        # 处理标签
        target = self.labels[idx]
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return spec, target, self.uq_idxs[idx]


def subsample_dataset(dataset, idxs):
    """对数据集进行子采样"""
    mask_len = len(dataset)
    mask = np.zeros(mask_len, dtype=bool)
    mask[idxs] = True
    
    # 对文件路径和标签进行子采样
    dataset.file_paths = [dataset.file_paths[i] for i in range(mask_len) if mask[i]]
    dataset.labels = dataset.labels[mask]
    dataset.uq_idxs = dataset.uq_idxs[mask]
    
    return dataset


def subsample_classes(dataset, include_classes):
    """只保留指定的类别"""
    # 筛选包含指定类别的样本索引
    cls_idxs = [i for i, label in enumerate(dataset.labels) if label in include_classes]
    
    # 构建标签映射（将保留的类别重新编号）
    target_xform_dict = {cls: i for i, cls in enumerate(include_classes)}
    
    # 子采样数据集
    dataset = subsample_dataset(dataset, cls_idxs)
    
    # 设置标签转换
    original_target_transform = dataset.target_transform
    
    def new_target_transform(x):
        x = target_xform_dict[x]
        if original_target_transform is not None:
            x = original_target_transform(x)
        return x
    
    dataset.target_transform = new_target_transform
    
    return dataset


def subsample_instances(dataset, prop_indices_to_subsample=0.8):
    """对每个类别随机采样一定比例的样本"""
    np.random.seed(42)
    all_idxs = []
    
    # 对每个类别进行采样
    for cls in np.unique(dataset.labels):
        cls_idxs = np.where(dataset.labels == cls)[0]
        num_to_subsample = int(prop_indices_to_subsample * len(cls_idxs))
        if num_to_subsample == 0:  # 确保至少保留一个样本
            num_to_subsample = 1
        subsampled_idxs = np.random.choice(cls_idxs, num_to_subsample, replace=False)
        all_idxs.extend(subsampled_idxs)
    
    return all_idxs


def get_train_val_indices(train_dataset, val_split=0.2):
    """将训练集划分为训练和验证集（按类别保持比例）"""
    np.random.seed(42)
    train_idxs = []
    val_idxs = []
    
    # 对每个类别进行划分
    for cls in np.unique(train_dataset.labels):
        cls_idxs = np.where(train_dataset.labels == cls)[0]
        val_size = int(val_split * len(cls_idxs))
        
        # 随机选择验证集索引
        val_idxs_cls = np.random.choice(cls_idxs, val_size, replace=False)
        train_idxs_cls = [idx for idx in cls_idxs if idx not in val_idxs_cls]
        
        train_idxs.extend(train_idxs_cls)
        val_idxs.extend(val_idxs_cls)
    
    return train_idxs, val_idxs


def get_radar_datasets(train_transform, test_transform, train_classes=range(24), 
                       prop_train_labels=0.8, split_train_val=False, seed=42, target_size=224):
    """获取用于广义类别发现任务的雷达数据集"""
    np.random.seed(seed)
    random.seed(seed)
    
    # 初始化完整训练集
    whole_training_set = CustomRadarDataset(
        root=radar_root, 
        train=True, 
        transform=train_transform,
        download=False,
        selected_labels=None,  # 先不筛选，后续统一处理
        augmentation=True,
        target_size=target_size
    )
    
    # 获取带标签的训练集（只包含指定类别）
    train_dataset_labelled = subsample_classes(
        deepcopy(whole_training_set), 
        include_classes=train_classes
    )
    # 保持增强开启
    train_dataset_labelled.augmentation = True
    
    # 对带标签数据集进行子采样
    subsample_indices = subsample_instances(
        train_dataset_labelled, 
        prop_indices_to_subsample=prop_train_labels
    )
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)
    
    # 划分训练集和验证集
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.augmentation = False  # 验证集关闭增强
    val_dataset_labelled_split.transform = test_transform
    
    # 获取无标签数据集（训练集中不在带标签数据集中的样本）
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(
        deepcopy(whole_training_set), 
        np.array(list(unlabelled_indices))
    )
    # 无标签数据可以保留增强
    train_dataset_unlabelled.augmentation = True
    
    # 获取测试集
    test_dataset = CustomRadarDataset(
        root=radar_root, 
        train=False, 
        transform=test_transform,
        download=False,
        selected_labels=None,
        augmentation=False,
        target_size=target_size
    )
    
    # 根据是否需要划分，设置训练集和验证集
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None
    
    return {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
    }
    