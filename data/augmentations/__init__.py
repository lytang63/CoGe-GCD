from torchvision import transforms
from data.augmentations.cut_out import *
from data.augmentations.randaugment import RandAugment

# 修正版：兼容 PIL.Image / numpy / torch.Tensor 的 ToTensorFloat，并修正命名冲突
from PIL import Image
import numpy as np
import random
import torch
import torch.nn.functional as F  # 保留为 F 用于 interpolate / pad
from torchvision import transforms
from torchvision.transforms.functional import pil_to_tensor, to_tensor
import torchaudio


# ---------- Safe ToTensorFloat ----------
class ToTensorFloat:
    """
    Accepts: PIL.Image, numpy.ndarray, or torch.Tensor.
    Returns tensor of shape (C, F, T) with dtype float32.
    - If input is RGB/3-channel -> convert to single channel by mean().
    - If input is 2D -> returns (1, H, W).
    - If input is already (C,H,W) with C==1 -> keep.
    """
    def __call__(self, x):
        # PIL Image
        if isinstance(x, Image.Image):
            # pil_to_tensor returns uint8 tensor with shape (C, H, W)
            t = pil_to_tensor(x).float() / 255.0
            # convert multi-channel -> mono by mean
            if t.shape[0] > 1:
                t = t.mean(dim=0, keepdim=True)   # (1, H, W)
            x = t
        # torch.Tensor
        elif isinstance(x, torch.Tensor):
            x = x.float()
            # If channels-last e.g. (H, W, C)
            if x.dim() == 3 and x.shape[2] in (1, 3, 4):
                x = x.permute(2, 0, 1)  # -> (C,H,W)
            # If (C,H,W) with C>1 -> average to 1 channel
            if x.dim() == 3 and x.shape[0] > 1:
                x = x.mean(dim=0, keepdim=True)
            if x.dim() == 2:
                x = x.unsqueeze(0)
        # numpy array
        elif isinstance(x, np.ndarray):
            if x.ndim == 2:
                x = torch.from_numpy(x).float()
                x = x.unsqueeze(0)
            elif x.ndim == 3:
                # assume HWC
                if x.shape[2] in (1, 3, 4):
                    t = torch.from_numpy(x).permute(2, 0, 1).float()
                    t = t / 255.0 if t.max() > 1.0 else t
                    if t.shape[0] > 1:
                        t = t.mean(dim=0, keepdim=True)
                    x = t
                else:
                    # assume CHW already
                    x = torch.from_numpy(x).float()
                    if x.shape[0] > 1:
                        x = x.mean(dim=0, keepdim=True)
            else:
                x = torch.from_numpy(x).float()
                if x.dim() == 2:
                    x = x.unsqueeze(0)
        else:
            # fallback to torchvision.to_tensor which handles several types
            x = to_tensor(x).float()
            if x.dim() == 3 and x.shape[0] > 1:
                x = x.mean(dim=0, keepdim=True)

        # ensure final shape is (C, H, W)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        return x


# ---------- 修正后的辅助变换（注意避免覆盖 F） ----------
class MinMaxNormalize:
    def __call__(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        minv = x.amin(dim=(-2, -1), keepdim=True)
        maxv = x.amax(dim=(-2, -1), keepdim=True)
        return (x - minv) / (maxv - minv + 1e-6)


class StandardizePerSample:
    def __call__(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True)
        return (x - mean) / (std + 1e-6)


class To3Channels:
    def __call__(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.shape[0] == 1:
            return x.repeat(3, 1, 1)
        return x[:3]


class RandomGain:
    def __init__(self, min_gain=0.8, max_gain=1.25, p=0.5):
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            g = random.uniform(self.min_gain, self.max_gain)
            return x * g
        return x


class AdditiveNoise:
    def __init__(self, noise_level=0.005, p=0.5):
        self.noise_level = noise_level
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            return x + torch.randn_like(x) * (self.noise_level * x.abs().amax())
        return x


class RandomTimeFreqRoll:
    def __init__(self, max_shift_frac=0.1, axis='time', p=0.5, wrap=False):
        assert axis in ['time', 'freq']
        self.max_shift_frac = max_shift_frac
        self.axis = axis
        self.p = p
        self.wrap = wrap

    def __call__(self, x):
        if random.random() >= self.p:
            return x
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_after = True
        else:
            squeeze_after = False

        C, freq_bins, time_bins = x.shape
        if self.axis == 'time':
            max_shift = int(self.max_shift_frac * time_bins)
            if max_shift <= 0:
                return x[0] if squeeze_after else x
            shift = random.randint(-max_shift, max_shift)
            if shift == 0:
                return x[0] if squeeze_after else x
            if self.wrap:
                out = torch.roll(x, shifts=shift, dims=-1)
            else:
                out = torch.zeros_like(x)
                if shift > 0:
                    out[..., shift:] = x[..., :-shift]
                else:
                    s = -shift
                    out[..., :-s] = x[..., s:]
        else:  # freq axis
            max_shift = int(self.max_shift_frac * freq_bins)
            if max_shift <= 0:
                return x[0] if squeeze_after else x
            shift = random.randint(-max_shift, max_shift)
            if shift == 0:
                return x[0] if squeeze_after else x
            if self.wrap:
                out = torch.roll(x, shifts=shift, dims=-2)
            else:
                out = torch.zeros_like(x)
                if shift > 0:
                    out[:, shift:, :] = x[:, :-shift, :]
                else:
                    s = -shift
                    out[:, :-s, :] = x[:, s:, :]

        return out[0] if squeeze_after else out


class RandomTimeCropPad:
    def __init__(self, target_time, pad_val=0.0):
        self.target_time = int(target_time)
        self.pad_val = pad_val

    def __call__(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_after = True
        else:
            squeeze_after = False

        C, freq_bins, time_bins = x.shape
        if time_bins == self.target_time:
            out = x
        elif time_bins > self.target_time:
            start = random.randint(0, time_bins - self.target_time)
            out = x[:, :, start:start + self.target_time]
        else:
            pad_total = self.target_time - time_bins
            left = random.randint(0, pad_total)
            right = pad_total - left
            # F.pad expects pad=(left,right,top,bottom) for 2D (applied on last two dims via torch.nn.functional)
            # For (C, Freq, Time) we use F.pad on each sample in batch-like shape, so use pad on last two dims:
            out = F.pad(x, (left, right, 0, 0), value=self.pad_val)
        return out[0] if squeeze_after else out


class ResizeSpectrogram:
    def __init__(self, size):  # size = (H, W)
        self.size = size

    def __call__(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        # x: (C, F, T) -> add batch dim -> (1, C, F, T)
        x_b = x.unsqueeze(0)
        x_resized = F.interpolate(x_b, size=self.size, mode='bilinear', align_corners=False)
        return x_resized.squeeze(0)


# ---------- SpecAugment wrapper ----------
class SpecAugment:
    def __init__(self, time_mask_param=30, freq_mask_param=15, n_time_masks=1, n_freq_masks=1, p=1.0):
        self.p = p
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.n_time_masks = n_time_masks
        self.n_freq_masks = n_freq_masks
        self._tm = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
        self._fm = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)

    def __call__(self, x):
        if random.random() > self.p:
            return x
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_after = True
        else:
            squeeze_after = False

        out = x.clone()
        for ch in range(out.shape[0]):
            for _ in range(self.n_time_masks):
                out[ch] = self._tm(out[ch])
            for _ in range(self.n_freq_masks):
                out[ch] = self._fm(out[ch])
        return out[0] if squeeze_after else out

class ExpandTo3Channels:
    """把单通道 (1, H, W) 扩展成 3 通道 (3, H, W)"""
    def __call__(self, x):
        if x.ndim == 2:  # (H, W)
            x = x.unsqueeze(0)
        if x.shape[0] == 1:  # (1, H, W)
            x = x.repeat(3, 1, 1)
        return x

# ------------ Train Transform ------------
def make_spec_train_transform(image_size=224,
                              time_mask_param=30,
                              freq_mask_param=15,
                              n_time_masks=2,
                              n_freq_masks=2):
    """
    Train transform for spectrograms.
    Final output: (3, image_size, image_size)
    """
    return transforms.Compose([
        # 这里假设输入是 numpy 或 PIL.Image
        transforms.Resize((image_size, image_size)),
        transforms.RandomApply([
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05))
        ], p=0.3),   # 轻微平移
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3)
        ], p=0.2),   # 模糊增强，模拟频谱扰动
        transforms.ToTensor(),  # (1, H, W)
        transforms.Normalize(mean=[0.5], std=[0.5]),
        ExpandTo3Channels(),    # (3, H, W)
    ])


# ------------ Test Transform ------------
def make_spec_test_transform(image_size=224):
    """
    Test/validation transform for spectrograms.
    Final output: (3, image_size, image_size)
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),               # (1, H, W)
        transforms.Normalize(mean=[0.5], std=[0.5]),
        ExpandTo3Channels(),                 # (3, H, W)
    ])



def get_transform(transform_type='default', image_size=32, args=None):
    if transform_type == 'imagenet':

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        interpolation = args.interpolation
        crop_pct = args.crop_pct

        train_transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        test_transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

    elif transform_type == 'dc':
        train_transform = make_spec_train_transform(image_size=image_size)
        test_transform = make_spec_test_transform(image_size=image_size)

    elif transform_type == 'pytorch-cifar':

        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    elif transform_type == 'herbarium_default':

        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomResizedCrop(image_size, scale=(args.resize_lower_bound, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    elif transform_type == 'cutout':

        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])

        train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            normalize(mean, std),
            cutout(mask_size=int(image_size / 2),
                   p=1,
                   cutout_inside=False),
            to_tensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    elif transform_type == 'rand-augment':

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        train_transform.transforms.insert(0, RandAugment(args.rand_aug_n, args.rand_aug_m, args=None))

        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    elif transform_type == 'random_affine':

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        interpolation = args.interpolation
        crop_pct = args.crop_pct

        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation),
            transforms.RandomAffine(degrees=(-45, 45),
                                    translate=(0.1, 0.1), shear=(-15, 15), scale=(0.7, args.crop_pct)),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        test_transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

    else:

        raise NotImplementedError

    return (train_transform, test_transform)