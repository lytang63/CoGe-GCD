"""User-configurable paths for CoGe-GCD.

Set environment variables to point to datasets/checkpoints on another machine.
"""
import os
from pathlib import Path

_DATA_ROOT = Path(os.environ.get("COGE_DATA_ROOT", "data/datasets"))
# cifar_10_root = '../../data/datasets/cifar10'
# cifar_100_root = '../../data/datasets/cifar100'
# cub_root = '../../data/datasets/CUB'
# car_root = '../../data/datasets/stanford_car/cars_{}/'
# pets_root = '../../data/datasets/pets/'

# aircraft_root = '../../data/datasets/aircraft/fgvc-aircraft-2013b'
# herbarium_dataroot = '../../data/datasets/herbarium_19/'
# imagenet_root = '../../data/datasets/ImageNet/'#ILSVRC12'

cifar_10_root = os.environ.get("COGE_CIFAR10_ROOT", str(_DATA_ROOT / "cifar10"))
cifar_100_root = os.environ.get("COGE_CIFAR100_ROOT", str(_DATA_ROOT / "cifar100"))
cub_root = os.environ.get("COGE_CUB_ROOT", str(_DATA_ROOT / "cub"))
aircraft_root = os.environ.get("COGE_AIRCRAFT_ROOT", str(_DATA_ROOT / "fgvc-aircraft-2013b"))
car_root = os.environ.get("COGE_CARS_ROOT", str(_DATA_ROOT / "stanford_cars"))
herbarium_dataroot = os.environ.get("COGE_HERBARIUM_ROOT", str(_DATA_ROOT / "herbarium_19"))
imagenet_root = os.environ.get("COGE_IMAGENET_ROOT", str(_DATA_ROOT / "imagenet"))
pets_root = os.environ.get("COGE_PETS_ROOT", str(_DATA_ROOT / "pets"))
radar_root = os.environ.get("COGE_RADAR_ROOT", str(_DATA_ROOT / "radar"))

# OSR Split dir
osr_split_dir = os.environ.get("COGE_SPLIT_DIR", "data/ssb_splits")

# -----------------
# OTHER PATHS
# -----------------
dino_pretrain_path2 = os.environ.get("COGE_DINO_V2_PATH", "pretrained_models/dinov2_vitb14_reg4_pretrain.pth")

dino_pretrain_path = os.environ.get("COGE_DINO_PATH", "pretrained_models/dino_vitbase16_pretrain.pth")
warmup_pretrain_path = os.environ.get("COGE_WARMUP_PATH", "pretrained_models/GCDTeacher/model_best.pth")
feature_extract_dir = os.environ.get("COGE_FEATURE_DIR", "outputs/features")
exp_root = os.environ.get("COGE_EXP_ROOT", "outputs/experiments")
