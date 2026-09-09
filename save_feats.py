dino_v1 = True

import argparse
import os

from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.optim import SGD, lr_scheduler
from project_utils.cluster_utils import mixed_eval, AverageMeter
if dino_v1:
    from models import vision_transformer as vits
else:
    from models import vision_transformer2 as vits

from models.vision_transformer import DinoWithNeck


from project_utils.general_utils import init_experiment, get_mean_lr, str2bool, get_dino_head_weights

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from tqdm import tqdm

from torch.nn import functional as F

from project_utils.cluster_and_log_utils import log_accs_from_preds
from config import exp_root, dino_pretrain_path, dino_pretrain_path2
from matplotlib import pyplot as plt
from methods.clustering.faster_mix_k_means_pytorch import K_Means as SemiSupKMeans

from kmeans_pytorch import kmeans
from models.compositional import CompositionalPerception

# TODO: Debug
import warnings
import math
warnings.filterwarnings("ignore")


def extract_features(projection_head, model, train_loader, test_loader, unlabelled_train_loader, merge_train_loader, args):
    with torch.no_grad():
        model.eval()
        all_features = []
        for batch_idx, batch in enumerate(tqdm(train_loader)):

            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]
            class_labels, mask_lab = class_labels.to(device), mask_lab.to(device).bool()
            images = images.to(device)

            # Extract features with base model
            features = model(images)
            features = features.cpu().detach().numpy()
            all_features.append(features)
        all_features = np.concatenate(all_features, axis=0)
        np.save('latent_feats/{}.npy'.format(args.exp_name), all_features)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v1', 'v2'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--dataset_name', type=str, default='cub', help='options: cifar10, cifar100, scars, aircraft, herbarium_19, imagenet_100')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', type=str2bool, default=True)

    parser.add_argument('--grad_from_block', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--save_best_thresh', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--base_model', type=str, default='vit_dino')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sup_con_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--contrast_unlabel_only', type=str2bool, default=False)

    parser.add_argument('--strategy', type=str, default='zero_one')
    parser.add_argument('--cluster_momentum', type=float, default=1)

    parser.add_argument('--unsupervised_smoothing', type=float, default=1)
    parser.add_argument('--distance', type=str, default='euclidean',
                        help='options: euclidean, cosine')

    parser.add_argument('--train_report_interval', default=200, type=int)
    parser.add_argument('--prototype_extraction_interval', default=1, type=int)

    parser.add_argument('--gpu_clustering', type=str2bool, default=True)
    parser.add_argument('--unbalanced', type=str2bool, default=False)

    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--report', type=str2bool, default=True)
    parser.add_argument('--exp_name', default=None, type=str)

    # Hypergraph related
    parser.add_argument('--num_primitives', default=8, type=int, help='the number of primitives')
    parser.add_argument('--num_heads', default=4, type=int, help='the number of heads')
    parser.add_argument('--disable_composition', action='store_true', help='Disable CoGe-GCD token refinement')


    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    args = get_class_splits(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['metric_learn_gcd'])
    print(f'Using evaluation function {args.eval_funcs[0]} to print results')

    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = 65536

    # ----------------------
    # BASE MODEL
    # ----------------------
    if args.base_model == 'vit_dino':

        args.interpolation = 3
        args.crop_pct = 0.875
        if dino_v1:
            pretrain_path = dino_pretrain_path
            model = vits.__dict__['vit_base']()
            torch.cuda.empty_cache()
            # state_dict = torch.load(pretrain_path, map_location='cpu')['teacher']
            state_dict = torch.load(pretrain_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=True)
            if not args.disable_composition:
                print('现在跑的是hyper')
                neck = CompositionalPerception(embed_dim=args.feat_dim, num_primitives=args.num_primitives, num_heads=args.num_heads)
                model = DinoWithNeck(model, neck)
            
        else:
            pretrain_path = dino_pretrain_path2
            model = vits.__dict__['vit_base']()
            torch.cuda.empty_cache()
            state_dict = torch.load(pretrain_path, map_location='cpu')
            model.load_state_dict(state_dict)

    if args.warmup_model_dir is not None:
        print(f'Loading weights from {args.warmup_model_dir}')
        state_dict = torch.load(args.warmup_model_dir+'model_best.pt', map_location='cpu')
        print(state_dict.keys())
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)

        # ----------------------
        # HOW MUCH OF BASE MODEL TO FINETUNE
        # ----------------------
        for m in model.parameters():
            m.requires_grad = False

        # Only finetune layers from block 'args.grad_from_block' onwards
        max_block=0
        for name, m in model.named_parameters():
            if 'block' in name:
                # 如果用 HyperGraph 时，修改成这个形式
                # 因为 DinoWithHead 会在其权重前添上前缀
                if not args.disable_composition:
                    block_num = int(name.split('.')[2])
                else:
                    block_num = int(name.split('.')[1])
                if block_num > max_block:
                    max_block=block_num

                if block_num >= args.grad_from_block:
                    m.requires_grad = True

    else:
        raise NotImplementedError
    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    # train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / (unlabelled_len+label_len) for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    merge_train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=args.batch_size, shuffle=False)
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=args.batch_size, shuffle=False)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projection_head = vits.__dict__['DINOHead'](in_dim=args.feat_dim,
                               out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    if args.warmup_model_dir is not None:
        print(f'Loading projection head weights from {args.warmup_model_dir}')
        projection_head.load_state_dict(torch.load(args.warmup_model_dir + 'model_proj_head_best.pt', map_location='cpu'), strict=False)

    projection_head.to(device)


    # ----------------------
    # TRAIN
    # ----------------------
    if not os.path.exists('Plots'):
        os.mkdir('Plots')
    extract_features(projection_head, model, train_loader, test_loader_labelled, test_loader_unlabelled, merge_train_loader, args)
    torch.cuda.empty_cache()
