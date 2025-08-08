import argparse
import logging
import sys

import clip
import numpy as np
import scipy as sp
import torch
import torch.nn as nn
from scipy import stats
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from torchvision.models import (AlexNet_Weights, EfficientNet_B1_Weights,
                                Inception_V3_Weights, alexnet, efficientnet_b1,
                                inception_v3)
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

import my_utils.utils as utils
from my_utils.utils import get_clip_image_embedder


def get_args():
    parser = argparse.ArgumentParser(description="Model Evaluation Configuration")
    
    parser.add_argument("--model_name", type=str, default="1->2")
    parser.add_argument("--data_path", type=str, default=".../your_data_path")
    parser.add_argument("--eval_path", type=str, default="evals")
    
    return parser.parse_args()


def setup_logger(level=logging.DEBUG):
    logger = logging.getLogger()
    logger.setLevel(level)
    
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def calculate_retrival_percent_correct(all_images, all_clip_voxels):
    clip_image_embedder = get_clip_image_embedder(device)
    
    fwd_percent_correct = []
    bwd_percent_correct = []
    with torch.cuda.amp.autocast(dtype=torch.float16):
        for i in tqdm(range(30)):
            random_samps = np.random.choice(np.arange(len(all_images)), size=300, replace=False)
            i_emb = clip_image_embedder(all_images[random_samps].to(device)).float()  # CLIP-image
            b_emb = all_clip_voxels[random_samps].to(device).float()                  # CLIP-brain

            # flatten if necessary
            i_emb = i_emb.reshape(len(i_emb), -1)
            b_emb = b_emb.reshape(len(b_emb), -1)

            # l2norm
            i_emb = nn.functional.normalize(i_emb, dim=-1)
            b_emb = nn.functional.normalize(b_emb, dim=-1)

            labels = torch.arange(len(i_emb)).to(device)
            fwd_sim = utils.batchwise_cosine_similarity(b_emb, i_emb)  # brain, clip
            bwd_sim = utils.batchwise_cosine_similarity(i_emb, b_emb)  # clip, brain

            fwd_percent_correct.append(utils.topk(fwd_sim, labels, k=1).item())
            bwd_percent_correct.append(utils.topk(bwd_sim, labels, k=1).item())

    mean_fwd_percent_correct = np.mean(fwd_percent_correct)
    mean_bwd_percent_correct = np.mean(bwd_percent_correct)

    fwd_sd = np.std(fwd_percent_correct) / np.sqrt(len(fwd_percent_correct))
    fwd_ci = stats.norm.interval(0.95, loc=mean_fwd_percent_correct, scale=fwd_sd)

    bwd_sd = np.std(bwd_percent_correct) / np.sqrt(len(bwd_percent_correct))
    bwd_ci = stats.norm.interval(0.95, loc=mean_bwd_percent_correct, scale=bwd_sd)

    print(f"fwd percent_correct: {mean_fwd_percent_correct:.4f} 95% CI: [{fwd_ci[0]:.4f},{fwd_ci[1]:.4f}]")
    print(f"bwd percent_correct: {mean_bwd_percent_correct:.4f} 95% CI: [{bwd_ci[0]:.4f},{bwd_ci[1]:.4f}]")

    return mean_fwd_percent_correct, mean_bwd_percent_correct


def calculate_pixcorr(gt, pd):
    preprocess = transforms.Compose([
        transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
    ])

    # flatten images while keeping the batch dimension
    gt_flat = preprocess(gt).reshape(len(gt), -1).cpu()
    pd_flat = preprocess(pd).reshape(len(pd), -1).cpu()
    logger.debug(f"gt_flat shape: {gt_flat.shape}")
    logger.debug(f"pd_flat shape: {pd_flat.shape}")

    print("image flattened, now calculating pixcorr...")
    pixcorr_score = []
    for gt, pd in tqdm(zip(gt_flat, pd_flat), total=len(gt_flat)):
        pixcorr_score.append(np.corrcoef(gt, pd)[0,1])
    
    return np.mean(pixcorr_score)


# see https://github.com/zijin-gu/meshconv-decoding/issues/3
def calculate_ssim(gt, pd):
    preprocess = transforms.Compose([
        transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR), 
    ])

    # convert image to grayscale with rgb2grey
    gt_gray = rgb2gray(preprocess(gt).permute((0,2,3,1)).cpu())
    pd_gray = rgb2gray(preprocess(pd).permute((0,2,3,1)).cpu())
    logger.debug(f"gt_gray shape: {gt_gray.shape}")
    logger.debug(f"pd_gray shape: {pd_gray.shape}")
    
    print("image converted to grayscale, now calculating ssim...")
    ssim_score = []
    for gt, pd in tqdm(zip(gt_gray, pd_gray), total=len(gt_gray)):
        ssim_score.append(ssim(gt, pd, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0))

    return np.mean(ssim_score)


def get_model(model_name):
    if model_name == 'Alex':
        weights = AlexNet_Weights.IMAGENET1K_V1
        model = create_feature_extractor(alexnet(weights=weights), return_nodes=['features.4', 'features.11'])
    elif model_name == 'Incep':
        weights = Inception_V3_Weights.DEFAULT
        model = create_feature_extractor(inception_v3(weights=weights), return_nodes=['avgpool'])
    elif model_name == 'CLIP':
        model, _ = clip.load("ViT-L/14", device=device)
        return model.encode_image
    elif model_name == 'Eff':
        weights = EfficientNet_B1_Weights.DEFAULT
        model = create_feature_extractor(efficientnet_b1(weights=weights), return_nodes=['avgpool'])
    elif model_name == 'SwAV':
        model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
        model = create_feature_extractor(model, return_nodes=['avgpool'])
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model.to(device).eval().requires_grad_(False)


def get_preprocess(model_name):
    if model_name == 'Alex':
        # see alex_weights.transforms()
        return transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif model_name == 'Incep':
        return transforms.Compose([
            transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif model_name == 'CLIP':
        return transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])
    elif model_name == 'Eff':
        return transforms.Compose([
            transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif model_name == 'SwAV':
        return transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def process_in_batches(images, model, preprocess, layer, batch_size):
    feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        batch = preprocess(batch).to(device)
        with torch.no_grad():
            if layer is None:
                feat = model(batch).float().flatten(1)
            else:
                feat = model(batch)[layer].float().flatten(1)
            feats.append(feat)
    return torch.cat(feats, dim=0).cpu().numpy()


def two_way_identification(gt, pd, return_avg=True):
    num_samples = len(gt)
    corr_mat = np.corrcoef(gt, pd)                   # compute correlation matrix
    corr_mat = corr_mat[:num_samples, num_samples:]  # extract relevant quadrant of correlation matrix
    
    congruent = np.diag(corr_mat)
    success = corr_mat < congruent
    success_cnt = np.sum(success, axis=0)

    if return_avg:
        return np.mean(success_cnt) / (num_samples - 1)
    else:
        return success_cnt, num_samples - 1


def calculate_metric(model_name, gt, pd, model, preprocess, layer, batch_size):
    gt = process_in_batches(gt, model, preprocess, layer, batch_size)
    pd = process_in_batches(pd, model, preprocess, layer, batch_size)

    if model_name in ['Alex', 'Incep', 'CLIP']:
        return two_way_identification(gt, pd)
    elif model_name in ['Eff', 'SwAV']:
        return np.array([sp.spatial.distance.correlation(gt[i], pd[i]) for i in range(len(gt))]).mean()
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def print_results(results):
    """
    Print formatted results with proper alignment
    """
    print("\n" + "="*50)
    print("Final Results Summary")
    print("="*50)

    print("\nLow Level Metrics:")
    print("-"*30)
    print(f"{'PixCorr:':<20} {results['PixCorr']:>10.4f}")
    print(f"{'SSIM:':<20} {results['SSIM']:>10.4f}")
    print(f"{'Alex(2):':<20} {results['Alex_2']:>10.4f} (2-way percent correct)")
    print(f"{'Alex(5):':<20} {results['Alex_5']:>10.4f} (2-way percent correct)")

    print("\nHigh Level Metrics:")
    print("-"*30)
    print(f"{'Incep:':<20} {results['Incep_avgpool']:>10.4f} (2-way percent correct)")
    print(f"{'CLIP:':<20} {results['CLIP_None']:>10.4f} (2-way percent correct)")
    print(f"{'Eff:':<20} {results['Eff_avgpool']:>10.4f}")
    print(f"{'SwAV:':<20} {results['SwAV_avgpool']:>10.4f}")

    print("\nRetrieval Metrics:")
    print("-"*30)
    print(f"{'fwd_percent_correct:':<20} {results['fwd_percent_correct']:>10.4f}")
    print(f"{'bwd_percent_correct:':<20} {results['bwd_percent_correct']:>10.4f}")


def _resize_tensors(size, *tensors):
    resize_transform = transforms.Resize((size, size))
    return tuple([resize_transform(tensor).float() for tensor in tensors])


logger = setup_logger()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = get_args()
results = {}
enhanced = False

def main():
    print(args)
    
    all_images = torch.load(f"{args.data_path}/src/evals/all_images.pt")
    all_recons = torch.load(f"{args.eval_path}/{args.model_name}/{args.model_name}_all_recons.pt")
    all_clip_voxels = torch.load(f"{args.eval_path}/{args.model_name}/{args.model_name}_all_clip_voxels.pt")

    if enhanced:
        print("Using enhanced reconstructions...\n")
        all_recons = torch.load(f"{args.eval_path}/{args.model_name}/{args.model_name}_all_enhanced_recons.pt")
        all_blurry_recons = torch.load(f"{args.eval_path}/{args.model_name}/{args.model_name}_all_blurry_recons.pt")
        all_recons = all_recons * .75 + all_blurry_recons * .25

    imsize = 256
    all_images, all_recons = _resize_tensors(imsize, all_images, all_recons)

    results['PixCorr'] = calculate_pixcorr(all_images, all_recons)
    print(f"PixCorr: {results['PixCorr']:.6f}\n")
    
    results['SSIM'] = calculate_ssim(all_images, all_recons)
    print(f"SSIM: {results['SSIM']:.6f}\n")

    net_list = [
        ('Alex', '2'),
        ('Alex', '5'),
        ('Incep', 'avgpool'),
        ('CLIP', None),  # final layer
        ('Eff', 'avgpool'),
        ('SwAV', 'avgpool'),
    ]

    batch_size = 32
    for model_name, layer in net_list:
        logger.info(f"calculating {model_name} with layer {layer}...")
        
        model = get_model(model_name)
        preprocess = get_preprocess(model_name)

        if model_name == 'Alex':
            feature_layer = {
                '2': 'features.4',
                '5': 'features.11',
            }.get(layer)
            results[f"{model_name}_{layer}"] = calculate_metric(model_name, all_images, all_recons, model, preprocess, feature_layer, batch_size)
        else:
            results[f"{model_name}_{layer}"] = calculate_metric(model_name, all_images, all_recons, model, preprocess, layer, batch_size)
        logger.info(f"{model_name}({layer}): {results[f'{model_name}_{layer}']:.6f}")

        # clear GPU memory
        del model
        torch.cuda.empty_cache()

    fwd_percent_correct, bwd_percent_correct = calculate_retrival_percent_correct(all_images, all_clip_voxels)
    results['fwd_percent_correct'] = fwd_percent_correct
    results['bwd_percent_correct'] = bwd_percent_correct
    
    print_results(results)


if __name__ == "__main__":
    main()
