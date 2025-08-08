import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder
from generative_models.sgm.util import append_dims
from models.models import *
from torchvision import transforms


def seed_all(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        # needs to be False to use conv3D
        print('Note: not using cudnn.deterministic')


def get_log_config(args):
    print(f"wandb {args.wandb_project} run {args.model_name}")

    wandb_config = {
        "known_subj_id": args.k_subj,
        "novel_subj_id": args.n_subj,
        "bfa_latent": args.bfa_latent,
        "lr_b": args.lr_b,
        "lr_n": args.lr_n,
        "lr_f": args.lr_f,
        "lr_scheduler_type": args.lr_scheduler_type,
        "num_sessions": args.num_sessions,
        "seed": args.seed
    }

    print("wandb_config:\n", wandb_config)
    return wandb_config



#################################### Model ####################################
def get_clip_config():
    CLIP_CONFIG = {
        'seq_dim': 256,
        'emb_dim': 1664,
        'text_seq_dim': 257,
        'text_emb_dim': 1024
    }
    return CLIP_CONFIG


def get_clip_image_embedder(device=None):
    clip_image_embedder = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14",
        version="laion2b_s39b_b160k",
        output_tokens=True,
        only_tokens=True,
    )
    if device is not None:
        clip_image_embedder.to(device)
    return clip_image_embedder


def get_decoding_model(args):
    class DecodingModel(nn.Module):
        def __init__(self):
            super(DecodingModel, self).__init__()
        
        def load_ckpt(self, ckpt_path, strict=True):
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            model_state_dict = checkpoint['model_state_dict']
            self.load_state_dict(model_state_dict, strict=strict)
            print(f"Loaded decoding model checkpoint from {ckpt_path}!")
            del checkpoint
        
        def forward(self, x):
            return x
    
    class RidgeRegression(nn.Module):
        def __init__(self, in_features, out_feature):
            super(RidgeRegression, self).__init__()
            self.linears = nn.ModuleList([
                nn.Linear(in_feature, out_feature) for in_feature in in_features
            ])
        
        def forward(self, x, subj_idx):
            out = self.linears[subj_idx](x[:, 0]).unsqueeze(1)
            return out

    model = DecodingModel()

    # {1: 15724, 2: 14278, 3: 15226, 4: 13153, 5: 13039, 6: 17907, 7: 12682, 8: 14386}
    num_voxels_dict = {
        1: [14278, 15226, 13153, 13039, 17907, 12682, 14386],
        2: [15724, 15226, 13153, 13039, 17907, 12682, 14386],
        5: [15724, 14278, 15226, 13153, 17907, 12682, 14386],
        7: [15724, 14278, 15226, 13153, 13039, 17907, 14386],
    }
    num_voxels_list = num_voxels_dict[args.n_subj]
    model.ridge = RidgeRegression(num_voxels_list, out_feature=args.hidden_dim)
    count_params(model.ridge, "model.ridge")

    CLIP_CONFIG = get_clip_config()
    clip_seq_dim = CLIP_CONFIG['seq_dim']
    clip_emb_dim = CLIP_CONFIG['emb_dim']
    model.backbone = BrainNetwork(h=args.hidden_dim, in_dim=args.hidden_dim, seq_len=1, n_blocks=args.n_blocks,
                                  clip_size=clip_emb_dim, out_dim=clip_emb_dim*clip_seq_dim,
                                  blurry_recon=args.blurry_recon, clip_scale=args.clip_scale)
    count_params(model.backbone, "model.backbone")

    # setup diffusion prior network
    out_dim = clip_emb_dim
    depth = 6
    dim_head = 52
    heads = clip_emb_dim // dim_head
    timesteps = 100
    prior_network = PriorNetwork(
        dim=out_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        num_tokens=clip_seq_dim,
        learned_query_mode="pos_emb",
    )
    model.diffusion_prior = BrainDiffusionPrior(
        net=prior_network,
        image_embed_dim=out_dim,
        condition_on_text_encodings=False,
        timesteps=timesteps,
        cond_drop_prob=0.2,
        image_embed_scale=None,
    )
    count_params(model.diffusion_prior, "model.diffusion_prior")
    
    return model


def count_params(model, model_name, verbose=True):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"\n{model_name} param counts:\n{total_params:,} total\n{trainable_params:,} trainable")
    return total_params, trainable_params



def batchwise_cosine_similarity(Z, B):
    Z = Z.flatten(1)
    B = B.flatten(1).T

    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)  # Size (1, b).

    cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
    return cosine_similarity


def batchwise_pearson_correlation(Z, B):
    Z_mean = torch.mean(Z, dim=1, keepdim=True)
    B_mean = torch.mean(B, dim=1, keepdim=True)

    # center the data by subtracting the mean
    Z_centered = Z - Z_mean
    B_centered = B - B_mean

    # compute the numerator of pearson correlation
    numerator = Z_centered @ B_centered.T

    # compute the denominator of pearson correlation
    Z_centered_norm = torch.linalg.norm(Z_centered, dim=1, keepdim=True)
    B_centered_norm = torch.linalg.norm(B_centered, dim=1, keepdim=True)
    denominator = Z_centered_norm @ B_centered_norm.T

    pearson_correlation = numerator / denominator
    return pearson_correlation


def pixcorr(images, brains, nan=True):
    """
    Compute the mean Pearson correlation between flattened image and brain reconstruction data.
    """
    preprocess = transforms.Compose([
        transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
    ])
    all_images_flattened = preprocess(images).reshape(len(images), -1)
    all_brains_flattened = preprocess(brains).reshape(len(brains), -1)

    corr_diag = torch.diag(batchwise_pearson_correlation(all_images_flattened, all_brains_flattened))
    if nan:
        corr_mean = torch.nanmean(corr_diag)
    else:
        corr_mean = torch.mean(corr_diag)
    return corr_mean


def topk(similarities, labels, k=5):
    if k > similarities.shape[0]:
        k = similarities.shape[0]
    topsum = 0
    for i in range(k):
        topsum += torch.sum(torch.argsort(similarities, axis=1)[:, -(i+1)] == labels) / len(labels)
    return topsum


def soft_clip_loss(preds, targs, temp=0.125):
    t_t = (targs @ targs.T) / temp
    p_t = (preds @ targs.T) / temp
    
    loss1 = -(p_t.log_softmax(-1) * t_t.softmax(-1)).sum(-1).mean()
    loss2 = -(p_t.T.log_softmax(-1) * t_t.softmax(-1)).sum(-1).mean()

    return (loss1 + loss2) / 2


def soft_cont_loss(student_preds, teacher_preds, teacher_aug_preds, temp=0.125):
    teacher_teacher_aug = (teacher_preds @ teacher_aug_preds.T) / temp
    teacher_teacher_aug_t = (teacher_aug_preds @ teacher_preds.T) / temp
    student_teacher_aug = (student_preds @ teacher_aug_preds.T) / temp
    student_teacher_aug_t = (teacher_aug_preds @ student_preds.T) / temp

    loss1 = -(student_teacher_aug.log_softmax(-1) * teacher_teacher_aug.softmax(-1)).sum(-1).mean()
    loss2 = -(student_teacher_aug_t.log_softmax(-1) * teacher_teacher_aug_t.softmax(-1)).sum(-1).mean()

    return (loss1 + loss2) / 2


def resize(image, image_size=128):
    if image.ndim == 3: image = image[None]
    return nn.functional.interpolate(image, size=(image_size, image_size), mode='nearest')





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')






def mixco(data, perm=None, b_conc=0.15, beta=None, s_thresh=0.5, select=None):
    device, dtype = data.device, data.dtype

    if perm is None:
        perm = torch.randperm(data.shape[0])
    if beta is None:
        beta = torch.distributions.Beta(b_conc, b_conc).sample([data.shape[0]]).to(device=device, dtype=dtype)
    if select is None:
        select = (torch.rand(data.shape[0]) <= s_thresh).to(device=device, dtype=torch.bool)

    beta_shape = [-1] + [1] * (len(data.shape) - 1)
    data_shuffle = data[perm].to(device=device, dtype=dtype)
    data[select] = data[select] * beta[select].reshape(*beta_shape) + \
        data_shuffle[select] * (1 - beta[select]).reshape(*beta_shape)
    beta[~select] = 1
    
    return data, perm, beta, select


def mixco_nce(preds, targs, temp=0.1, perm=None, betas=None, select=None, distributed=False, 
              accelerator=None, local_rank=None, bidirectional=True):
    brain_clip = (preds @ targs.T)/temp
    
    if perm is not None and betas is not None and select is not None:
        probs = torch.diag(betas)
        probs[torch.arange(preds.shape[0]).to(preds.device), perm] = 1 - betas

        loss = -(brain_clip.log_softmax(-1) * probs).sum(-1).mean()
        if bidirectional:
            loss2 = -(brain_clip.T.log_softmax(-1) * probs.T).sum(-1).mean()
            loss = (loss + loss2)/2
        return loss
    else:
        loss =  F.cross_entropy(brain_clip, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
        if bidirectional:
            loss2 = F.cross_entropy(brain_clip.T, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
            loss = (loss + loss2)/2
        return loss

    
def check_loss(loss):
    if loss.isnan().any():
        raise ValueError('NaN loss')

def cosine_anneal(start, end, steps):
    return end + (start - end)/2 * (1 + torch.cos(torch.pi*torch.arange(steps)/(steps-1)))





def unclip_recon(x, diffusion_engine, vector_suffix,
                 num_samples=1, offset_noise_level=0.04):
    assert x.ndim==3
    if x.shape[0]==1:
        x = x[[0]]
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16), diffusion_engine.ema_scope():
        z = torch.randn(num_samples,4,96,96).to(device) # starting noise, can change to VAE outputs of initial image for img2img


        token_shape = x.shape
        tokens = x
        c = {"crossattn": tokens.repeat(num_samples,1,1), "vector": vector_suffix.repeat(num_samples,1)}

        tokens = torch.randn_like(x)
        uc = {"crossattn": tokens.repeat(num_samples,1,1), "vector": vector_suffix.repeat(num_samples,1)}

        for k in c:
            c[k], uc[k] = map(lambda y: y[k][:num_samples].to(device), (c, uc))

        noise = torch.randn_like(z)
        sigmas = diffusion_engine.sampler.discretization(diffusion_engine.sampler.num_steps)
        sigma = sigmas[0].to(z.device)

        if offset_noise_level > 0.0:
            noise = noise + offset_noise_level * append_dims(
                torch.randn(z.shape[0], device=z.device), z.ndim
            )
        noised_z = z + noise * append_dims(sigma, z.ndim)
        noised_z = noised_z / torch.sqrt(
            1.0 + sigmas[0] ** 2.0
        )  # Note: hardcoded to DDPM-like scaling. need to generalize later.

        def denoiser(x, sigma, c):
            return diffusion_engine.denoiser(diffusion_engine.model, x, sigma, c)

        samples_z = diffusion_engine.sampler(denoiser, noised_z, cond=c, uc=uc)
        samples_x = diffusion_engine.decode_first_stage(samples_z)
        samples = torch.clamp((samples_x*.8+.2), min=0.0, max=1.0)
        # samples = torch.clamp((samples_x + .5) / 2.0, min=0.0, max=1.0)
        return samples
    



# numpy utility
def interate_range(start, length, batchsize):
    batch_count = int(length // batchsize)
    residual = int(length % batchsize)
    for i in range(batch_count):
        yield range(start + i * batchsize, start + (i + 1) * batchsize), batchsize
    if residual > 0:
        yield range(start + batch_count * batchsize, start + length), residual


def get_value(_x):
    return np.copy(_x.data.cpu().numpy())
