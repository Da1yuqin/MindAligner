import kornia
import torch
from diffusers import AutoencoderKL
from kornia.augmentation import AugmentationSequential

from models.autoencoder.convnext import ConvnextXL


def post_process(feat, thr):
    """
    Post-process of features (threshold and normalization).
    """
    # ensure the input is a PyTorch tensor
    if not isinstance(feat, torch.Tensor):
        raise TypeError("Input should be a PyTorch tensor.")
    
    if thr > 0:
        # clamp the feature values to the range [-thr, thr]
        feat = torch.clamp(feat, min=-thr, max=thr)
    
    # compute the L2 norm along the last dimension and keep the dimensions for broadcasting
    norm = torch.norm(feat, p=2, dim=-1, keepdim=True)
    # avoid division by zero
    norm = torch.max(norm, torch.tensor(1e-12, device=feat.device))
    
    return feat / norm


def extract_suffix_numbers(strs):
    """
    Using list comprehension to extract and convert the numeric part after the underscore.
    """
    return [int(str.split('_')[1]) for str in strs]


def get_image_augment():
    return AugmentationSequential(
        kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.3),
        same_on_batch=False,
        data_keys=["input"],
    )


def get_SD_VAE_model(data_path, device):
    autoenc = AutoencoderKL(
        down_block_types=['DownEncoderBlock2D'] * 4,
        up_block_types=['UpDecoderBlock2D'] * 4,
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2, sample_size=256,
    )
    autoenc_ckpt = torch.load(f'{data_path}/sd_image_var_autoenc.pth')
    autoenc.load_state_dict(autoenc_ckpt)
    autoenc.to(device)
    autoenc.eval().requires_grad_(False)

    cnx = ConvnextXL(f'{data_path}/convnext_xlarge_alpha0.75_fullckpt.pth')
    cnx.to(device)
    cnx.eval().requires_grad_(False)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
    std = torch.tensor([0.228, 0.224, 0.225], device=device).reshape(1, 3, 1, 1)

    blur_augs = AugmentationSequential(
        kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
        kornia.augmentation.RandomGrayscale(p=0.1),
        kornia.augmentation.RandomSolarize(p=0.1),
        kornia.augmentation.RandomResizedCrop((224, 224), scale=(.9, .9), ratio=(1, 1), p=1.0),
        data_keys=["input"],
    )

    return autoenc, cnx, mean, std, blur_augs
