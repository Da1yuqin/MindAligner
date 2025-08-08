import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import AutoencoderKL
from omegaconf import OmegaConf
from torch import nn
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoProcessor

import my_utils.utils as utils
from args import get_test_args
from models.models import BTM
from my_utils.data_utils import get_test_dataloader
from my_utils.modeling_git import GitForCausalLMClipEmb

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
sys.path.append('generative_models/')
from generative_models.sgm.models.diffusion import DiffusionEngine


args = get_test_args()
args.cache_dir = args.data_path  # modify it if needed

# constants
SUBJ_DIMS = {1: 15724, 2: 14278, 3: 15226, 4: 13153, 5: 13039, 6: 17907, 7: 12682, 8: 14386}
SUBJ_LAYER_IDs = {
    1: {2: 0, 5: 3, 7: 5}, 
    2: {1: 0, 5: 3, 7: 5}, 
    5: {1: 0, 2: 1, 7: 5}, 
    7: {1: 0, 2: 1, 5: 4}
}
SUBJ_LAYER_ID = SUBJ_LAYER_IDs[args.n_subj]
align_name = f"{args.n_subj}->{args.k_subj}"


def load_test_datav2(subj):
    voxels = {}
    with h5py.File(f'{args.data_path}/betas_all_subj0{subj}_fp32_renorm.hdf5', 'r') as f:
        betas = torch.tensor(f['betas'][:]).to("cpu")
        voxels[f'subj0{subj}'] = betas
        print(f"num_voxels for subj0{subj}: {betas.shape[-1]}")

    test_dl, num_test = get_test_dataloader(args.data_path, subj, args.new_test)

    test_images_idx = []
    test_voxels_idx = []
    for test_i, (behav, _, _, _) in enumerate(test_dl):
        test_voxels = voxels[f'subj0{subj}'][behav[:,0,5].cpu().long()]
        
        curr_voxels_idx = behav[:,0,5].cpu().numpy()
        curr_images_idx = behav[:,0,0].cpu().numpy()
        test_voxels_idx = np.append(test_voxels_idx, curr_voxels_idx)
        test_images_idx = np.append(test_images_idx, curr_images_idx)
        
    test_images_idx = test_images_idx.astype(int)
    test_voxels_idx = test_voxels_idx.astype(int)
    assert (test_i + 1) * num_test == len(test_voxels) == len(test_images_idx)
    print(f"test_i: {test_i}, num_test: {num_test}")
    print(f"test_voxels: {test_voxels.shape}, test_images_idx: {test_images_idx.shape}\n")

    return test_voxels, test_images_idx


# setup text caption networks
def get_text_model():
    """
    Returns:
        tuple: (processor, clip_text_model, clip_convert)
    """
    # processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
    # clip_text_model = GitForCausalLMClipEmb.from_pretrained("microsoft/git-large-coco")
    processor = AutoProcessor.from_pretrained(f"{args.cache_dir}/git-large-coco_cache")
    clip_text_model = GitForCausalLMClipEmb.from_pretrained(f"{args.cache_dir}/git-large-coco_cache")
    clip_text_model.to(device)  # if you get OOM running this script, you can switch this to cpu and lower minibatch_size to 4
    clip_text_model.eval().requires_grad_(False)

    CLIP_CONFIG = utils.get_clip_config()
    class CLIPConverter(nn.Module):
        def __init__(self):
            super(CLIPConverter, self).__init__()
            self.linear1 = nn.Linear(CLIP_CONFIG['seq_dim'], CLIP_CONFIG['text_seq_dim'])
            self.linear2 = nn.Linear(CLIP_CONFIG['emb_dim'], CLIP_CONFIG['text_emb_dim'])
        
        def forward(self, x):
            x = x.permute(0, 2, 1)
            x = self.linear1(x)
            x = self.linear2(x.permute(0, 2, 1))
            return x
    
    clip_convert = CLIPConverter()
    state_dict = torch.load(f"{args.data_path}/bigG_to_L_epoch8.pth", map_location='cpu')['model_state_dict']
    clip_convert.load_state_dict(state_dict, strict=True)
    clip_convert.to(device)  # if you get OOM running this script, you can switch this to cpu and lower minibatch_size to 4
    clip_convert.eval().requires_grad_(False)

    return processor, clip_text_model, clip_convert


def get_unCLIP_model():
    """
    Returns:
        tuple: (diffusion_engine, vector_suffix)
    """
    def _load_unclip_config(config_path):
        config = OmegaConf.load(config_path)
        config = OmegaConf.to_container(config, resolve=True)
        params = config["model"]["params"]
        params["first_stage_config"]['target'] = 'sgm.models.autoencoder.AutoencoderKL'
        params["sampler_config"]['params']['num_steps'] = 38

        return params
    
    def _get_vector_suffix(model, device):
        batch = {
            "jpg": torch.randn(1, 3, 1, 1).to(device),  # jpg doesnt get used, it's just a placeholder
            "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
            "crop_coords_top_left": torch.zeros(1, 2).to(device)
        }
        out = model.conditioner(batch)
        vector_suffix = out["vector"].to(device)
        print(f"vector_suffix: {vector_suffix.shape}")

        return vector_suffix
    
    config = _load_unclip_config("generative_models/configs/unclip6.yaml")
    diffusion_engine = DiffusionEngine(network_config=config["network_config"],
                                       denoiser_config=config["denoiser_config"],
                                       first_stage_config=config["first_stage_config"],
                                       conditioner_config=config["conditioner_config"],
                                       sampler_config=config["sampler_config"],
                                       scale_factor=config["scale_factor"],
                                       disable_first_stage_autocast=config["disable_first_stage_autocast"])
    
    ckpt_path = f'{args.data_path}/unclip6_epoch0_step110000.ckpt'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    diffusion_engine.load_state_dict(ckpt['state_dict'])
    
    # set to inference
    diffusion_engine.to(device)
    diffusion_engine.eval().requires_grad_(False)
    
    vector_suffix = _get_vector_suffix(diffusion_engine, device)
    
    return diffusion_engine, vector_suffix


def get_SD_VAE_model():
    encoder_blocks = ['DownEncoderBlock2D'] * 4
    decoder_blocks = ['UpDecoderBlock2D'] * 4
    channel_sizes = [128, 256, 512, 512]

    autoencoder = AutoencoderKL(
        down_block_types=encoder_blocks,
        up_block_types=decoder_blocks,
        block_out_channels=channel_sizes,
        layers_per_block=2,
        sample_size=256,
    )
    ckpt = torch.load(f'{args.data_path}/sd_image_var_autoenc.pth')
    autoencoder.load_state_dict(ckpt)

    autoencoder.to(device)
    autoencoder.eval().requires_grad_(False)
    
    return autoencoder


def load_ckpt(model, tag="last", strict=True):
    ckpt = torch.load(f"{args.decoding_model_path}/final_multisubject_subj0{args.n_subj}/{tag}.pth", map_location='cpu')
    model_state = ckpt['model_state_dict']
    model.load_state_dict(model_state, strict=strict)

    print("successfully loaded checkpoint!")
    return model.to(device).eval().requires_grad_(False)


def save_outputs(all_clip_voxels, all_pred_captions, all_recons, all_blurry_recons=None):
    imsize = 256
    all_recons = transforms.Resize((imsize, imsize))(all_recons).float()
    if args.blurry_recon:
        all_blurry_recons = transforms.Resize((imsize, imsize))(all_blurry_recons).float()
    print(f"reconstructions shape: {all_recons.shape}")

    output_dir = os.path.join("evals", f"{align_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    outputs = {
        'all_recons': all_recons,
        'all_pred_captions': all_pred_captions,
        'all_clip_voxels': all_clip_voxels
    }
    if all_blurry_recons is not None:
        outputs['all_blurry_recons'] = all_blurry_recons
    
    for name, data in outputs.items():
        torch.save(data, os.path.join(output_dir, f"{align_name}_{name}.pt"))
    
    print(f"saved test outputs to {output_dir}!")


def main():
    torch.backends.cuda.matmul.allow_tf32 = True    # tf32 data type is faster than standard float32
    utils.seed_all(args.seed)                       # seed all random functions
    
    test_voxels, test_images_idx = load_test_datav2(args.n_subj)
    test_voxels = torch.tensor(test_voxels)

    model = utils.get_decoding_model(args)
    model = load_ckpt(model, tag="last", strict=True)
    
    processor, clip_text_model, clip_convert = get_text_model()
    diffusion_engine, vector_suffix = get_unCLIP_model()
    autoenc = get_SD_VAE_model() if args.blurry_recon else None
    
    patch_dim = args.bfa_latent
    i_dim = SUBJ_DIMS[args.n_subj]
    o_dim = SUBJ_DIMS[args.k_subj]
    ckpt_path = f"./ckpts/on_subj{args.n_subj}/{args.n_subj}->{args.k_subj}"
    
    checkpoint = torch.load(os.path.join(ckpt_path, 'best.pt'), map_location=device)
    align_model = BTM(i_dim, patch_dim, o_dim).to(device)
    align_model.load_state_dict(checkpoint['AlignModel'])
    del checkpoint

    if args.plotting:
        plt_save_pth = f"evals/out_plot/on_subj{args.n_subj}/{align_name}/" 
        os.makedirs(plt_save_pth, exist_ok=True)

    all_pred_captions = []
    clip_voxels_list, recons_list, blurry_recons_list = [], [], []

    minibatch_size = 1
    num_samples_per_image = 1
    assert num_samples_per_image == 1

    unique_image_ids = np.unique(test_images_idx)
    total_images = len(np.unique(test_images_idx))
    
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        for batch_idx in tqdm(range(0, total_images, minibatch_size)):
            curr_img_ids = unique_image_ids[batch_idx:batch_idx + minibatch_size]
            batch_voxels = []
            for img_id in curr_img_ids:
                # find all occurrences of this image in the dataset
                indices = np.where(test_images_idx == img_id)[0]
                if len(indices) == 1:
                    indices = np.tile(indices, 3)
                elif len(indices) == 2:
                    indices = np.tile(indices, 2)[:3]
                assert len(indices) == 3, f"Expected 3 samples for image {img_id}, got {len(indices)}."
                batch_voxels.append(test_voxels[indices])  # 1,  num_image_repetitions, num_voxels
            voxel = torch.stack(batch_voxels).to(device)   # bs, num_image_repetitions, num_voxels

            voxel = align_model(voxel)
            
            for rep in range(3):
                voxel_ridge = model.ridge(voxel[:, [rep]], SUBJ_LAYER_ID[args.k_subj]) 
                backbone0, clip_voxels0, blurry_image_enc0 = model.backbone(voxel_ridge)
                if rep == 0:
                    clip_voxels = clip_voxels0
                    backbone = backbone0
                    blurry_image_enc = blurry_image_enc0[0]
                else:
                    clip_voxels += clip_voxels0
                    backbone += backbone0
                    blurry_image_enc += blurry_image_enc0[0]
            clip_voxels /= 3
            backbone /= 3
            blurry_image_enc /= 3

            # save retrieval submodule outputs
            clip_voxels_list.append(clip_voxels.cpu())

            # feed voxels through OpenCLIP-bigG diffusion prior
            prior_out = model.diffusion_prior.p_sample_loop(backbone.shape,
                                                            text_cond=dict(text_embed=backbone),
                                                            cond_scale=1., timesteps=20)
            predicted_caption_emb = clip_convert(prior_out)
            generated_caption_ids = clip_text_model.generate(pixel_values=predicted_caption_emb, max_length=20)
            generated_caption = processor.batch_decode(generated_caption_ids, skip_special_tokens=True)
            all_pred_captions = np.hstack((all_pred_captions, generated_caption))
            print(generated_caption)

            # feed diffusion prior outputs through unCLIP
            for i in range(len(voxel)):
                samples = utils.unclip_recon(prior_out[[i]], diffusion_engine,
                                             vector_suffix, num_samples=num_samples_per_image)
                recons_list.append(samples.cpu())
                
                if args.plotting:
                    for s in range(num_samples_per_image):
                        plt.figure(figsize=(2, 2))
                        plt.imshow(transforms.ToPILImage()(samples[s]))
                        plt.axis('off')
                        plt.savefig(plt_save_pth + f"recon_{generated_caption}_{i+s}.png")
                        plt.close()

            if args.blurry_recon:
                blurry_image = (autoenc.decode(blurry_image_enc / 0.18215).sample / 2 + 0.5).clamp(0, 1)
                for i in range(len(voxel)):
                    im = torch.Tensor(blurry_image[i])
                    blurry_recons_list.append(im[None].cpu())
                    
                    if args.plotting:
                        plt.figure(figsize=(2, 2))
                        plt.imshow(transforms.ToPILImage()(im))
                        plt.axis('off')
                        plt.savefig(plt_save_pth + f"blurry_{generated_caption}_{i+s}.png")
                        plt.close()

        all_clip_voxels = torch.cat(clip_voxels_list, dim=0)
        all_recons = torch.cat(recons_list, dim=0)
        if args.blurry_recon:
            all_blurry_recons = torch.cat(blurry_recons_list, dim=0)
        else:
            all_blurry_recons = None
        
        save_outputs(all_clip_voxels, all_pred_captions, all_recons, all_blurry_recons)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    main()
