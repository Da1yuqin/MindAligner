import argparse
import os
import sys

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torchvision import transforms
from tqdm import tqdm

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
sys.path.append('generative_models/')
import my_utils.utils as utils
from generative_models.sgm.models.diffusion import DiffusionEngine
from generative_models.sgm.modules.encoders.modules import FrozenCLIPEmbedder, FrozenOpenCLIPEmbedder2
from generative_models.sgm.util import append_dims


def get_args():
    parser = argparse.ArgumentParser(description="Model Evaluation Configuration")
    
    parser.add_argument("--model_name", type=str, default="1->2")
    parser.add_argument("--data_path", type=str, default=".../your_data_path")
    parser.add_argument("--eval_path", type=str, default="evals")
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()


def get_diffusion_model():
    config = OmegaConf.load("generative_models/configs/unclip6.yaml")
    config = OmegaConf.to_container(config, resolve=True)
    unclip_params = config["model"]["params"]
    sampler_config = unclip_params["sampler_config"]
    sampler_config['params']['num_steps'] = 38

    config = OmegaConf.load("generative_models/configs/inference/sd_xl_base.yaml")
    config = OmegaConf.to_container(config, resolve=True)
    refiner_params = config["model"]["params"]

    base_ckpt_path = f"{args.data_path}/zavychromaxl_v30.safetensors"
    base_engine = DiffusionEngine(network_config=refiner_params["network_config"],
                                  denoiser_config=refiner_params["denoiser_config"],
                                  first_stage_config=refiner_params["first_stage_config"],
                                  conditioner_config=refiner_params["conditioner_config"],
                                  sampler_config=sampler_config,  # using the one defined by the unclip
                                  scale_factor=refiner_params["scale_factor"],
                                  disable_first_stage_autocast = refiner_params["disable_first_stage_autocast"],
                                  ckpt_path=base_ckpt_path)
    base_engine.to(device)
    base_engine.eval().requires_grad_(False)

    conditioner_config = refiner_params["conditioner_config"]
    base_text_embedder1 = FrozenCLIPEmbedder(
        layer=conditioner_config['params']['emb_models'][0]['params']['layer'],
        layer_idx=conditioner_config['params']['emb_models'][0]['params']['layer_idx'],
    )
    base_text_embedder1.to(device)

    base_text_embedder2 = FrozenOpenCLIPEmbedder2(
        arch=conditioner_config['params']['emb_models'][1]['params']['arch'],
        version=conditioner_config['params']['emb_models'][1]['params']['version'],
        freeze=conditioner_config['params']['emb_models'][1]['params']['freeze'],
        layer=conditioner_config['params']['emb_models'][1]['params']['layer'],
        always_return_pooled=conditioner_config['params']['emb_models'][1]['params']['always_return_pooled'],
        legacy=conditioner_config['params']['emb_models'][1]['params']['legacy'],
    )
    base_text_embedder2.to(device)

    return base_engine, base_text_embedder1, base_text_embedder2


def save_outputs(all_enhanced_recons, tag=None):
    # resize outputs before saving
    imsize = 256
    all_enhanced_recons = transforms.Resize((imsize, imsize))(all_enhanced_recons).float()
    print(f"enhanced reconstructions shape: {all_enhanced_recons.shape}")

    output_dir = os.path.join(args.eval_path, args.model_name + ("_" + str(tag) if tag is not None else ""))
    torch.save(all_enhanced_recons, os.path.join(output_dir, f"{args.model_name}_all_enhanced_recons.pt"))
    print(f"saved {args.model_name} outputs to {output_dir}!")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = get_args()
verbose = False

def main():
    print(args)
    
    torch.backends.cuda.matmul.allow_tf32 = True  # tf32 data type is faster than standard float32
    utils.seed_all(args.seed)                     # seed all random functions
    
    # Some of these files are downloadable from huggingface: https://huggingface.co/datasets/pscotti/mindeyev2/tree/main/evals
    # The others are obtained from running recon_inference.ipynb first with your desired model
    all_images = torch.load(f"{args.data_path}/src/evals/all_images.pt")
    all_recons = torch.load(f"{args.eval_path}/{args.model_name}/{args.model_name}_all_recons.pt")  # these are the unrefined MindEye2 recons!
    all_blurry_recons = torch.load(f"{args.eval_path}/{args.model_name}/{args.model_name}_all_blurry_recons.pt")
    all_pred_captions = torch.load(f"{args.eval_path}/{args.model_name}/{args.model_name}_all_pred_captions.pt")

    all_recons = transforms.Resize((768, 768))(all_recons).float()
    all_blurry_recons = transforms.Resize((768, 768))(all_blurry_recons).float()
    print(f"model_name: {args.model_name}")
    print(f"all_images.shape: {all_images.shape}, all_recons.shape: {all_recons.shape}")
    print(f"all_blurry_recons.shape: {all_blurry_recons.shape}")
    print(f"all_pred_captions.shape: {all_pred_captions.shape}")

    base_engine, base_text_embedder1, base_text_embedder2 = get_diffusion_model()
    batch = {
        "txt": "",
        "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
        "crop_coords_top_left": torch.zeros(1, 2).to(device),
        "target_size_as_tuple": torch.ones(1, 2).to(device) * 1024
    }
    out = base_engine.conditioner(batch)
    vector_suffix = out["vector"][:,-1536:].to(device)

    batch_uc = {
        "txt": "painting, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, skinny, glitchy, double torso, extra arms, extra hands, mangled fingers, missing lips, ugly face, distorted face, extra legs, anime",
        "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
        "crop_coords_top_left": torch.zeros(1, 2).to(device),
        "target_size_as_tuple": torch.ones(1, 2).to(device) * 1024
    }
    out = base_engine.conditioner(batch_uc)
    crossattn_uc = out["crossattn"].to(device)
    vector_uc = out["vector"].to(device)
    print(f"crossattn_uc.shape: {crossattn_uc.shape}, vector_uc.shape: {vector_uc.shape}")

    if verbose:
        clip_image_embedder = utils.get_clip_image_embedder(device)

    num_samples = 1         # PS: I tried increasing this to 16 and picking highest cosine similarity like we did in MindEye1, it didnt seem to increase eval performance!
    img2img_timepoint = 13  # 9: higher number means more reliance on prompt, less reliance on matching the conditioning image
    base_engine.sampler.guider.scale = 5  # 5: cfg
    def denoiser(x, sigma, c): return base_engine.denoiser(base_engine.model, x, sigma, c)

    all_enhanced_recons = None
    for image_idx in tqdm(range(len(all_recons))):
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16), base_engine.ema_scope():
            base_engine.sampler.num_steps = 25

            image = all_recons[[image_idx]]
            if verbose:
                print("blur pixcorr:", utils.pixcorr(all_blurry_recons[[image_idx]].float(), all_images[[image_idx]].float()))
                print("recon pixcorr:", utils.pixcorr(image.float(), all_images[[image_idx]].float()))

                print("blur cossim:", nn.functional.cosine_similarity(clip_image_embedder(utils.resize(all_blurry_recons[[image_idx]].float(), 224).to(device)).flatten(1),
                                                                      clip_image_embedder(utils.resize(all_images[[image_idx]].float(), 224).to(device)).flatten(1)))
                print("recon cossim:", nn.functional.cosine_similarity(clip_image_embedder(utils.resize(image.float(), 224).to(device)).flatten(1),
                                                                       clip_image_embedder(utils.resize(all_images[[image_idx]].float(), 224).to(device)).flatten(1)))

            image = image.to(device)
            assert image.shape[-1] == 768
            
            prompt = all_pred_captions[[image_idx]][0]
            print(f"prompt: {prompt}")

            openai_clip_text = base_text_embedder1(prompt)
            clip_text_tokenized, clip_text_emb = base_text_embedder2(prompt)
            clip_text_emb = torch.hstack((clip_text_emb, vector_suffix))
            clip_text_tokenized = torch.cat((openai_clip_text, clip_text_tokenized), dim=-1)
            c = {"crossattn": clip_text_tokenized.repeat(num_samples, 1, 1), "vector": clip_text_emb.repeat(num_samples, 1)}
            uc = {"crossattn": crossattn_uc.repeat(num_samples, 1, 1), "vector": vector_uc.repeat(num_samples, 1)}

            z = base_engine.encode_first_stage(image * 2 - 1).repeat(num_samples, 1, 1, 1)
            noise = torch.randn_like(z)
            sigmas = base_engine.sampler.discretization(base_engine.sampler.num_steps).to(device)
            init_z = (z + noise * append_dims(sigmas[-img2img_timepoint], z.ndim)) / torch.sqrt(1.0 + sigmas[0] ** 2.0)
            sigmas = sigmas[-img2img_timepoint:].repeat(num_samples, 1)

            base_engine.sampler.num_steps = sigmas.shape[-1] - 1
            noised_z, _, _, _, c, uc = base_engine.sampler.prepare_sampling_loop(init_z, cond=c, uc=uc, num_steps=base_engine.sampler.num_steps)
            for timestep in range(base_engine.sampler.num_steps):
                noised_z = base_engine.sampler.sampler_step(sigmas[:,timestep], sigmas[:,timestep+1],
                                                            denoiser, noised_z, cond=c, uc=uc, gamma=0)

            samples_z_base = noised_z
            samples_x = base_engine.decode_first_stage(samples_z_base)
            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

            # find best sample
            if num_samples == 1:
                samples = samples[0]
            else:
                sample_cossim = nn.functional.cosine_similarity(clip_image_embedder(utils.resize(samples, 224).to(device)).flatten(1),
                                                                clip_image_embedder(utils.resize(all_images[[image_idx]].float(), 224).to(device)).flatten(1))
                which_sample = torch.argmax(sample_cossim)
                samples = samples[which_sample]
            
            samples = samples.cpu()[None]
            if all_enhanced_recons is None:
                all_enhanced_recons = samples
            else:
                all_enhanced_recons = torch.vstack((all_enhanced_recons, samples))

    save_outputs(all_enhanced_recons, tag=None)


if __name__ == "__main__":
    main()
