import os
import time

import clip
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch import nn
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import my_utils.data_utils as data_utils
import my_utils.train_utils as train_utils
import my_utils.utils as utils
from args import get_train_args
from models.models import *
from models.models import BTM, FiLM
from my_utils.data_utils import PairedIndexDataset, custom_collate
from my_utils.losses import *


args = get_train_args()

# constants
SUBJ_DIMS = {1: 15724, 2: 14278, 3: 15226, 4: 13153, 5: 13039, 6: 17907, 7: 12682, 8: 14386}
SUBJ_LAYER_IDs = {
    1: {2: 0, 5: 3, 7: 5}, 
    2: {1: 0, 5: 3, 7: 5}, 
    5: {1: 0, 2: 1, 7: 5}, 
    7: {1: 0, 2: 1, 5: 4}
}
SUBJ_LAYER_ID = SUBJ_LAYER_IDs[args.n_subj]


def load_cached_clip_embeds(image_data1, image_data2, device):
    os.makedirs("./cache_clip_similarity", exist_ok=True)
    cache_pth1 = f"./cache_clip_similarity/subj{args.k_subj}_{args.num_sessions}session_clip.pt"
    cache_pth2 = f"./cache_clip_similarity/subj{args.n_subj}_{args.num_sessions}session_clip.pt"

    # compute embeddings if not cached
    if not os.path.exists(cache_pth1) or not os.path.exists(cache_pth2):
        model, preprocess = clip.load("ViT-L/14", device=device)
        all_subj1_clip_embeds = []
        all_subj2_clip_embeds = []
        
        with torch.no_grad():
            for _, (img1_tensor, img2_tensor) in enumerate(tqdm(zip(image_data1, image_data2))):
                img1 = transforms.ToPILImage()(img1_tensor).convert('RGB')
                img2 = transforms.ToPILImage()(img2_tensor).convert('RGB')

                image_input1 = preprocess(img1).unsqueeze(0).to(device)
                image_input2 = preprocess(img2).unsqueeze(0).to(device)

                image_features1 = model.encode_image(image_input1)
                image_features2 = model.encode_image(image_input2)

                all_subj1_clip_embeds.append(image_features1.cpu())
                all_subj2_clip_embeds.append(image_features2.cpu())

                del image_input1, image_input2, image_features1, image_features2
                torch.cuda.empty_cache() 

        torch.save(all_subj1_clip_embeds, cache_pth1)
        torch.save(all_subj2_clip_embeds, cache_pth2)

    all_subj1_clip_embeds = torch.load(cache_pth1)
    all_subj2_clip_embeds = torch.load(cache_pth2)

    all_subj1_clip_embeds = torch.from_numpy(np.asarray(all_subj1_clip_embeds)).squeeze()
    all_subj2_clip_embeds = torch.from_numpy(np.asarray(all_subj2_clip_embeds)).squeeze()

    all_subj1_clip_embeds = train_utils.post_process(feat=all_subj1_clip_embeds, thr=1.5)
    all_subj2_clip_embeds = train_utils.post_process(feat=all_subj2_clip_embeds, thr=1.5)

    return all_subj1_clip_embeds, all_subj2_clip_embeds


def prepare_models(align_in_dim, align_out_dim, patch_dim, device):
    align_model = BTM(align_in_dim, patch_dim, align_out_dim).to(device)
    NM_model = FiLM(768, patch_dim).to(device)
    FE_model = nn.Linear(patch_dim, patch_dim).to(device)

    decoding_model = utils.get_decoding_model(args)
    decoding_model.load_ckpt(ckpt_path=f"{args.decoding_model_path}/final_multisubject_subj0{args.n_subj}/last.pth")  
    decoding_model = decoding_model.to(device)

    return align_model, NM_model, FE_model, decoding_model


def main():
    utils.seed_all(seed=args.seed)

    # setup basic configurations
    data_type = torch.float32  # change depending on your mixed_precision
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    subj_list = [args.k_subj, args.n_subj]
    patch_dim = args.bfa_latent
    align_in_dim  = SUBJ_DIMS[args.n_subj]
    align_out_dim = SUBJ_DIMS[args.k_subj]
    
    train_dls, voxel_dataset, num_train = data_utils.get_train_dataloaders(
        subj_ids=subj_list, num_sessions=args.num_sessions, return_num_train=True, data_path=args.data_path,
    )
    print(f"\ntrain number: {num_train}")  # [682, 682]
    test_dataloader, num_test = data_utils.get_test_dataloader(args.data_path, args.n_subj, new_test=True)

    behav1, _, _, _ = next(iter(train_dls[0]))
    image_idx1 = behav1[:,0,0].cpu().long().numpy()
    voxel_idx1 = behav1[:,0,5].cpu().long().numpy()
    image_uniq_idx1, image_sorted_idx1 = np.unique(image_idx1, return_index=True)  # 536
    voxel_sorted_idx1 = voxel_idx1[image_sorted_idx1]
    voxel_data1 = voxel_dataset[f'subj0{subj_list[0]}'][voxel_sorted_idx1]
    voxel_data1 = torch.tensor(voxel_data1).unsqueeze(1)
    
    behav2, _, _, _ = next(iter(train_dls[1]))
    image_idx2 = behav2[:,0,0].cpu().long().numpy()
    voxel_idx2 = behav2[:,0,5].cpu().long().numpy()
    image_uniq_idx2, image_sorted_idx2 = np.unique(image_idx2, return_index=True)  # 536
    voxel_sorted_idx2 = voxel_idx2[image_sorted_idx2]
    voxel_data2 = voxel_dataset[f'subj0{subj_list[1]}'][voxel_sorted_idx2]
    voxel_data2 = torch.tensor(voxel_data2).unsqueeze(1)

    image_dataset = data_utils.load_nsd_images(args.data_path)
    image_np1 = image_uniq_idx1.reshape(-1)
    image_np2 = image_uniq_idx2.reshape(-1)
    image_data1 = torch.tensor(np.array([image_dataset[idx] for idx in image_np1]))
    image_data2 = torch.tensor(np.array([image_dataset[idx] for idx in image_np2]))

    # load or compute CLIP embeddings
    all_subj1_clip_embeds, all_subj2_clip_embeds = load_cached_clip_embeds(image_data1, image_data2, device)

    # load paired data for training
    json_file1 = f'./sim_dataset/v2subj1257/category_image_idx_subj{args.n_subj}.json'
    json_file2 = f'./sim_dataset/v2subj1257/category_image_idx_subj{args.k_subj}.json'

    pair_dataset = PairedIndexDataset(json_file1, json_file2, seed=42)
    pair_dataloader = DataLoader(pair_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate)

    # prepare models
    align_model, NM_model, FE_model, model = prepare_models(align_in_dim, align_out_dim, patch_dim, device)

    # freeze decoding model parameters
    align_model.requires_grad_(True)
    NM_model.requires_grad_(True)
    FE_model.requires_grad_(True)
    model.ridge.requires_grad_(False)
    model.backbone.requires_grad_(False)
    model.diffusion_prior.requires_grad_(False)
    print("\nDone with model preparations!")
    
    clip_image_embedder = utils.get_clip_image_embedder(device)
    if args.blurry_recon:
        autoenc, cnx, mean, std, blur_augs = train_utils.get_SD_VAE_model(args.data_path, device)
        # utils.count_params(autoenc, 'autoenc')
        # utils.count_params(cnx, 'cnx')

    ckpt_path = f"./ckpts/on_subj{args.n_subj}/{args.n_subj}->{args.k_subj}"
    os.makedirs(ckpt_path, exist_ok=True)

    # load checkpoint if exists
    start_epoch = 0
    if os.path.exists(os.path.join(ckpt_path, 'best.pt')):
        checkpoint = torch.load(os.path.join(ckpt_path, 'best.pt'), map_location=device)
        align_model.load_state_dict(checkpoint['AlignModel'])
        NM_model.load_state_dict(checkpoint['NeuralMapper'])
        FE_model.load_state_dict(checkpoint['FunctionalEmbedder'])
        print(f"Checkpoint loaded. Resuming training from epoch {start_epoch}.")

    # setup training configuration
    optimizer_align = optim.Adam(align_model.parameters(), lr=args.lr_b)  # learning rate for the BTM model
    optimizer_NM = optim.Adam(NM_model.parameters(), lr=args.lr_n)        # learning rate for the NeuralMapper model
    optimizer_FE = optim.Adam(FE_model.parameters(), lr=args.lr_f)        # learning rate for the FunctionalEmbedder model

    scheduler_config = {
        'total_steps': 399999,
        'final_div_factor': 1000,
        'last_epoch': -1,
        'pct_start': 2/150
    }

    lr_scheduler_alignM = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_align,
        max_lr=args.lr_b,
        **scheduler_config
    )
    lr_scheduler_NeuroM = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_NM,
        max_lr=args.lr_n,
        **scheduler_config
    )
    
    mse = nn.MSELoss()
    l1 = nn.L1Loss()
    soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, args.num_epochs - int(args.mixup_pct * args.num_epochs))
    
    if args.wandb_log:
        wandb_config = utils.get_log_config(args)
        wandb.init(
            id=f"{args.n_subj}to{args.k_subj}_{int(time.time())}", 
            name=f"{args.n_subj}to{args.k_subj}", 
            project=args.wandb_project,
            config=wandb_config,
            resume="allow",
        )

    best_retival = 0
    epoch = 0
    test_voxel, test_image = None, None
    for epoch_all in range(args.iters):
        align_model.train()
        NM_model.train()
        FE_model.train()
        
        optimizer_align.zero_grad()
        optimizer_NM.zero_grad()
        optimizer_FE.zero_grad()

        epoch = epoch_all % 150
        lr = args.lr_f * (0.7 ** (epoch // 25))
        for g in optimizer_FE.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)	

        batch = next(iter(pair_dataloader))
        pick1 = train_utils.extract_suffix_numbers(batch['img_ids1'])
        pick2 = train_utils.extract_suffix_numbers(batch['img_ids2'])

        vovel1 = voxel_data1[pick1, 0:1]
        vovel2 = voxel_data2[pick2, 0:1]
        subj1_clip_embedding = all_subj1_clip_embeds[pick1].to(device).to(torch.float32)
        subj2_clip_embedding = all_subj2_clip_embeds[pick2].to(device).to(torch.float32)
        diff12 = subj1_clip_embedding - subj2_clip_embedding  # diff12: torch.Size([800, 768])
        diff12 = diff12.unsqueeze(1).to(torch.float32)

        vovel1 = torch.tensor(vovel1).to(device).to(torch.float32) 
        vovel2 = torch.tensor(vovel2).to(device).to(torch.float32)

        vovel1_dist = Categorical(logits=vovel1)

        z2 = align_model.layers[0](vovel2).to(torch.float32) # [800, 3, 4096]
        z1 = NM_model(z2, diff12).to(torch.float32)
        vovel1_pred = align_model.layers[1](z1).to(torch.float32)
        vovel1_pred_dist = Categorical(logits=vovel1_pred)
        
        recon_loss = loss_function(vovel1_pred.to(torch.float32), vovel1.to(torch.float32), vovel1_pred_dist, vovel1_dist)
        
        fmri_embedding1 = FE_model(torch.mean(z1, dim=1))  # [500, 512]
        fmri_embedding2 = FE_model(torch.mean(z2, dim=1))  # [500, 512]

        rdm_feat = cal_rdm(subj1_clip_embedding, subj2_clip_embedding)  
        rdm_fmri = cal_rdm(fmri_embedding1, fmri_embedding2) # (500, 500)
        loss_rdm = regularization_F(rdm_feat.to(torch.float32) - rdm_fmri.to(torch.float32))

        loss_abmodel = recon_loss + \
                    loss_rdm * 1e-3
        
        loss_abmodel.backward()
        optimizer_align.step()
        optimizer_NM.step()
        optimizer_FE.step()
        lr_scheduler_NeuroM.step()
        torch.cuda.empty_cache()

        optimizer_align.zero_grad()
        model.eval()
        fwd_percent_correct = 0.
        bwd_percent_correct = 0.
        test_fwd_percent_correct = 0.
        test_bwd_percent_correct = 0.

        recon_cossim = 0.
        test_recon_cossim = 0.
        recon_mse = 0.
        test_recon_mse = 0.

        loss_clip_total = 0.
        loss_blurry_total = 0.
        loss_blurry_cont_total = 0.
        test_loss_clip_total = 0.

        loss_prior_total = 0.
        test_loss_prior_total = 0.

        blurry_pixcorr = 0.
        test_blurry_pixcorr = 0.  # needs >.456 to beat low-level subj01 results in mindeye v1

        # pre-load all batches for this epoch (it's MUCH faster to pre-load in bulk than to separate loading per batch)
        voxel_iters = {}
        image_iters = {}
        image_id_iters = {}
        perm_iters, betas_iters, select_iters = {}, {}, {}
    
        image_picked_id2 = image_uniq_idx2[pick2]
        voxel_iters[f"iter{0}"] = vovel2.to(torch.float32) 
        image_id_iters[f"iter{0}"] = image_picked_id2
        image_np = image_picked_id2.reshape(-1)                      
        image_data = np.array([image_dataset[idx] for idx in image_np])  
        image_tensor = torch.tensor(image_data).to(device)
        image_iters[f"iter{0}"] = image_tensor.to(torch.float32)         

        if epoch < int(args.mixup_pct * args.num_epochs):
            voxel, perm, betas, select = utils.mixco(vovel2)
            perm_iters[f"iter{0}"] = perm
            betas_iters[f"iter{0}"] = betas.to(torch.float32)
            select_iters[f"iter{0}"] = select


        with torch.cuda.amp.autocast(dtype=data_type):
            loss = 0.

            voxel = voxel_iters[f"iter{0}"].detach().to(device).to(torch.float32)
            image = image_iters[f"iter{0}"].detach().to(device).to(torch.float32)

            if args.use_image_aug:
                image_augment = train_utils.get_image_augment()
                image = image_augment(image)

            clip_target = clip_image_embedder(image)
            clip_target = clip_target.to(torch.float32)

            assert not torch.any(torch.isnan(clip_target))

            if epoch < int(args.mixup_pct * args.num_epochs):
                perm = perm_iters[f"iter{0}"].detach().to(device)
                betas = betas_iters[f"iter{0}"].detach().to(device)
                select = select_iters[f"iter{0}"].detach().to(device)

            voxel = align_model(voxel)
            voxel = model.ridge(voxel, SUBJ_LAYER_ID[args.k_subj])
            voxel = voxel.to(torch.float32)

            backbone, clip_voxels, blurry_image_enc_ = model.backbone(voxel)
            
            if args.clip_scale > 0:
                clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1).to(torch.float32)
                clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1).to(torch.float32)

                if epoch < int(args.mixup_pct * args.num_epochs):
                    loss_clip = utils.mixco_nce(
                        clip_voxels_norm,
                        clip_target_norm,
                        temp=.006,
                        perm=perm, betas=betas, select=select)
                else:
                    epoch_temp = soft_loss_temps[epoch - int(args.mixup_pct * args.num_epochs)]
                    loss_clip = utils.soft_clip_loss(
                        clip_voxels_norm,
                        clip_target_norm,
                        temp=epoch_temp)

                loss_clip_total += loss_clip.item()
                loss_clip *= args.clip_scale
                loss += loss_clip

            if args.use_prior:
                loss_prior, prior_out = model.diffusion_prior(text_embed=backbone, image_embed=clip_target)
                loss_prior_total += loss_prior.item()
                loss_prior *= args.prior_scale
                loss += loss_prior

                recon_cossim += nn.functional.cosine_similarity(prior_out, clip_target).mean().item()
                recon_mse += mse(prior_out, clip_target).item()


            if args.blurry_recon:
                image_enc_pred, transformer_feats = blurry_image_enc_

                image_enc = autoenc.encode(2 * image - 1).latent_dist.mode() * 0.18215
                loss_blurry = l1(image_enc_pred, image_enc)
                loss_blurry_total += loss_blurry.item()

                if epoch < int(args.mixup_pct * args.num_epochs):
                    image_enc_shuf = image_enc[perm]
                    betas_shape = [-1] + [1] * (len(image_enc.shape) - 1)

                    image_enc[select] = image_enc[select] * betas[select].reshape(*betas_shape).half() + \
                                        image_enc_shuf[select] * (1 - betas[select]).reshape(*betas_shape).half()

                    image_norm = (image - mean) / std
                    image_aug = (blur_augs(image) - mean) / std
                    _, cnx_embeds = cnx(image_norm)
                    _, cnx_aug_embeds = cnx(image_aug)

                    cont_loss = utils.soft_cont_loss(
                        nn.functional.normalize(transformer_feats.reshape(-1, transformer_feats.shape[-1]), dim=-1),
                        nn.functional.normalize(cnx_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1),
                        nn.functional.normalize(cnx_aug_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1),
                        temp=0.2)
                    loss_blurry_cont_total += cont_loss.item()
          

                    loss += (loss_blurry + 0.1 * cont_loss) * args.blur_scale  # /.18215
            
            if args.clip_scale > 0:
                # forward and backward top 1 accuracy
                labels = torch.arange(len(clip_voxels_norm)).to(clip_voxels_norm.device)
                fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()
                bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()

            if args.blurry_recon:
                with torch.no_grad():
                    # only doing pixcorr eval on a subset of the samples per batch because its costly & slow to compute autoenc.decode()
                    random_samps = np.random.choice(np.arange(len(image)), size=len(image) // 5, replace=False)
                    blurry_recon_images = (
                                autoenc.decode(image_enc_pred[random_samps] / 0.18215).sample / 2 + 0.5).clamp(0, 1)
                    pixcorr = utils.pixcorr(image[random_samps], blurry_recon_images)
                    blurry_pixcorr += pixcorr.item()

            loss.backward()
            optimizer_align.step()
            lr_scheduler_alignM.step()
            torch.cuda.empty_cache()


        # every 100 epochs, evaluate on test set
        if epoch_all % 100 == 0:
            model.eval()
            align_model.eval()
            with torch.no_grad():
                for test_i, (behav, past_behav, future_behav, old_behav) in enumerate(test_dataloader):
                    if test_image is None:
                        image_idx = behav[:,0,0].cpu().long()
                        voxel_idx = behav[:,0,5].cpu().long()
                        voxel = voxel_dataset[f'subj0{args.n_subj}'][voxel_idx].unsqueeze(1)

                        unique_image, sort_indices = torch.unique(image_idx, return_inverse=True)
                        for idx in unique_image:
                            locs = torch.where(idx == image_idx)[0]
                            if len(locs) == 1:
                                locs = locs.repeat(3)
                            elif len(locs) == 2:
                                locs = locs.repeat(2)[:3]
                            assert len(locs) == 3
                            if test_image is None:
                                test_image = torch.Tensor(image_dataset[idx][None])
                                test_voxel = voxel[locs][None]
                            else:
                                test_image = torch.vstack((test_image, torch.Tensor(image_dataset[idx][None])))
                                test_voxel = torch.vstack((test_voxel, voxel[locs][None]))
                    
                    test_indices = torch.arange(len(test_voxel))[:300]
                    voxel = test_voxel[test_indices].to(device).to(torch.float32)
                    image = test_image[test_indices].to(device).to(torch.float32)
                    assert len(image) == 300

                    loss = 0.

                    clip_target = clip_image_embedder(image.float())

                    backbone, clip_voxels, blur_encods = None, None, None
                    for rep in range(3):
                        voxel_align = align_model(voxel[:, rep])
                        voxel_ridge = model.ridge(voxel_align, SUBJ_LAYER_ID[args.k_subj])
                        backbone_rep, clip_voxels_rep, blur_encods_rep = model.backbone(voxel_ridge)

                        backbone = backbone_rep if backbone is None else backbone + backbone_rep
                        clip_voxels = clip_voxels_rep if clip_voxels is None else clip_voxels + clip_voxels_rep
                        blur_encods = blur_encods_rep if blur_encods is None else (blur_encods[0] + blur_encods_rep[0], blur_encods[1] + blur_encods_rep[1])

                    backbone /= 3
                    clip_voxels /= 3
                    blurry_image_enc_ = (blur_encods[0] / 3, blur_encods[1] / 3)

                    if args.clip_scale > 0:
                        clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1).to(torch.float32)
                        clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1).to(torch.float32)

                    # for some evals, only doing a subset of the samples per batch because of computational cost
                    random_samps = np.random.choice(np.arange(len(image)), size=len(image) // 5, replace=False)

                    if args.use_prior:
                        loss_prior, contaminated_prior_out = model.diffusion_prior(text_embed=backbone[random_samps],
                                                                                    image_embed=clip_target[random_samps])
                        test_loss_prior_total += loss_prior.item()
                        loss_prior *= args.prior_scale
                        loss += loss_prior

                    if args.clip_scale > 0:
                        loss_clip = utils.soft_clip_loss(
                            clip_voxels_norm,
                            clip_target_norm,
                            temp=.006)

                        test_loss_clip_total += loss_clip.item()
                        loss_clip = loss_clip * args.clip_scale
                        loss += loss_clip

                    if args.blurry_recon:
                        image_enc_pred, _ = blurry_image_enc_
                        blurry_recon_images = (autoenc.decode(image_enc_pred[random_samps] / 0.18215).sample / 2 + 0.5).clamp(0, 1)
                        pixcorr = utils.pixcorr(image[random_samps].to(torch.float32), blurry_recon_images.to(torch.float32))
                        test_blurry_pixcorr += pixcorr.item()

                    if args.clip_scale > 0:
                        # forward and backward top 1 accuracy
                        labels = torch.arange(len(clip_voxels_norm)).to(clip_voxels_norm.device)
                        test_fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()
                        test_bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()

                    utils.check_loss(loss)

                # save logs
                test_i = 0
                logs = {"test/loss": loss.item(),
                        "test/test_fwd_pct_correct": test_fwd_percent_correct / (test_i + 1),
                        "test/test_bwd_pct_correct": test_bwd_percent_correct / (test_i + 1),
                        "test/loss_clip_total": test_loss_clip_total / (test_i + 1),
                        "test/blurry_pixcorr": test_blurry_pixcorr / (test_i + 1),
                        "test/recon_cossim": test_recon_cossim / (test_i + 1),
                        "test/recon_mse": test_recon_mse / (test_i + 1),
                        "test/loss_prior": test_loss_prior_total / (test_i + 1),
                        }
            
                if args.wandb_log: wandb.log(logs)

                # save ckpt
                if test_fwd_percent_correct > best_retival:
                    checkpoint = {
                        'AlignModel': align_model.state_dict(),
                        'NeuralMapper': NM_model.state_dict(),
                        'FunctionalEmbedder': FE_model.state_dict(),
                    }
                    torch.save(checkpoint, os.path.join(ckpt_path, 'best.pt'))  
                    best_retival = test_fwd_percent_correct
                    print(f"best updated: {test_fwd_percent_correct}")
            

    print("\n===Finished!===\n")


if __name__ == "__main__":
    main()
