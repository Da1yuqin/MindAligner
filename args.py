import argparse


def get_train_args():
    parser = argparse.ArgumentParser(description="Model Training Configuration")

    # MindAligner arguments
    parser.add_argument("--n_subj", type=int, default=1, choices=[1, 2, 5, 7])
    parser.add_argument("--k_subj", type=int, default=2, choices=[1, 2, 5, 7])
    parser.add_argument("--bfa_latent", type=int, default=4096)
    
    parser.add_argument("--lr_b", type=float, default=1e-5, help="learning rate for the BTM model")
    parser.add_argument("--lr_n", type=float, default=1e-5, help="learning rate for the NeuralMapper model")
    parser.add_argument("--lr_f", type=float, default=1e-2, help="learning rate for the FunctionalEmbedder model")
    
    parser.add_argument("--num_sessions", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--iters", type=int, default=99999999)

    # decoding model arguments
    parser.add_argument("--decoding_model_path", type=str, default=".../your_data_path/decoding_model")
    parser.add_argument("--hidden_dim", type=int, default=4096)
    parser.add_argument("--n_blocks", type=int, default=4)

    # data arguments
    parser.add_argument(
        "--data_path", type=str, default=".../your_data_path",
        help="path to where NSD data is stored / where to download it to",
    )

    # training arguments
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="batch size can be increased by 10x if only training retreival submodule and not diffusion prior",
    )
    parser.add_argument(
        "--max_lr", type=float, default=3e-6,
    )
    parser.add_argument(
        "--lr_scheduler_type", type=str, default='cycle', choices=['cycle', 'linear'],
    )
    parser.add_argument(
        "--mixup_pct", type=float, default=.33,
        help="proportion of way through training when to switch from BiMixCo to SoftCLIP",
    )
    parser.add_argument(
        "--use_image_aug", action=argparse.BooleanOptionalAction, default=False,
        help="whether to use image augmentation",
    )

    # loss arguments
    parser.add_argument(
        "--clip_scale", type=float, default=1.,
        help="multiply contrastive loss by this number",
    )
    parser.add_argument(
        "--blurry_recon", action=argparse.BooleanOptionalAction, default=True, 
        help="whether to output blurry reconstructions",
    )
    parser.add_argument(
        "--blur_scale", type=float, default=.5,
        help="multiply loss from blurry recons by this number",
    )
    parser.add_argument(
        "--use_prior", action=argparse.BooleanOptionalAction, default=True,
        help="whether to train diffusion prior (True) or just rely on retrieval part of the pipeline (False)",
    )
    parser.add_argument(
        "--prior_scale", type=float, default=30,
        help="multiply diffusion prior loss by this",
    )

    # logging arguments
    parser.add_argument("--wandb_log", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb_project", type=str, default="MindAligner")

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def get_test_args():
    parser = argparse.ArgumentParser(description="Model Testing Configuration")

    # MindAligner arguments
    parser.add_argument("--n_subj", type=int, default=1, choices=[1, 2, 5, 7])
    parser.add_argument("--k_subj", type=int, default=2, choices=[1, 2, 5, 7])
    parser.add_argument("--bfa_latent", type=int, default=4096)

    parser.add_argument("--plotting", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--new_test", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)

    # decoding model arguments
    parser.add_argument("--decoding_model_path", type=str, default=".../your_data_path/decoding_model") 
    parser.add_argument("--hidden_dim", type=int, default=4096)
    parser.add_argument("--n_blocks", type=int, default=4)

    # data arguments
    parser.add_argument(
        "--data_path", type=str, default=".../your_data_path",
        help="path to where NSD data is stored / where to download it to",
    )

    # loss arguments
    parser.add_argument(
        "--clip_scale", type=float, default=1.,
        help="multiply contrastive loss by this number",
    )
    parser.add_argument(
        "--blurry_recon", action=argparse.BooleanOptionalAction, default=True, 
        help="whether to output blurry reconstructions",
    )
    parser.add_argument(
        "--blur_scale", type=float, default=.5,
        help="multiply loss from blurry recons by this number",
    )
    parser.add_argument(
        "--use_prior", action=argparse.BooleanOptionalAction, default=True,
        help="whether to train diffusion prior (True) or just rely on retrieval part of the pipeline (False)",
    )
    parser.add_argument(
        "--prior_scale", type=float, default=30,
        help="multiply diffusion prior loss by this",
    )

    return parser.parse_args()
