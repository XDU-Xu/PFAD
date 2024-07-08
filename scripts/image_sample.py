"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import argparse
import os
import random
import numpy as np
import torch
import torch as th
import yaml
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import SimpleITK as sitk


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, required=False, default='../conf/brain_sample_config.yml')
    parser.add_argument('--img_dir', type=str, required=False, default='brain')
    parser.add_argument('--save_path', type=str, required=False, default='motion_remove')
    config = parser.parse_args()
    with open(config.conf_path) as f:
        c = yaml.load(f, Loader=yaml.FullLoader)
    add_dict_to_argparser(parser, c)
    args = parser.parse_args()

    dist_util.setup_dist()
    logger.configure()
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()

    sample_fn = diffusion.PFAD_sample
    logger.log("sampling...")

    # load motion-corrupted image
    motion_corrupted_dir = f'../data/{args.img_dir}'
    save_path = f'../results/{args.save_path}/{args.img_dir}'
    os.makedirs(save_path, exist_ok=True)
    motion_corrupted_list = os.listdir(motion_corrupted_dir)
    motion_corrupted_list.sort()

    for i, d in enumerate(motion_corrupted_list):
        d_path = f'{motion_corrupted_dir}/{d}'
        t = sitk.GetArrayFromImage(sitk.ReadImage(d_path)).astype(np.float32)
        t = (t - t.min()) / (t.max() - t.min()) * 2 - 1
        t_d = np.expand_dims(t, axis=(0, 1))
        logger.log(f"Artifact Removal: {d}")
        batch = {
            'GT': t_d,
        }
        for k in batch.keys():
            if not isinstance(batch[k], th.Tensor):
                batch[k] = th.from_numpy(batch[k])
                batch[k] = batch[k].to(dist_util.dev())
        model_kwargs = {"gt": batch['GT'], 'diff': diffusion}

        with th.no_grad():
            sample = sample_fn(
                model,
                (t_d.shape[0], 1, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                device=dist_util.dev(),
                conf=args
            )
            recovered = sample['out'].cpu().detach().numpy().squeeze()
            recovered = recovered.clip(-1, 1)
            recovered = (((recovered + 1) / 2) * 65535).astype(np.uint16)
            recovered = sitk.GetImageFromArray(recovered)
            sitk.WriteImage(recovered, f'{save_path}/clean_{i}.tiff')

    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
