import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from tabulate import tabulate

from ..misc.image_io import load_image, save_image
from ..visualization.annotation import add_label
from ..visualization.layout import add_border, hcat
from .evaluation_cfg import EvaluationCfg
from .metrics import compute_dists, compute_lpips, compute_pcc, compute_psnr, compute_ssim


class MetricComputer(LightningModule):
    cfg: EvaluationCfg

    def __init__(self, cfg: EvaluationCfg) -> None:
        super().__init__()
        self.cfg = cfg

    def on_test_epoch_start(self) -> None:
        self.scores = dict(dists={}, lpips={}, ssim={}, psnr={}, pcc={})

    def test_step(self, batch, batch_idx):
        scene = batch["scene"][0]
        b, cv, _, _, _ = batch["context"]["image"].shape
        assert b == 1 and cv == 2
        _, v, _, _, _ = batch["target"]["image"].shape

        context_index_str = "_".join(map(str, sorted(batch["context"]["index"][0].tolist())))

        # Skip scenes.
        for method in self.cfg.methods:
            if not (method.path / scene).exists():
                print(f'Skipping "{scene}".')
                return

        # Load the images.
        all_images = {}
        try:
            for method in self.cfg.methods:
                images = [
                    load_image(method.path / scene / context_index_str / f"color/{index.item():0>6}.png")
                    for index in batch["target"]["index"][0]
                ]
                all_images[method.key] = torch.stack(images).to(self.device)
        except FileNotFoundError:
            print(f'Skipping "{scene}".')
            return

        rel_depth = batch["target"].get("rel_depth")
        all_depths = {}
        if rel_depth is not None:
            rel_depth = rel_depth[0].to(self.device)
            for method in self.cfg.methods:
                try:
                    depths = [
                        np.load(method.path / scene / context_index_str / f"depth/{index.item():0>6}.npy")
                        for index in batch["target"]["index"][0]
                    ]
                except FileNotFoundError:
                    continue
                all_depths[method.key] = (
                    torch.from_numpy(np.stack(depths)).float().to(self.device)
                )

        # Compute metrics.
        all_metrics = {}
        rgb_gt = batch["target"]["image"][0]
        for key, images in all_images.items():
            dists = compute_dists(rgb_gt, images).mean()
            lpips = compute_lpips(rgb_gt, images).mean()
            ssim = compute_ssim(rgb_gt, images).mean()
            psnr = compute_psnr(rgb_gt, images).mean()
            for metric, score in zip(
                ("dists", "lpips", "ssim", "psnr"),
                (dists, lpips, ssim, psnr)
            ):
                if scene not in self.scores[metric]:
                    self.scores[metric][scene] = {}
                self.scores[metric][scene][key] = score.item()
            all_metrics = {
                **all_metrics,
                f"dists_{key}": dists,
                f"lpips_{key}": lpips,
                f"ssim_{key}": ssim,
                f"psnr_{key}": psnr,
            }
            if key in all_depths and rel_depth is not None:
                pred_depth = all_depths[key]
                if pred_depth.shape[-2:] != rel_depth.shape[-2:]:
                    pred_depth = F.interpolate(
                        pred_depth.unsqueeze(1),
                        size=rel_depth.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)
                pcc = compute_pcc(rel_depth, pred_depth)
                if scene not in self.scores["pcc"]:
                    self.scores["pcc"][scene] = {}
                self.scores["pcc"][scene][key] = pcc.item()
                all_metrics = {
                    **all_metrics,
                    f"pcc_{key}": pcc,
                }
        self.log_dict(all_metrics)
        self.print_preview_metrics(all_metrics)

        # Skip the rest if no side-by-side is needed.
        if self.cfg.side_by_side_path is None:
            return

        # Create side-by-side.
        scene_key = f"{batch_idx:0>6}_{scene}"
        for i in range(v):
            true_index = batch["target"]["index"][0, i]
            row = [add_label(batch["target"]["image"][0, i], "Ground Truth")]
            for method in self.cfg.methods:
                image = all_images[method.key][i]
                image = add_label(image, method.name)
                row.append(image)
            start_frame = batch["target"]["index"][0, 0]
            end_frame = batch["target"]["index"][0, -1]
            label = f"Scene {batch['scene'][0]} (frames {start_frame} to {end_frame})"
            row = add_border(add_label(hcat(*row), label, font_size=16))
            save_image(
                row,
                self.cfg.side_by_side_path / scene_key / context_index_str / f"{true_index:0>6}.png",
            )

        # Create an animation.
        if self.cfg.animate_side_by_side:
            (self.cfg.side_by_side_path / "videos").mkdir(exist_ok=True, parents=True)
            command = (
                'ffmpeg -y -framerate 30 -pattern_type glob -i "*.png"  -c:v libx264 '
                '-pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"'
            )
            os.system(
                f"cd {self.cfg.side_by_side_path / scene_key / context_index_str} && {command} "
                f"{Path.cwd()}/{self.cfg.side_by_side_path}/videos/{scene_key}.mp4"
            )

    def print_preview_metrics(self, metrics: dict[str, float]) -> None:
        if getattr(self, "running_metrics", None) is None:
            self.running_metrics = metrics
            self.running_metric_steps = 1
        else:
            s = self.running_metric_steps
            self.running_metrics = {
                k: ((s * v) + metrics[k]) / (s + 1)
                for k, v in self.running_metrics.items()
            }
            self.running_metric_steps += 1

        metrics = ["psnr", "lpips", "dists", "ssim"]
        if all(
            f"pcc_{method.key}" in self.running_metrics for method in self.cfg.methods
        ):
            metrics.append("pcc")

        table = []
        for method in self.cfg.methods:
            row = [
                f"{self.running_metrics[f'{metric}_{method.key}']:.3f}"
                for metric in metrics
            ]
            table.append((method.key, *row))

        headers = ["Method", "PSNR (dB)", "LPIPS", "DISTS", "SSIM"]
        if "pcc" in metrics:
            headers.append("PCC")
        table = tabulate(table, headers)
        print(table)
