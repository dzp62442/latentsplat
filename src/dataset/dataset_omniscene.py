import copy
import json
import os.path as osp
import pickle as pkl
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from einops import repeat
from torch.utils.data import Dataset

from .dataset import DatasetCfgCommon
from .types import Stage
from .utils_omniscene import load_conditions, load_info
from .view_sampler import ViewSampler


bins_dynamic_demo = [
    "scenee7ef871f77f44331aefdebc24ec034b7_bin010",
    "scenee7ef871f77f44331aefdebc24ec034b7_bin200",
    "scene30ae9c1092f6404a9e6aa0589e809780_bin100",
    "scene84e056bd8e994362a37cba45c0f75558_bin100",
    "scene717053dec2ef4baa913ba1e24c09edff_bin000",
    "scene82240fd6d5ba4375815f8a7fa1561361_bin050",
    "scene724957e51f464a9aa64a16458443786d_bin000",
    "scened3c39710e9da42f48b605824ce2a1927_bin050",
    "scene034256c9639044f98da7562ef3de3646_bin000",
    "scenee0b14a8e11994763acba690bbcc3f56a_bin080",
    "scene7e2d9f38f8eb409ea57b3864bb4ed098_bin150",
    "scene50ff554b3ecb4d208849d042b7643715_bin000",
]


@dataclass
class DatasetOmniSceneCfg(DatasetCfgCommon):
    name: Literal["omniscene"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    skip_bad_shape: bool = True
    near: float = 0.5
    far: float = 100.0
    baseline_scale_bounds: bool = False
    shuffle_val: bool = True
    train_times_per_scene: int = 1
    highres: bool = False


class DatasetOmniScene(Dataset):
    cfg: DatasetOmniSceneCfg
    stage: Stage
    view_sampler: ViewSampler

    data_version: str = "interp_12Hz_trainval"
    dataset_prefix: str = "/datasets/nuScenes"
    camera_types = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    def __init__(
        self,
        cfg: DatasetOmniSceneCfg,
        stage: Stage,
        view_sampler: ViewSampler,
        force_shuffle: bool = False,
        load_rel_depth: bool | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.force_shuffle = force_shuffle
        self.near = cfg.near
        self.far = cfg.far

        self.reso = cfg.image_shape
        self.data_root = str(cfg.roots[0])
        self.load_rel_depth = stage == "test" if load_rel_depth is None else load_rel_depth
        if stage != "test":
            self.load_rel_depth = False

        if stage == "train":
            self.bin_tokens = json.load(
                open(osp.join(self.data_root, self.data_version, "bins_train_3.2m.json"))
            )["bins"]
        elif stage == "val":
            self.bin_tokens = json.load(
                open(osp.join(self.data_root, self.data_version, "bins_val_3.2m.json"))
            )["bins"]
            self.bin_tokens = self.bin_tokens[:30000:3000][:10]
        elif stage == "test":
            self.bin_tokens = json.load(
                open(osp.join(self.data_root, self.data_version, "bins_val_3.2m.json"))
            )["bins"]
            self.bin_tokens = self.bin_tokens[0::14][:2048]
        elif stage == "demo":
            self.bin_tokens = bins_dynamic_demo
        else:
            self.bin_tokens = []

        if self.cfg.test_len > 0 and stage == "test":
            self.bin_tokens = self.bin_tokens[: self.cfg.test_len]

    def __len__(self) -> int:
        return len(self.bin_tokens)

    def __getitem__(self, index: int):
        bin_token = self.bin_tokens[index]
        with open(
            osp.join(self.data_root, self.data_version, "bin_infos_3.2m", bin_token + ".pkl"),
            "rb",
        ) as f:
            bin_info = pkl.load(f)

        sensor_info_center = {
            sensor: bin_info["sensor_info"][sensor][0]
            for sensor in self.camera_types + ["LIDAR_TOP"]
        }

        # 输入视图（key-frame）
        input_img_paths, input_c2ws = [], []
        for cam in self.camera_types:
            info = copy.deepcopy(sensor_info_center[cam])
            img_path, c2w, _ = load_info(info)
            img_path = img_path.replace(self.dataset_prefix, self.data_root)
            input_img_paths.append(img_path)
            input_c2ws.append(c2w)
        input_c2ws = torch.as_tensor(input_c2ws, dtype=torch.float32)

        input_imgs, input_masks, input_cks, input_rel_depths = load_conditions(
            input_img_paths,
            self.reso,
            is_input=True,
            load_rel_depth=self.load_rel_depth,
        )
        input_cks = torch.as_tensor(input_cks, dtype=torch.float32)

        # 输出视图（非关键帧）
        output_img_paths, output_c2ws = [], []
        frame_num = len(bin_info["sensor_info"]["LIDAR_TOP"])
        assert frame_num >= 3, f"only got {frame_num} frames for bin{bin_token}"
        rend_indices = [[1, 2]] * len(self.camera_types)

        for cam_id, cam in enumerate(self.camera_types):
            indices = rend_indices[cam_id]
            for ind in indices:
                info = copy.deepcopy(bin_info["sensor_info"][cam][ind])
                img_path, c2w, _ = load_info(info)
                img_path = img_path.replace(self.dataset_prefix, self.data_root)
                output_img_paths.append(img_path)
                output_c2ws.append(c2w)
        output_c2ws = torch.as_tensor(output_c2ws, dtype=torch.float32)

        output_imgs, output_masks, output_cks, output_rel_depths = load_conditions(
            output_img_paths,
            self.reso,
            is_input=False,
            load_rel_depth=self.load_rel_depth,
        )
        output_cks = torch.as_tensor(output_cks, dtype=torch.float32)

        # 将输入视图拼回输出，保证监督覆盖输入视角
        output_imgs = torch.cat([output_imgs, input_imgs], dim=0)
        output_masks = torch.cat([output_masks, input_masks], dim=0)
        output_c2ws = torch.cat([output_c2ws, input_c2ws], dim=0)
        output_cks = torch.cat([output_cks, input_cks], dim=0)
        if output_rel_depths is not None and input_rel_depths is not None:
            output_rel_depths = torch.cat(
                [output_rel_depths, input_rel_depths], dim=0
            )

        context = {
            "extrinsics": input_c2ws,
            "intrinsics": input_cks,
            "image": input_imgs,
            "near": repeat(
                torch.tensor(self.near, dtype=torch.float32), "-> v", v=len(input_c2ws)
            ),
            "far": repeat(
                torch.tensor(self.far, dtype=torch.float32), "-> v", v=len(input_c2ws)
            ),
            "index": torch.arange(len(input_c2ws)),
        }

        target = {
            "extrinsics": output_c2ws,
            "intrinsics": output_cks,
            "image": output_imgs,
            "near": repeat(
                torch.tensor(self.near, dtype=torch.float32), "-> v", v=len(output_c2ws)
            ),
            "far": repeat(
                torch.tensor(self.far, dtype=torch.float32), "-> v", v=len(output_c2ws)
            ),
            "index": torch.arange(len(output_c2ws)),
            "masks": output_masks,
        }
        if output_rel_depths is not None:
            target["rel_depth"] = output_rel_depths

        return {
            "context": context,
            "target": target,
            "scene": bin_token,
        }
