import json
import numpy as np
from PIL import Image
import torch


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    h, w, c = x.shape
    assert c == 1 or c == 3 or c == 4
    if c == 3:
        return x
    if c == 1:
        return np.concatenate([x, x, x], axis=2)
    if c == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def load_info(info):
    img_path = info["data_path"]
    c2w = info["sensor2lidar_transform"]

    lidar2cam_r = np.linalg.inv(info["sensor2lidar_rotation"])
    lidar2cam_t = info["sensor2lidar_translation"] @ lidar2cam_r.T
    w2c = np.eye(4)
    w2c[:3, :3] = lidar2cam_r.T
    w2c[3, :3] = -lidar2cam_t

    return img_path, c2w, w2c


def load_conditions(img_paths, reso, is_input=False, load_rel_depth=False):
    def maybe_resize(img, tgt_reso, ck):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        resize_flag = False
        if img.height != tgt_reso[0] or img.width != tgt_reso[1]:
            fx, fy, cx, cy = ck[0, 0], ck[1, 1], ck[0, 2], ck[1, 2]
            scale_h, scale_w = tgt_reso[0] / img.height, tgt_reso[1] / img.width
            fx_scaled, fy_scaled, cx_scaled, cy_scaled = (
                fx * scale_w,
                fy * scale_h,
                cx * scale_w,
                cy * scale_h,
            )
            ck = np.array([[fx_scaled, 0, cx_scaled], [0, fy_scaled, cy_scaled], [0, 0, 1]])
            img = img.resize((tgt_reso[1], tgt_reso[0]))
            resize_flag = True
        return np.array(img), ck, resize_flag

    imgs, cks = [], []
    rel_depths = [] if load_rel_depth else None
    masks = []
    for img_path in img_paths:
        param_path = img_path.replace("samples", "samples_param_small")
        param_path = param_path.replace("sweeps", "sweeps_param_small")
        param_path = param_path.replace(".jpg", ".json")
        param = json.load(open(param_path))
        ck = np.array(param["camera_intrinsic"])

        img_path = img_path.replace("samples", "samples_small")
        img_path = img_path.replace("sweeps", "sweeps_small")
        img = Image.open(img_path)
        img, ck, resize_flag = maybe_resize(img, reso, ck)
        ck[0, :] = ck[0, :] / reso[1]
        ck[1, :] = ck[1, :] / reso[0]
        img = HWC3(img)
        imgs.append(img)
        cks.append(ck)

        if load_rel_depth:
            depth_path = img_path.replace("sweeps_small", "sweeps_dpt_small")
            depth_path = depth_path.replace("samples_small", "samples_dpt_small")
            depth_path = depth_path.replace(".jpg", ".npy")
            disp = np.load(depth_path).astype(np.float32)
            if resize_flag:
                disp = Image.fromarray(disp)
                disp = disp.resize((reso[1], reso[0]), Image.BILINEAR)
                disp = np.array(disp)
            ratio = min(disp.max() / (disp.min() + 0.001), 50.0)
            max_val = disp.max()
            min_val = max_val / ratio
            depth = 1 / np.maximum(disp, min_val)
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            rel_depths.append(depth)

        if is_input:
            mask = np.ones(tuple(reso), dtype=np.float32)
        else:
            mask_path = img_path.replace("sweeps_small", "sweeps_mask_small")
            mask_path = mask_path.replace("samples_small", "samples_mask_small")
            mask_path = mask_path.replace(".jpg", ".png")
            mask = Image.open(mask_path).convert("L")
            if resize_flag:
                mask = mask.resize((reso[1], reso[0]), Image.BILINEAR)
            mask = np.array(mask).astype(np.float32)
            mask = mask / 255.0
        masks.append(mask)

    imgs = (
        torch.from_numpy(np.stack(imgs, axis=0)).permute(0, 3, 1, 2).float() / 255.0
    )
    masks = torch.from_numpy(np.stack(masks, axis=0)).bool()
    cks = torch.as_tensor(cks, dtype=torch.float32)
    rel_depths_tensor = (
        None
        if rel_depths is None
        else torch.from_numpy(np.stack(rel_depths, axis=0)).float()
    )

    return imgs, masks, cks, rel_depths_tensor
