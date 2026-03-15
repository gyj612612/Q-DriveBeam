from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _latlon_to_xy(lat: np.ndarray, lon: np.ndarray, lat_ref: float, lon_ref: float) -> Tuple[np.ndarray, np.ndarray]:
    r = 6371000.0
    dlat = np.deg2rad(lat - lat_ref)
    dlon = np.deg2rad(lon - lon_ref)
    x = r * dlon * np.cos(np.deg2rad(lat_ref))
    y = r * dlat
    return x.astype(np.float32), y.astype(np.float32)


def _to_path_string(x: object) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    return str(x)


@dataclass
class Scenario36Prepared:
    gps: np.ndarray
    power: np.ndarray
    labels: np.ndarray
    image_a: np.ndarray
    image_b: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    gps_mean: np.ndarray
    gps_std: np.ndarray
    power_mean: np.ndarray
    power_std: np.ndarray
    scenario_dir: Path


def prepare_scenario36(
    scenario_root: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    image_key_a: str,
    image_key_b: str,
    power_use_log: bool = False,
    power_log_clip_min: float = 1e-6,
    max_samples: Optional[int] = None,
) -> Scenario36Prepared:
    root = Path(scenario_root)
    pkl_path = root / "scenario36.p"
    scenario_dir = root / "scenario36"

    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    lat1 = np.asarray(data["unit1_gps1_lat"], dtype=np.float32)
    lon1 = np.asarray(data["unit1_gps1_lon"], dtype=np.float32)
    lat2 = np.asarray(data["unit2_gps1_lat"], dtype=np.float32)
    lon2 = np.asarray(data["unit2_gps1_lon"], dtype=np.float32)
    lat_ref = float(lat1[0])
    lon_ref = float(lon1[0])
    x1, y1 = _latlon_to_xy(lat1, lon1, lat_ref, lon_ref)
    x2, y2 = _latlon_to_xy(lat2, lon2, lat_ref, lon_ref)

    gps = np.stack(
        [
            x1,
            y1,
            np.asarray(data["unit1_gps1_altitude"], dtype=np.float32),
            np.asarray(data["unit1_gps1_pdop"], dtype=np.float32),
            np.asarray(data["unit1_gps1_hdop"], dtype=np.float32),
            np.asarray(data["unit1_gps1_vdop"], dtype=np.float32),
            x2,
            y2,
            np.asarray(data["unit2_gps1_altitude"], dtype=np.float32),
            np.asarray(data["unit2_gps1_pdop"], dtype=np.float32),
            np.asarray(data["unit2_gps1_hdop"], dtype=np.float32),
            np.asarray(data["unit2_gps1_vdop"], dtype=np.float32),
        ],
        axis=1,
    )

    pwr = [
        np.asarray(data["unit1_pwr1"], dtype=np.float32),
        np.asarray(data["unit1_pwr2"], dtype=np.float32),
        np.asarray(data["unit1_pwr3"], dtype=np.float32),
        np.asarray(data["unit1_pwr4"], dtype=np.float32),
    ]
    power_linear = np.concatenate(pwr, axis=1)
    if power_use_log:
        power_log = 10.0 * np.log10(np.clip(power_linear, power_log_clip_min, None))
        power = np.concatenate([power_linear, power_log], axis=1)
    else:
        power = power_linear

    labels = np.asarray(data["unit1_pwr1_best-beam"], dtype=np.int64) - 1
    image_a = np.asarray([_to_path_string(x) for x in data[image_key_a]], dtype=object)
    image_b = np.asarray([_to_path_string(x) for x in data[image_key_b]], dtype=object)

    valid_mask = labels >= 0
    valid_mask &= np.array([len(x) > 0 for x in image_a], dtype=bool)
    valid_mask &= np.array([len(x) > 0 for x in image_b], dtype=bool)

    gps = gps[valid_mask]
    power = power[valid_mask]
    labels = labels[valid_mask]
    image_a = image_a[valid_mask]
    image_b = image_b[valid_mask]

    if max_samples is not None:
        keep = min(max_samples, len(labels))
        gps = gps[:keep]
        power = power[:keep]
        labels = labels[:keep]
        image_a = image_a[:keep]
        image_b = image_b[:keep]

    n = len(labels)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    t_end = int(n * train_ratio)
    v_end = t_end + int(n * val_ratio)
    train_idx = perm[:t_end]
    val_idx = perm[t_end:v_end]
    test_idx = perm[v_end:]

    gps_mean = gps[train_idx].mean(axis=0)
    gps_std = gps[train_idx].std(axis=0) + 1e-6
    power_mean = power[train_idx].mean(axis=0)
    power_std = power[train_idx].std(axis=0) + 1e-6

    return Scenario36Prepared(
        gps=gps.astype(np.float32),
        power=power.astype(np.float32),
        labels=labels.astype(np.int64),
        image_a=image_a,
        image_b=image_b,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        gps_mean=gps_mean.astype(np.float32),
        gps_std=gps_std.astype(np.float32),
        power_mean=power_mean.astype(np.float32),
        power_std=power_std.astype(np.float32),
        scenario_dir=scenario_dir,
    )


class Scenario36Dataset(Dataset):
    def __init__(
        self,
        prepared: Scenario36Prepared,
        split: str,
        image_size: int = 224,
        use_dual_view: bool = True,
    ) -> None:
        self.prepared = prepared
        self.use_dual_view = use_dual_view
        if split == "train":
            self.indices = prepared.train_idx
        elif split == "val":
            self.indices = prepared.val_idx
        elif split == "test":
            self.indices = prepared.test_idx
        else:
            raise ValueError(f"Unknown split: {split}")

        self.tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.indices)

    def _load_image(self, rel_path: str) -> torch.Tensor:
        img_path = self.prepared.scenario_dir / rel_path
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            return self.tf(img)

    def __getitem__(self, idx: int):
        i = int(self.indices[idx])

        gps = (self.prepared.gps[i] - self.prepared.gps_mean) / self.prepared.gps_std
        power = (self.prepared.power[i] - self.prepared.power_mean) / self.prepared.power_std
        image = self._load_image(self.prepared.image_a[i])
        if self.use_dual_view:
            image_aux = self._load_image(self.prepared.image_b[i])
        else:
            image_aux = image

        label = self.prepared.labels[i]
        return {
            "gps": torch.from_numpy(gps.astype(np.float32)),
            "power": torch.from_numpy(power.astype(np.float32)),
            "image": image,
            "image_aux": image_aux,
            "label": torch.tensor(label, dtype=torch.long),
        }
