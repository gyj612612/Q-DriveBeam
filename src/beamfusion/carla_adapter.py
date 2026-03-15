from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .models import DetrIemfBeamModel


@dataclass
class CarlaAdapterConfig:
    image_size: int = 224
    topk: int = 5


class CarlaBeamAdapter:
    def __init__(
        self,
        model: DetrIemfBeamModel,
        device: str = "cpu",
        cfg: Optional[CarlaAdapterConfig] = None,
    ) -> None:
        self.model = model.eval()
        self.device = torch.device(device)
        self.cfg = cfg or CarlaAdapterConfig()
        self.tf = transforms.Compose(
            [
                transforms.Resize((self.cfg.image_size, self.cfg.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        pil = Image.fromarray(image)
        return self.tf(pil.convert("RGB"))

    @torch.no_grad()
    def predict(
        self,
        rgb: np.ndarray,
        gps: np.ndarray,
        power: np.ndarray,
        rgb_aux: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        image = self._image_to_tensor(rgb).unsqueeze(0).to(self.device)
        image_aux = self._image_to_tensor(rgb_aux if rgb_aux is not None else rgb).unsqueeze(0).to(self.device)
        gps_t = torch.tensor(gps, dtype=torch.float32).unsqueeze(0).to(self.device)
        power_t = torch.tensor(power, dtype=torch.float32).unsqueeze(0).to(self.device)

        out = self.model(
            {
                "image": image,
                "image_aux": image_aux,
                "gps": gps_t,
                "power": power_t,
            }
        )
        probs = torch.softmax(out["fused_logits"], dim=-1)
        values, indices = torch.topk(probs, k=min(self.cfg.topk, probs.size(-1)), dim=-1)
        return {
            "topk_beams": indices.squeeze(0).cpu().numpy(),
            "topk_probs": values.squeeze(0).cpu().numpy(),
            "gate_weights": out["weights"].squeeze(0).cpu().numpy(),
        }
