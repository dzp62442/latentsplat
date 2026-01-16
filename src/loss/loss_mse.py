from dataclasses import dataclass
from typing import Literal

from jaxtyping import Float
from torch import Tensor

from ..model.types import Prediction, GroundTruth
from .loss import LossCfg, Loss


@dataclass
class LossMseCfg(LossCfg):
    name: Literal["mse"] = "mse"


class LossMse(Loss):
    def unweighted_loss(
        self,
        prediction: Prediction,
        gt: GroundTruth,
    ) -> Float[Tensor, ""]:
        delta = prediction.image - gt.image
        mask = gt.mask
        if mask is not None and mask.max() > 0.5 and mask.min() < 0.5:
            mask = mask.bool().unsqueeze(2).expand_as(delta)
            delta = delta[~mask]
        return (delta**2).mean()
