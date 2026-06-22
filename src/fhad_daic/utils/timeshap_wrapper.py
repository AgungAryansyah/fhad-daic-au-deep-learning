import numpy as np
import torch


class FusionTCNTimeShapWrapper:
    def __init__(
        self,
        model: torch.nn.Module,
        vis_baseline: np.ndarray,
        aud_baseline: np.ndarray,
        aux_baseline_v: dict[str, float],
        aux_baseline_a: dict[str, float],
        device: torch.device,
        target_class: int = 1,
    ):
        self.model = model
        self.vis_baseline = vis_baseline.astype(np.float32)
        self.aud_baseline = aud_baseline.astype(np.float32)
        self.aux_baseline_v = aux_baseline_v
        self.aux_baseline_a = aux_baseline_a
        self.device = device
        self.target_class = target_class
        self.model.eval()

    def predict_visual(self, x_np: np.ndarray) -> np.ndarray:
        return self._predict(x_np, modality="visual")

    def predict_audio(self, x_np: np.ndarray) -> np.ndarray:
        return self._predict(x_np, modality="audio")

    def _predict(self, x_np: np.ndarray, modality: str) -> np.ndarray:
        scores = []
        for i in range(len(x_np)):
            if modality == "visual":
                X_v = torch.from_numpy(x_np[i]).unsqueeze(0).to(self.device)
                mask_v = torch.ones(1, X_v.size(1), dtype=torch.bool, device=self.device)
                X_a = torch.from_numpy(self.aud_baseline).unsqueeze(0).to(self.device)
                mask_a = torch.ones(1, X_a.size(1), dtype=torch.bool, device=self.device)
            else:
                X_a = torch.from_numpy(x_np[i]).unsqueeze(0).to(self.device)
                mask_a = torch.ones(1, X_a.size(1), dtype=torch.bool, device=self.device)
                X_v = torch.from_numpy(self.vis_baseline).unsqueeze(0).to(self.device)
                mask_v = torch.ones(1, X_v.size(1), dtype=torch.bool, device=self.device)

            av = {k: torch.full((1,), v, device=self.device) for k, v in self.aux_baseline_v.items()}
            aa = {k: torch.full((1,), v, device=self.device) for k, v in self.aux_baseline_a.items()}

            with torch.no_grad():
                logits = self.model(X_v, mask_v, X_a, mask_a, av, aa)
                scores.append(logits[:, self.target_class].cpu().numpy())

        return np.concatenate(scores, axis=0)[:, np.newaxis]
