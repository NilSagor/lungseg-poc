"""Joint training loops for baseline + diffusion refiner."""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from lungseg.losses import BCEDiceLoss, BoundaryLoss


class JointTrainer:
    """Encapsulates warmup / alternating / end-to-end joint training.

    mode:
        "warmup"      : baseline frozen, refiner-only loss
        "alternating" : one step of baseline seg, one step of refiner diff
        "end_to_end"  : single step, combined loss backprop through both
    """

    def __init__(self, baseline, refiner, schedule, cfg, device):
        self.baseline = baseline
        self.refiner = refiner
        self.schedule = schedule
        self.cfg = cfg
        self.device = device

        t = cfg["training"]
        self.w_bnd = t["boundary_weight"]
        self.w_diff = t["diffusion_weight"]
        self.use_boundary = cfg.get("joint", {}).get("use_boundary", False)

        self.seg_loss = BCEDiceLoss()
        self.bnd_loss = BoundaryLoss() if self.use_boundary else None

        self.opt_baseline = torch.optim.AdamW(
            baseline.parameters(),
            lr=t["learning_rate"],
            weight_decay=t["weight_decay"],
        )
        self.opt_refiner = torch.optim.AdamW(
            refiner.parameters(),
            lr=t["learning_rate"],
            weight_decay=t["weight_decay"],
        )
        amp = t["amp"] and device.type == "cuda"
        self.scaler_b = torch.cuda.amp.GradScaler(enabled=amp)
        self.scaler_r = torch.cuda.amp.GradScaler(enabled=amp)
        self.amp = amp

    # ------------------------------------------------------------------
    def _seg_step(self, image, label):
        self.baseline.train()
        self.opt_baseline.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.amp):
            logits = self.baseline(image)
            loss = self.seg_loss(logits, label)
            if self.bnd_loss is not None:
                loss = loss + self.w_bnd * self.bnd_loss(logits, label)
        self.scaler_b.scale(loss).backward()
        self.scaler_b.step(self.opt_baseline)
        self.scaler_b.update()
        return loss.item()

    def _diff_step(self, image, label, detach_baseline=True):
        self.refiner.train()
        self.opt_refiner.zero_grad(set_to_none=True)
        with torch.no_grad() if detach_baseline else torch.enable_grad():
            initial_logits = self.baseline(image)
        if detach_baseline:
            initial_logits = initial_logits.detach()

        noise = torch.randn_like(label)
        t = torch.randint(0, self.schedule.timesteps, (image.shape[0],), device=self.device)
        noisy = self.schedule.add_noise(label, noise, t)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.amp):
            eps_hat = self.refiner(image, initial_logits, noisy, t)
            loss = self.w_diff * F.mse_loss(eps_hat, noise)

        self.scaler_r.scale(loss).backward()
        self.scaler_r.step(self.opt_refiner)
        self.scaler_r.update()
        return loss.item()

    def _e2e_step(self, image, label):
        self.baseline.train()
        self.refiner.train()
        self.opt_baseline.zero_grad(set_to_none=True)
        self.opt_refiner.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.amp):
            initial_logits = self.baseline(image)
            seg = self.seg_loss(initial_logits, label)
            if self.bnd_loss is not None:
                seg = seg + self.w_bnd * self.bnd_loss(initial_logits, label)

            noise = torch.randn_like(label)
            t = torch.randint(0, self.schedule.timesteps, (image.shape[0],), device=self.device)
            noisy = self.schedule.add_noise(label, noise, t)
            eps_hat = self.refiner(image, initial_logits.detach(), noisy, t)
            diff = self.w_diff * F.mse_loss(eps_hat, noise)

            loss = seg + diff

        self.scaler_b.scale(loss).backward()
        self.scaler_b.step(self.opt_baseline)
        self.scaler_b.step(self.opt_refiner)
        self.scaler_b.update()
        return loss.item()

    # ------------------------------------------------------------------
    def train_one_epoch(self, loader, mode="alternating"):
        total = 0.0
        for batch in tqdm(loader, desc=f"joint[{mode}]", leave=False):
            image = batch["image"].to(self.device)
            label = batch["label"].to(self.device).float()

            if mode == "warmup":
                l = self._diff_step(image, label, detach_baseline=True)
            elif mode == "alternating":
                l1 = self._seg_step(image, label)
                l2 = self._diff_step(image, label, detach_baseline=True)
                l = 0.5 * (l1 + l2)
            elif mode == "end_to_end":
                l = self._e2e_step(image, label)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            total += l
        return total / max(len(loader), 1)