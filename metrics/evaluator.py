import torch
import torch.nn.functional as F


class SaliencyEvaluator:

    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon

    # ---------- UTILITIES ----------

    def _flatten_btchw(self, x):
        #Convert BTCHW → (B*T, C, H, W), otherwise return BCHW unchanged.
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            return x.reshape(B * T, C, H, W)
        return x

    def _prepare(self, pred_logits, sal_gt=None, fix_gt=None):
        pred = torch.sigmoid(pred_logits)          # (BT,1,H,W)
        pred = self._flatten_btchw(pred)

        if sal_gt is not None:
            sal_gt = self._flatten_btchw(sal_gt)

        if fix_gt is not None:
            fix_gt = self._flatten_btchw(fix_gt)

        return pred, sal_gt, fix_gt

    # ---------- METRICS ----------

    @torch.no_grad()
    def kl_div(self, pred_logits, saliency_gt):
        pred, sal, _ = self._prepare(pred_logits, saliency_gt)

        P = pred / (pred.sum(dim=[-2,-1], keepdim=True) + self.epsilon)
        Q = sal  / (sal.sum(dim=[-2,-1], keepdim=True) + self.epsilon)

        # Correct KL(P || Q)
        KL = Q * (torch.log(Q + self.epsilon) - torch.log(P + self.epsilon))
        return KL.sum(dim=[-2,-1]).mean()

    @torch.no_grad()
    def cc(self, pred_logits, saliency_gt):
        pred, sal, _ = self._prepare(pred_logits, saliency_gt)

        pred_n = (pred - pred.mean(dim=[-2,-1], keepdim=True)) / \
                 (pred.std(dim=[-2,-1], keepdim=True) + self.epsilon)

        sal_n = (sal  - sal.mean(dim=[-2,-1], keepdim=True)) / \
                (sal.std(dim=[-2,-1], keepdim=True) + self.epsilon)

        return (pred_n * sal_n).mean()

    @torch.no_grad()
    def nss(self, pred_logits, fixation_gt):
        pred, _, fix = self._prepare(pred_logits, fix_gt=fixation_gt)

        pred_n = (pred - pred.mean(dim=[-2,-1], keepdim=True)) / \
                 (pred.std(dim=[-2,-1], keepdim=True) + self.epsilon)

        fix = (fix > 0).float()  

        return ((pred_n * fix).sum(dim=[-2,-1]) / (fix.sum(dim=[-2,-1]) + self.epsilon)).mean()

    @torch.no_grad()
    def sim(self, pred_logits, saliency_gt):
        pred, sal, _ = self._prepare(pred_logits, saliency_gt)

        P = pred / (pred.sum(dim=[-2,-1], keepdim=True) + self.epsilon)
        Q = sal  / (sal.sum(dim=[-2,-1], keepdim=True) + self.epsilon)

        return torch.min(P, Q).sum(dim=[-2,-1]).mean()

    # ---------- AUC-JUDD ----------
    @torch.no_grad()
    def auc_judd(self, pred_logits, fixation_gt, jitter=True):

        pred, _, fix = self._prepare(pred_logits, fix_gt=fixation_gt)

        pred = pred[:, 0]                    # (BT, H, W)
        fix  = (fix[:, 0] > 0)               # (BT, H, W) bool

        BT = pred.size(0)
        aucs = []

        for i in range(BT):
            p = pred[i].flatten().float()
            f = fix[i].flatten()

            n_fix = int(f.sum().item())
            n_pixels = p.numel()

            if n_fix == 0:
                aucs.append(torch.tensor(float('nan'), device=p.device))
                continue

            # 1. Jitter scaled to map magnitude
            if jitter:
                scale = p.max() - p.min()
                if scale > 0:
                    p = p + torch.rand_like(p) * scale * 1e-7
                else:
                    aucs.append(torch.tensor(0.5, device=p.device))
                    continue

            # 2. Thresholds = saliency values at fixation locations (descending)
            sal_at_fix = p[f]
            thresholds, _ = torch.sort(sal_at_fix, descending=True)

            # 3. Counts of all pixels >= each threshold (Judd convention: denom = n_pixels)
            p_asc, _ = torch.sort(p)
            above = n_pixels - torch.searchsorted(p_asc, thresholds, right=False)
            above = above.float()

            tp_rate = torch.arange(1, n_fix + 1, device=p.device).float() / n_fix
            fp_rate = above / n_pixels

            # 4. Anchor curve at (0,0) and (1,1)
            tp_rate = torch.cat([torch.zeros(1, device=p.device),
                                tp_rate,
                                torch.ones(1, device=p.device)])
            fp_rate = torch.cat([torch.zeros(1, device=p.device),
                                fp_rate,
                                torch.ones(1, device=p.device)])

            # 5. Sort by FPR ascending before trapz
            order = torch.argsort(fp_rate)
            fp_rate = fp_rate[order]
            tp_rate = tp_rate[order]

            aucs.append(torch.trapz(tp_rate, fp_rate))

        if len(aucs) == 0:
            return torch.tensor(float('nan'), device=pred_logits.device)

        return torch.stack(aucs).nanmean()
    
    @torch.no_grad()
    def evaluate(self, pred_logits, saliency_gt, fixation_gt=None, other_fixations=None):
        results = {
            "KL": self.kl_div(pred_logits, saliency_gt),
            "CC": self.cc(pred_logits, saliency_gt),
            "SIM": self.sim(pred_logits, saliency_gt),
        }

        if fixation_gt is not None:
            results["NSS"] = self.nss(pred_logits, fixation_gt)
            results["AUC-Judd"] = self.auc_judd(pred_logits, fixation_gt)

            if other_fixations is not None:
                results["AUC-Shuffled"] = self.auc_shuffled(pred_logits, fixation_gt, other_fixations)

        return results





class SaliencyEvaluator_sample:
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon

    @torch.no_grad()
    def evaluate(self, pred_logits, saliency_gt, fixation_gt=None):
        """
        pred_logits: [N, H, W] or [N, 1, H, W] raw model outputs
        saliency_gt: [N, H, W] or [N, 1, H, W] continuous maps
        fixation_gt: [N, H, W] or [N, 1, H, W] binary masks (1 at fixation)
        """
        # 1. Pre-process (Sigmoid + Flattening to N, 1, H, W)
        pred = torch.sigmoid(pred_logits)
        if pred.dim() == 3: pred = pred.unsqueeze(1)
        if saliency_gt.dim() == 3: saliency_gt = saliency_gt.unsqueeze(1)
        if fixation_gt.dim() == 3: fixation_gt = fixation_gt.unsqueeze(1)

        if pred.shape != saliency_gt.shape:
            B, C, H, W = saliency_gt.shape
            pred = F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=False)
        

        # 2. Metric Calculations
        results = {
            "KL":  self.kl_div(pred, saliency_gt),
            "CC":  self.cc(pred, saliency_gt),
            "SIM": self.sim(pred, saliency_gt),
        }

        if fixation_gt is not None:
            if fixation_gt.dim() == 3: fixation_gt = fixation_gt.unsqueeze(1)
            results["NSS"] = self.nss(pred, fixation_gt)
            results["AUC-J"] = self.auc_judd(pred, fixation_gt)

        return results

    def kl_div(self, pred, sal):
        P = pred / (pred.sum(dim=[-2, -1], keepdim=True) + self.epsilon)
        Q = sal / (sal.sum(dim=[-2, -1], keepdim=True) + self.epsilon)
        kl = Q * (torch.log(Q + self.epsilon) - torch.log(P + self.epsilon))
        return kl.sum(dim=[-2, -1]).mean().item()

    def cc(self, pred, sal):
        def norm(x):
            return (x - x.mean(dim=[-2, -1], keepdim=True)) / \
                   (x.std(dim=[-2, -1], keepdim=True, unbiased=False) + self.epsilon)
        return (norm(pred) * norm(sal)).mean(dim=[-2, -1]).mean().item()

    def sim(self, pred, sal):
        P = pred / (pred.sum(dim=[-2, -1], keepdim=True) + self.epsilon)
        Q = sal / (sal.sum(dim=[-2, -1], keepdim=True) + self.epsilon)
        return torch.min(P, Q).sum(dim=[-2, -1]).mean().item()

    def nss(self, pred, fix):
        pred_n = (pred - pred.mean(dim=[-2, -1], keepdim=True)) / \
                 (pred.std(dim=[-2, -1], keepdim=True, unbiased=False) + self.epsilon)
        fix = (fix > 0).float()
        # Sum of normalized pred at fixations / number of fixations
        score = (pred_n * fix).sum(dim=[-2, -1]) / (fix.sum(dim=[-2, -1]) + self.epsilon)
        return score.mean().item()


    
    @torch.no_grad()
    def auc_judd(self, pred, fix, jitter=True):

        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if fix.dim() == 4:
            fix = fix.squeeze(1)

        aucs = []
        for i in range(pred.size(0)):
            p = pred[i].flatten().float()
            f = (fix[i] > 0.5).flatten()

            n_fix = int(f.sum().item())
            n_pixels = p.numel()

            if n_fix == 0:
                aucs.append(torch.tensor(float('nan'), device=p.device))
                continue

            if jitter:
                scale = p.max() - p.min()
                if scale > 0:
                    p = p + torch.rand_like(p) * scale * 1e-7
                else:
                    # Constant map → AUC is 0.5 by definition
                    aucs.append(torch.tensor(0.5, device=p.device))
                    continue

            sal_at_fix = p[f]
            thresholds, _ = torch.sort(sal_at_fix, descending=True)


            p_asc, _ = torch.sort(p)  # ascending
            above = n_pixels - torch.searchsorted(p_asc, thresholds, right=False)
            above = above.float()

            tp_rate = torch.arange(1, n_fix + 1, device=p.device).float() / n_fix
            fp_rate = above / n_pixels

            tp_rate = torch.cat([torch.zeros(1, device=p.device),
                                tp_rate,
                                torch.ones(1, device=p.device)])
            fp_rate = torch.cat([torch.zeros(1, device=p.device),
                                fp_rate,
                                torch.ones(1, device=p.device)])

            order = torch.argsort(fp_rate)
            fp_rate = fp_rate[order]
            tp_rate = tp_rate[order]

            aucs.append(torch.trapz(tp_rate, fp_rate))

        if len(aucs) == 0:
            return torch.tensor(float('nan'), device=pred.device)

        return torch.stack(aucs).nanmean()