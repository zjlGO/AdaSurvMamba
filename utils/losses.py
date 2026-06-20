import numpy as np
import torch


def nll_loss(hazards, survival, labels, censorship, alpha=0.0, eps=1e-7):
    batch_size = len(labels)
    labels = labels.view(batch_size, 1)
    censorship = censorship.view(batch_size, 1).float()
    if survival is None:
        survival = torch.cumprod(1 - hazards, dim=1)

    survival_padded = torch.cat([torch.ones_like(censorship), survival], dim=1)
    uncensored_loss = -(1 - censorship) * (
        torch.log(torch.gather(survival_padded, 1, labels).clamp(min=eps))
        + torch.log(torch.gather(hazards, 1, labels).clamp(min=eps))
    )
    censored_loss = -censorship * torch.log(torch.gather(survival_padded, 1, labels + 1).clamp(min=eps))
    loss = (1 - alpha) * (censored_loss + uncensored_loss) + alpha * uncensored_loss
    return loss.mean()


def ce_loss(hazards, survival, labels, censorship, alpha=0.0, eps=1e-7):
    batch_size = len(labels)
    labels = labels.view(batch_size, 1)
    censorship = censorship.view(batch_size, 1).float()
    if survival is None:
        survival = torch.cumprod(1 - hazards, dim=1)

    survival_padded = torch.cat([torch.ones_like(censorship), survival], dim=1)
    reg = -(1 - censorship) * (
        torch.log(torch.gather(survival_padded, 1, labels).clamp(min=eps))
        + torch.log(torch.gather(hazards, 1, labels).clamp(min=eps))
    )
    ce = -censorship * torch.log(torch.gather(survival, 1, labels).clamp(min=eps))
    ce -= (1 - censorship) * torch.log((1 - torch.gather(survival, 1, labels)).clamp(min=eps))
    return ((1 - alpha) * ce + alpha * reg).mean()


class NLLSurvLoss:
    def __init__(self, alpha=0.0):
        self.alpha = alpha

    def __call__(self, hazards, survival, labels, censorship, alpha=None):
        return nll_loss(hazards, survival, labels, censorship, self.alpha if alpha is None else alpha)


class CrossEntropySurvLoss:
    def __init__(self, alpha=0.0):
        self.alpha = alpha

    def __call__(self, hazards, survival, labels, censorship, alpha=None):
        return ce_loss(hazards, survival, labels, censorship, self.alpha if alpha is None else alpha)


class CoxSurvLoss:
    def __call__(self, hazards, survival, labels=None, censorship=None):
        if censorship is None:
            raise ValueError("censorship is required for CoxSurvLoss.")
        batch_size = len(survival)
        risk_set = np.zeros((batch_size, batch_size), dtype=np.float32)
        survival_np = survival.detach().cpu().numpy().reshape(-1)
        for i in range(batch_size):
            risk_set[i] = survival_np >= survival_np[i]
        risk_set = torch.tensor(risk_set, device=hazards.device)
        theta = hazards.reshape(-1)
        exp_theta = torch.exp(theta)
        return -torch.mean((theta - torch.log(torch.sum(exp_theta * risk_set, dim=1))) * (1 - censorship))
