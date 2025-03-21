import torch
import torch.nn as nn
import torch.nn.functional as F

from ..attack import PatchAttack


class TPGD(PatchAttack):
    r"""
    PGD based on KL-Divergence loss in the paper 'Theoretically Principled Trade-off between Robustness and Accuracy'
    [https://arxiv.org/abs/1901.08573]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): strength of the attack or maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.TPGD(model, eps=8/255, alpha=2/255, steps=10)
        >>> adv_images = attack(images)

    """

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10):
        super().__init__("TPGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.supported_mode = ["default"]

    def forward(self, images, labels=None):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        logit_ori = self.get_logits(images).detach()

        adv_images = images + 0.001 * torch.randn_like(images)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        loss = nn.KLDivLoss(reduction="sum")

        for _ in range(self.steps):
            adv_images.requires_grad = True
            logit_adv = self.get_logits(adv_images)

            # Calculate loss
            cost = loss(F.log_softmax(logit_adv, dim=1), F.softmax(logit_ori, dim=1))

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
