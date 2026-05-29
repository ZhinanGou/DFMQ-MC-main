import torch
from torch import nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all', eps=1e-8):  # 新增eps参数
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.eps = eps  # 用于避免除零

    def forward(self, features, labels=None, mask=None):
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        features = F.normalize(features, dim=2)
        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # 新增：检查mask是否有效（至少有一个正样本）
        if mask.sum() == 0:
            raise ValueError("No positive pairs found in mask. Check labels or mask input.")

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # 计算logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 扩展mask并排除自身对比
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # 新增：检查扩展后的mask是否有正样本（避免全零）
        if mask.sum() == 0:
            raise ValueError("No valid positive pairs after masking self-contrast. Adjust labels or batch size.")

        # 计算log概率
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + self.eps)  # 加eps避免log(0)

        # 计算正样本的平均log概率（核心修改：避免除零）
        mask_sum = mask.sum(1)  # 每个样本的正样本数量
        mask_sum = torch.clamp(mask_sum, min=self.eps)  # 确保分母至少为eps，避免除零
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # 计算损失
        loss = -mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
