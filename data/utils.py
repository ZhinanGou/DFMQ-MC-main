import pickle
import numpy as np
import os
import torch
from torch.utils.data import DataLoader

def compute_modality_quality(text_feats, audio_feats, video_feats):
    """
    计算各模态质量特征
    
    参数:
        text_feats: 文本特征，形状为 [batch_size, seq_len, text_dim]
        audio_feats: 音频特征，形状为 [batch_size, seq_len, audio_dim]
        video_feats: 视频特征，形状为 [batch_size, seq_len, video_dim]
    
    返回:
        包含各模态质量分数的字典
    """
    # 1. 文本质量评估（完整性）
    # 计算非填充token比例（假设0为填充值）
    if len(text_feats.shape) == 3:
        text_mask = (text_feats != 0).any(dim=-1).float()  # [batch_size, seq_len]
    else:
        text_mask = (text_feats != 0).float()  # [batch_size, seq_len]
    text_length = torch.sum(text_mask, dim=1)  # [batch_size]
    text_quality = text_length / text_feats.shape[1]  # 归一化到[0,1]

    # 2. 音频质量评估（清晰度）
    # 基于能量和频谱熵的综合评估
    audio_energy = torch.norm(audio_feats, dim=-1).mean(dim=1)  # 平均能量 [batch_size]
    audio_energy_norm = torch.sigmoid(audio_energy - torch.mean(audio_energy))  # 归一化
    
    # 频谱熵（简单近似）
    if audio_feats.shape[-1] > 1:
        audio_power = torch.square(audio_feats).mean(dim=1)  # [batch_size, audio_dim]
        audio_power_norm = audio_power / (torch.sum(audio_power, dim=1, keepdim=True) + 1e-8)
        audio_entropy = -torch.sum(audio_power_norm * torch.log(audio_power_norm + 1e-8), dim=1)
        audio_entropy_norm = audio_entropy / torch.log(torch.tensor(audio_feats.shape[-1], device=audio_feats.device))
    else:
        audio_entropy_norm = torch.ones_like(audio_energy_norm)
    
    audio_quality = (audio_energy_norm + audio_entropy_norm) / 2  # 综合分数

    # 3. 视频质量评估（辨识度）
    # 基于帧间变化和对比度
    video_var = torch.var(video_feats, dim=1).mean(dim=1)  # 帧间变化 [batch_size]
    video_var_norm = torch.sigmoid(video_var - torch.mean(video_var))
    
    video_contrast = torch.std(video_feats, dim=1).mean(dim=1)  # 对比度 [batch_size]
    video_contrast_norm = torch.sigmoid(video_contrast - torch.mean(video_contrast))
    
    video_quality = (video_var_norm + video_contrast_norm) / 2  # 综合分数
      # 新增：打印各模态质量分数的统计信息
    # print("文本质量分数（均值/标准差）：", text_quality.mean().item(), text_quality.std().item())
    # print("音频质量分数（均值/标准差）：", audio_quality.mean().item(), audio_quality.std().item())
    # print("视频质量分数（均值/标准差）：", video_quality.mean().item(), video_quality.std().item())
    # # 若需要查看单样本分数，可打印前几个样本
    # print("前5个样本的文本质量：", text_quality[:5].squeeze().cpu().numpy())
    # print("前5个样本的音频质量：", audio_quality[:5].squeeze().cpu().numpy())
    # print("前5个样本的视频质量：", video_quality[:5].squeeze().cpu().numpy())

    return {
        'text_quality': text_quality.unsqueeze(1),  # [batch_size, 1]
        'audio_quality': audio_quality.unsqueeze(1),
        'video_quality': video_quality.unsqueeze(1)
    }

class QualityEnhancedDataLoader:
    """包装DataLoader，添加模态质量特征"""
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.batch_size = dataloader.batch_size
        self.num_workers = dataloader.num_workers

    def __iter__(self):
        for batch in self.dataloader:
            # 计算模态质量特征
            # 假设batch中包含'text_feats', 'audio_feats', 'video_feats'键
            quality_scores = compute_modality_quality(
                batch['text_feats'],
                batch['audio_feats'],
                batch['video_feats']
            )
            
            # 将质量特征添加到批次中
            batch.update(quality_scores)
            
            # 合并质量特征为一个张量，用于后续元学习权重预测
            batch['quality_features'] = torch.cat([
                batch['text_quality'],
                batch['audio_quality'],
                batch['video_quality']
            ], dim=1)  # [batch_size, 3]
            
            yield batch

    def __len__(self):
        return len(self.dataloader)

def get_dataloader(args, data):
    """创建带有模态质量评估的增强数据加载器"""
    # 创建基础数据加载器
    train_loader = DataLoader(
        data['train'], 
        shuffle=True, 
        batch_size=args.train_batch_size, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    dev_loader = DataLoader(
        data['dev'], 
        batch_size=args.eval_batch_size, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    test_loader = DataLoader(
        data['test'], 
        batch_size=args.eval_batch_size, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    
    # 包装为增强数据加载器，添加质量特征
    return {
        'train': QualityEnhancedDataLoader(train_loader),
        'dev': QualityEnhancedDataLoader(dev_loader),
        'test': QualityEnhancedDataLoader(test_loader)
    }

import pickle
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from matplotlib.ticker import MaxNLocator

# 设置中文显示（解决乱码问题）
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常


def compute_modality_quality(text_feats, audio_feats, video_feats):
    """计算各模态质量特征（已有代码，保持不变）"""
    # 1. 文本质量评估
    if len(text_feats.shape) == 3:
        text_mask = (text_feats != 0).any(dim=-1).float()  # [batch_size, seq_len]
    else:
        text_mask = (text_feats != 0).float()  # [batch_size, seq_len]
    text_length = torch.sum(text_mask, dim=1)  # [batch_size]
    text_quality = text_length / text_feats.shape[1]  # 归一化到[0,1]

    # 2. 音频质量评估
    audio_energy = torch.norm(audio_feats, dim=-1).mean(dim=1)  # 平均能量
    audio_energy_norm = torch.sigmoid(audio_energy - torch.mean(audio_energy))
    
    if audio_feats.shape[-1] > 1:
        audio_power = torch.square(audio_feats).mean(dim=1)
        audio_power_norm = audio_power / (torch.sum(audio_power, dim=1, keepdim=True) + 1e-8)
        audio_entropy = -torch.sum(audio_power_norm * torch.log(audio_power_norm + 1e-8), dim=1)
        audio_entropy_norm = audio_entropy / torch.log(torch.tensor(audio_feats.shape[-1], device=audio_feats.device))
    else:
        audio_entropy_norm = torch.ones_like(audio_energy_norm)
    audio_quality = (audio_energy_norm + audio_entropy_norm) / 2

    # 3. 视频质量评估
    video_var = torch.var(video_feats, dim=1).mean(dim=1)  # 帧间变化
    video_var_norm = torch.sigmoid(video_var - torch.mean(video_var))
    
    video_contrast = torch.std(video_feats, dim=1).mean(dim=1)  # 对比度
    video_contrast_norm = torch.sigmoid(video_contrast - torch.mean(video_contrast))
    video_quality = (video_var_norm + video_contrast_norm) / 2

    # # 打印批次统计信息（已有代码，保持不变）
    # print("文本质量分数（均值/标准差）：", text_quality.mean().item(), text_quality.std().item())
    # print("音频质量分数（均值/标准差）：", audio_quality.mean().item(), audio_quality.std().item())
    # print("视频质量分数（均值/标准差）：", video_quality.mean().item(), video_quality.std().item())

    return {
        'text_quality': text_quality.unsqueeze(1),  # [batch_size, 1]
        'audio_quality': audio_quality.unsqueeze(1),
        'video_quality': video_quality.unsqueeze(1)
    }


class QualityEnhancedDataLoader:
    """包装DataLoader，添加模态质量特征（已有代码，保持不变）"""
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.batch_size = dataloader.batch_size
        self.num_workers = dataloader.num_workers

    def __iter__(self):
        for batch in self.dataloader:
            quality_scores = compute_modality_quality(
                batch['text_feats'],
                batch['audio_feats'],
                batch['video_feats']
            )
            batch.update(quality_scores)
            batch['quality_features'] = torch.cat([
                batch['text_quality'],
                batch['audio_quality'],
                batch['video_quality']
            ], dim=1)  # [batch_size, 3]
            yield batch

    def __len__(self):
        return len(self.dataloader)


def get_dataloader(args, data):
    """创建增强数据加载器（已有代码，保持不变）"""
    train_loader = DataLoader(
        data['train'], 
        shuffle=True, 
        batch_size=args.train_batch_size, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    dev_loader = DataLoader(
        data['dev'], 
        batch_size=args.eval_batch_size, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    test_loader = DataLoader(
        data['test'], 
        batch_size=args.eval_batch_size, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    return {
        'train': QualityEnhancedDataLoader(train_loader),
        'dev': QualityEnhancedDataLoader(dev_loader),
        'test': QualityEnhancedDataLoader(test_loader)
    }


# --------------------------
# 新增：收集所有样本的质量分数
# --------------------------
def collect_all_quality_scores(dataloader):
    """
    遍历数据加载器，收集所有样本的文本、音频、视频质量分数
    返回：包含三个模态质量分数的字典（numpy数组）
    """
    text_scores = []
    audio_scores = []
    video_scores = []
    
    for batch in dataloader:
        # 从batch中提取质量分数（已由QualityEnhancedDataLoader计算）
        text_q = batch['text_quality'].cpu().numpy().flatten()  # 转为numpy并展平
        audio_q = batch['audio_quality'].cpu().numpy().flatten()
        video_q = batch['video_quality'].cpu().numpy().flatten()
        
        # 追加到列表
        text_scores.extend(text_q)
        audio_scores.extend(audio_q)
        video_scores.extend(video_q)
    
    # 转换为numpy数组（便于后续可视化）
    return {
        'text': np.array(text_scores),
        'audio': np.array(audio_scores),
        'video': np.array(video_scores)
    }


# --------------------------
# 新增：可视化质量分数分布
# --------------------------
def plot_quality_distribution(quality_data, save_dir='/home/jiamengyao/MCWP/image'):
    """
    生成各模态质量分数的分布可视化（直方图+折线图组合）
    横轴：质量分数（0-1），纵轴：数据量
    保存路径：save_dir/modality_quality.png
    """
    # 创建保存目录（确保存在）
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'modality_quality(3.0).png')
    
    # 配置画布（1行2列：直方图+核密度图）
    plt.figure(figsize=(14, 6))
    
    # --------------------------
    # 子图1：直方图（展示数据量分布）
    # --------------------------
    plt.subplot(1, 2, 1)
    bins = np.linspace(0, 1, 30)  # 质量分数范围0-1，分30个箱
    
    # 绘制各模态直方图
    plt.hist(quality_data['text'], bins=bins, alpha=0.5, label='text', color='blue', edgecolor='black')
    plt.hist(quality_data['audio'], bins=bins, alpha=0.5, label='audio', color='green', edgecolor='black')
    plt.hist(quality_data['video'], bins=bins, alpha=0.5, label='video', color='red', edgecolor='black')
    
    # 添加统计信息（均值）
    plt.axvline(quality_data['text'].mean(), color='blue', linestyle='--', label=f'text_mean: {quality_data["text"].mean():.2f}')
    plt.axvline(quality_data['audio'].mean(), color='green', linestyle='--', label=f'audio_mean: {quality_data["audio"].mean():.2f}')
    plt.axvline(quality_data['video'].mean(), color='red', linestyle='--', label=f'video_mean: {quality_data["video"].mean():.2f}')
    
    plt.xlabel('Quality_Score')
    plt.ylabel('Data Volume')
    plt.title('Distribution of quality scores for each modality (MELD)')
    plt.xlim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # --------------------------
    # 子图2：核密度图（展示分布趋势）
    # --------------------------
    plt.subplot(1, 2, 2)
    
    # 绘制核密度曲线（平滑的分布趋势）
    sns.kdeplot(quality_data['text'], label='text', color='blue', fill=True, alpha=0.3)
    sns.kdeplot(quality_data['audio'], label='audio', color='green', fill=True, alpha=0.3)
    sns.kdeplot(quality_data['video'], label='video', color='red', fill=True, alpha=0.3)
    
    plt.xlabel('Quality_Score')
    plt.ylabel('Density')
    plt.title('Distribution of quality scores for each modality(MELD) ')
    plt.xlim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"可视化图像已保存至：{save_path}")
