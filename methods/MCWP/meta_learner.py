import torch
import torch.nn as nn
from torchmeta.modules import MetaModule, MetaLinear

class MetaWeightPredictor(MetaModule):
    """基于MAML的模态权重预测器"""
    def __init__(self, input_dim=3, hidden_dim=16, output_dim=2):
        super().__init__()
        # 输入：[text_quality, audio_quality, video_quality]
        self.fc1 = MetaLinear(input_dim, hidden_dim)
        self.fc2 = MetaLinear(hidden_dim, output_dim)  # 输出：[video_weight, audio_weight]
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # 权重归一化

    def forward(self, quality_features, params=None):
        # quality_features: [batch_size, 3]（文本、音频、视频质量）
        x = self.fc1(quality_features, params=self.get_subdict(params, 'fc1'))
        x = self.relu(x)
        weights = self.fc2(x, params=self.get_subdict(params, 'fc2'))
        return self.softmax(weights)  # 输出视频和音频的动态权重（和为1）