# class Param():
    
#     def __init__(self, args):

#         self.common_param = self._get_common_parameters(args)
#         self.hyper_param = self._get_hyper_parameters(args)
    
    
#     def _get_common_parameters(self, args):
#         """
#             padding_mode (str): The mode for sequence padding ('zero' or 'normal').
#             padding_loc (str): The location for sequence padding ('start' or 'end'). 
#             eval_monitor (str): The monitor for evaluation ('loss' or metrics, e.g., 'f1', 'acc', 'precision', 'recall').  
#             need_aligned: (bool): Whether to perform data alignment between different modalities.
#             train_batch_size (int): The batch size for training.
#             eval_batch_size (int): The batch size for evaluation. 
#             test_batch_size (int): The batch size for testing.
#             wait_patience (int): Patient steps for Early Stop.
#         """
#         common_parameters = {
#             'max_cons_seq_length': 768 , # 根据实际文本序列长度调整（参考 text_feat_dim 或数据中的文本长度）
#             'feats_processing_type': 'padding',
#             'padding_mode': 'zero',
#             'padding_loc': 'end',
#             'need_aligned': True,
#             'eval_monitor': ['f1'],
#             'train_batch_size': 16,
#             'eval_batch_size': 8,
#             'test_batch_size': 8,
#             'wait_patience': 8,
#             'num_train_epochs': 100,
#         }
#         return common_parameters

#     def _get_hyper_parameters(self, args):
#         """
#         Args:
#             新增元学习参数，用于实现基于模态质量的动态权重调整
#         """
#         hyper_parameters = {
            
#             # 原有方法参数保持不变
#             'warmup_proportion': 0.1,
#             'grad_clip': [1.0],
#             'lr': [2e-5],  # 主模型学习率
#             'weight_decay': 0.1,
#             'mag_aligned_method': ['ctc'],
#             'aligned_method': ['sim'],
#             'shared_dim': [256],
#             'eps': 1e-9,
#             'loss': 'SupCon',
#             'temperature': [0.7],
#             'beta_shift': [0.01],
#             'dropout_prob': [0.3],
#             'use_ctx': True,
#             'prompt_len': 3,
#             'nheads': [16],
#             'n_levels': [5],
#             'attn_dropout': [0.3],
#             'relu_dropout': 0.0,
#             'embed_dropout': [0.3],
#             'res_dropout': 0.1,
#             'attn_mask': True,
#             'label_len': 4,
            
#             # 新增：元学习权重预测器参数
#             'modal_consistency_weight': 0.5,
#             'meta_hidden_dim': [64],  # 元学习网络隐藏层维度
#             'meta_lr': [1e-4],      # 元学习器单独学习率（高于主模型学习率）
#             'meta_loss_weight': [0.1], # 元学习损失在总损失中的权重
#             'meta_inner_steps': 4,     # 元学习内循环适应步数
#             'meta_inner_lr': 0.02      # 元学习内循环学习率
#         }
#         return hyper_parameters
class Param():
    
    def __init__(self, args):

        self.common_param = self._get_common_parameters(args)
        self.hyper_param = self._get_hyper_parameters(args)
    
    
    def _get_common_parameters(self, args):
        """
            padding_mode (str): The mode for sequence padding ('zero' or 'normal').
            padding_loc (str): The location for sequence padding ('start' or 'end'). 
            eval_monitor (str): The monitor for evaluation ('loss' or metrics, e.g., 'f1', 'acc', 'precision', 'recall').  
            need_aligned: (bool): Whether to perform data alignment between different modalities.
            train_batch_size (int): The batch size for training.
            eval_batch_size (int): The batch size for evaluation. 
            test_batch_size (int): The batch size for testing.
            wait_patience (int): Patient steps for Early Stop.
        """
        common_parameters = {
            'max_cons_seq_length': 768 , # 根据实际文本序列长度调整（参考 text_feat_dim 或数据中的文本长度）
            'feats_processing_type': 'padding',
            'padding_mode': 'zero',
            'padding_loc': 'end',
            'need_aligned': True,
            'eval_monitor': ['f1'],
            'train_batch_size': 32,
            'eval_batch_size': 16,
            'test_batch_size': 16,
            'wait_patience': 8,
            'num_train_epochs': 100,
        }
        return common_parameters

    def _get_hyper_parameters(self, args):
        """
        Args:
            新增元学习参数，用于实现基于模态质量的动态权重调整
        """
        hyper_parameters = {
            
            # 原有方法参数保持不变
            'warmup_proportion': 0.1,
            'grad_clip': [-1.0],
            'lr': [2e-5],  # 主模型学习率
            'weight_decay': 0.01,
            'mag_aligned_method': ['ctc'],
            'aligned_method': ['sim'],
            'shared_dim': [256],
            'eps': 1e-9,
            'loss': 'SupCon',
            'temperature': [1],
            'beta_shift': [0.01],
            'dropout_prob': [0.1],
            'use_ctx': True,
            'prompt_len': 3,
            'nheads': [16],
            'n_levels': [5],
            'attn_dropout': [0.1],
            'relu_dropout': 0.1,
            'embed_dropout': [0.2],
            'res_dropout': 0.1,
            'attn_mask': True,
            'label_len': 4,
            
            # 新增：元学习权重预测器参数
            'modal_consistency_weight': 0.2,
            'meta_hidden_dim': [64],  # 元学习网络隐藏层维度
            'meta_lr': [1.5e-4],      # 元学习器单独学习率（高于主模型学习率）
            'meta_loss_weight': [0.1], # 元学习损失在总损失中的权重
            'meta_inner_steps': 4,     # 元学习内循环适应步数
            'meta_inner_lr': 0.02      # 元学习内循环学习率
        }
        return hyper_parameters
    