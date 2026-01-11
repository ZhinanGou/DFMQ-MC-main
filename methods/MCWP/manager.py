# import os
# import torch
# import torch.nn.functional as F
# import logging
# from torch import nn
# from utils.functions import restore_model, save_model, EarlyStopping
# from tqdm import trange, tqdm
# from data.utils import collect_all_quality_scores, get_dataloader, plot_quality_distribution
# from utils.metrics import AverageMeter, Metrics
# from transformers import AdamW, get_linear_schedule_with_warmup
# from .model import MCWP_Wrapper # 修改类名引用
# from .loss import SupConLoss
# import numpy as np


# __all__ = ['MCWP_Manager']

# class MCWP_Manager:

#     def __init__(self, args, data, model):
             
#         self.logger = logging.getLogger(args.logger_name)
#         # self.device, self.model = model.device, model.model
#         self.device = torch.device(f'cuda:{args.gpu_id}')
#         args.device = self.device
#         self.model = MCWP_Wrapper(args)  # 修改类名引用
#         self.model.to(self.device)

#         self.optimizer, self.scheduler = self._set_optimizer(args, self.model)
#         # mm_dataloader = get_dataloader(args, data.mm_data)
#         # self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
#         #     data.mm_dataloader['train'], data.mm_dataloader['dev'], data.mm_dataloader['test']
#         mm_dataloader = get_dataloader(args, data.mm_data)
#         self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
#             mm_dataloader['train'], mm_dataloader['dev'], mm_dataloader['test']
                    
#         self.args = args
#         self.criterion = nn.CrossEntropyLoss()
#         self.cons_criterion = SupConLoss(temperature=args.temperature)
#         self.metrics = Metrics(args)
#         # 新增：元学习损失权重（从配置读取）
#         self.meta_loss_weight = args.meta_loss_weight if hasattr(args, 'meta_loss_weight') else 0.1
        
#         if args.train:
#             self.best_eval_score = 0
#         else:
#             self.model = restore_model(self.model, args.model_output_path, self.device)
            
#     def _set_optimizer(self, args, model):
        
#         param_optimizer = list(model.named_parameters())
#         no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
#         optimizer_grouped_parameters = [
#             {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
#             {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
#         ]
        
#         optimizer = AdamW(optimizer_grouped_parameters, lr = args.lr, correct_bias=False)
        
#         num_train_optimization_steps = int(args.num_train_examples / args.train_batch_size) * args.num_train_epochs
#         num_warmup_steps= int(args.num_train_examples * args.num_train_epochs * args.warmup_proportion / args.train_batch_size)
        
#         scheduler = get_linear_schedule_with_warmup(optimizer,
#                                                     num_warmup_steps=num_warmup_steps,
#                                                     num_training_steps=num_train_optimization_steps)
        
#         return optimizer, scheduler
#     def _set_optimizer(self, args, model):
#         # 收集所有参数ID用于去重
#         all_param_ids = set()
#         optimizer_grouped_parameters = []
        
#         # 基础参数分组（不含元学习参数）
#         param_optimizer = list(model.named_parameters())
#         no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        
#         # 第一组：带权重衰减的普通参数
#         normal_params = [
#             p for n, p in param_optimizer 
#             if not any(nd in n for nd in no_decay) 
#             and 'meta_predictor' not in n  # 排除元学习参数
#         ]
#         if normal_params:
#             optimizer_grouped_parameters.append({
#                 'params': normal_params,
#                 'weight_decay': args.weight_decay
#             })
#             all_param_ids.update(id(p) for p in normal_params)
        
#         # 第二组：不带权重衰减的普通参数
#         no_decay_params = [
#             p for n, p in param_optimizer 
#             if any(nd in n for nd in no_decay) 
#             and 'meta_predictor' not in n  # 排除元学习参数
#         ]
#         if no_decay_params:
#             optimizer_grouped_parameters.append({
#                 'params': no_decay_params,
#                 'weight_decay': 0.0
#             })
#             all_param_ids.update(id(p) for p in no_decay_params)
        
#         # 第三组：元学习预测器参数（单独设置学习率）
#         if hasattr(model.model.bert.MAG, 'meta_predictor'):
#             # 收集元学习参数并计算正确的weight_decay
#             meta_params = []
#             meta_weight_decay = []
#             for n, p in model.model.bert.MAG.meta_predictor.named_parameters():
#                 if id(p) not in all_param_ids:  # 确保不重复
#                     meta_params.append(p)
#                     # 根据参数名判断是否需要weight_decay
#                     meta_weight_decay.append(args.weight_decay if 'bias' not in n else 0.0)
#                     all_param_ids.add(id(p))
            
#             if meta_params:
#                 # 处理权重衰减（如果所有参数衰减相同则简化，否则使用参数组列表）
#                 if len(set(meta_weight_decay)) == 1:
#                     optimizer_grouped_parameters.append({
#                         'params': meta_params,
#                         'lr': args.meta_lr if hasattr(args, 'meta_lr') else args.lr * 5,
#                         'weight_decay': meta_weight_decay[0]
#                     })
#                 else:
#                     # 不同参数不同衰减率的情况
#                     for p, wd in zip(meta_params, meta_weight_decay):
#                         optimizer_grouped_parameters.append({
#                             'params': [p],
#                             'lr': args.meta_lr if hasattr(args, 'meta_lr') else args.lr * 5,
#                             'weight_decay': wd
#                         })
                        
#         # 新增：实例化优化器（之前缺失的部分）
#         optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, correct_bias=False)
    
#     # 新增：实例化学习率调度器（之前缺失的部分）
#         num_train_optimization_steps = int(args.num_train_examples / args.train_batch_size) * args.num_train_epochs
#         num_warmup_steps = int(args.num_train_examples * args.num_train_epochs * args.warmup_proportion / args.train_batch_size)
#         scheduler = get_linear_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=num_warmup_steps,
#         num_training_steps=num_train_optimization_steps
#     )
#         return optimizer, scheduler
    
#     def _train(self, args): 
        
#         early_stopping = EarlyStopping(args)
             
        
#         for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
#             self.model.train()
#             loss_record = AverageMeter()
#             cons_loss_record = AverageMeter()
#             cls_loss_record = AverageMeter()
#             meta_loss_record = AverageMeter()  # 新增：元学习损失记录
            
# #             for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):

# #                 text_feats = batch['text_feats'].to(self.device)   # [16,3,37]
# #                 cons_text_feats = batch['cons_text_feats'].to(self.device)  # [16,3,37]
# #                 condition_idx = batch['condition_idx'].to(self.device)  # [16]
# #                 video_feats = batch['video_feats'].to(self.device)  # [16,230,1024] 
# #                 audio_feats = batch['audio_feats'].to(self.device)  # [16,480,768]
# #                 label_ids = batch['label_ids'].to(self.device)  # [16]
                
# #                 with torch.set_grad_enabled(True):

# #                     logits, _, condition, cons_condition = self.model(text_feats, video_feats, audio_feats, cons_text_feats, condition_idx)  # [16,20] [16,768] [16,768] [16,768]
                                      
# #                     cons_feature = torch.cat((condition.unsqueeze(1), cons_condition.unsqueeze(1)), dim=1)  # [16,2,768]<- [16,1,768] [16,1,768]
# #                     cons_loss = self.cons_criterion(cons_feature)
# #                     cls_loss = self.criterion(logits, label_ids)
# #                     loss = cls_loss + cons_loss
# #                     self.optimizer.zero_grad()
            
# #                     loss.backward()
# #                     loss_record.update(loss.item(), label_ids.size(0))
# #                     cons_loss_record.update(cons_loss.item(), label_ids.size(0))
# #                     cls_loss_record.update(cls_loss.item(), label_ids.size(0))

# #                     if args.grad_clip != -1.0:
# #                         nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], args.grad_clip)

# #                     self.optimizer.step()
# #                     self.scheduler.step()
            
# #             outputs = self._get_outputs(args, self.eval_dataloader)
# #             eval_score = outputs[args.eval_monitor]

# #             eval_results = {
# #                 'train_loss': round(loss_record.avg, 4),
# #                 'cons_loss': round(cons_loss_record.avg, 4),
# #                 'cls_loss': round(cls_loss_record.avg, 4),
# #                 'eval_score': round(eval_score, 4),
# #                 'best_eval_score': round(early_stopping.best_score, 4),
# #             }
#             for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):

#                 # 新增：获取模态质量特征
#                 quality_features = batch['quality_features'].to(self.device)
                
#                 # 原有特征加载
#                 text_feats = batch['text_feats'].to(self.device)
#                 cons_text_feats = batch['cons_text_feats'].to(self.device)
#                 condition_idx = batch['condition_idx'].to(self.device)
#                 video_feats = batch['video_feats'].to(self.device)
#                 audio_feats = batch['audio_feats'].to(self.device)
#                 label_ids = batch['label_ids'].to(self.device)
                

#                 with torch.set_grad_enabled(True):
#                     # 新增：传递质量特征和训练模式标识
#                     logits, _, condition, cons_condition = self.model(
#                         text_feats, 
#                         video_feats, 
#                         audio_feats, 
#                         cons_text_feats, 
#                         condition_idx,
#                         quality_features=quality_features,  # 传递质量特征
#                         is_train=True  # 标识训练模式
#                     )
                    
#                     # 原有损失计算
#                     cons_feature = torch.cat((condition.unsqueeze(1), cons_condition.unsqueeze(1)), dim=1)
#                     cons_loss = self.cons_criterion(cons_feature)
#                     cls_loss = self.criterion(logits, label_ids)
                    
#                     # 修正：从模型正确获取元学习损失（通过模型内部存储）
#                     meta_loss = self.model.model.meta_loss if hasattr(self.model.model, 'meta_loss') else 0.0
                    
#                     # 总损失 = 分类损失 + 对比损失 + 元学习损失
#                     loss = cls_loss + cons_loss + self.meta_loss_weight * meta_loss
                    
#                     self.optimizer.zero_grad()
#                     loss.backward()
                    
#                     # 更新损失记录
#                     loss_record.update(loss.item(), label_ids.size(0))
#                     cons_loss_record.update(cons_loss.item(), label_ids.size(0))
#                     cls_loss_record.update(cls_loss.item(), label_ids.size(0))
#                     meta_loss_record.update(meta_loss.item() if isinstance(meta_loss, torch.Tensor) else 0, label_ids.size(0))

#                     if args.grad_clip != -1.0:
#                         nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], args.grad_clip)

#                     self.optimizer.step()
#                     self.scheduler.step()
            
#             # 评估过程
#             outputs = self._get_outputs(args, self.eval_dataloader)
#             eval_score = outputs[args.eval_monitor]

#             # 新增元学习损失记录到日志
#             eval_results = {
#                 'train_loss': round(loss_record.avg, 4),
#                 'cons_loss': round(cons_loss_record.avg, 4),
#                 'cls_loss': round(cls_loss_record.avg, 4),
#                 'meta_loss': round(meta_loss_record.avg, 4),  # 元损失日志
#                 'eval_score': round(eval_score, 4),
#                 'best_eval_score': round(early_stopping.best_score, 4),
#             }


#             self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
#             for key in eval_results.keys():
#                 self.logger.info("  %s = %s", key, str(eval_results[key]))
            
#             early_stopping(eval_score, self.model)

#             if early_stopping.early_stop:
#                 self.logger.info(f'EarlyStopping at epoch {epoch + 1}')
#                 break

#         self.best_eval_score = early_stopping.best_score
#         self.model = early_stopping.best_model  
        
#         if args.save_model:
#             self.logger.info('Trained models are saved in %s', args.model_output_path)
#             save_model(self.model, args.model_output_path)   

#     def _get_outputs(self, args, dataloader, show_results = False):

#         self.model.eval()

#         total_labels = torch.empty(0,dtype=torch.long).to(self.device)
#         total_preds = torch.empty(0,dtype=torch.long).to(self.device)
#         total_logits = torch.empty((0, args.num_labels)).to(self.device)
#         total_features = torch.empty((0, args.feat_size)).to(self.device)
        
# #         for batch in tqdm(dataloader, desc="Iteration"):

# #             text_feats = batch['text_feats'].to(self.device)
# #             cons_text_feats = batch['cons_text_feats'].to(self.device)
# #             condition_idx = batch['condition_idx'].to(self.device)
# #             video_feats = batch['video_feats'].to(self.device)
# #             audio_feats = batch['audio_feats'].to(self.device)
# #             label_ids = batch['label_ids'].to(self.device)
                
# #             with torch.set_grad_enabled(False):
                
# #                 logits, features, condition, cons_condition = self.model(text_feats, video_feats, audio_feats, cons_text_feats, condition_idx)
#         for batch in tqdm(dataloader, desc="Iteration"):
#             # 新增：获取模态质量特征（评估/测试时也需要）
#             quality_features = batch['quality_features'].to(self.device)
            
#             # 原有特征加载
#             text_feats = batch['text_feats'].to(self.device)
#             cons_text_feats = batch['cons_text_feats'].to(self.device)
#             condition_idx = batch['condition_idx'].to(self.device)
#             video_feats = batch['video_feats'].to(self.device)
#             audio_feats = batch['audio_feats'].to(self.device)
#             label_ids = batch['label_ids'].to(self.device)
                
#             with torch.set_grad_enabled(False):
#                 # 新增：传递质量特征和推理模式标识
#                 logits, features, condition, cons_condition = self.model(
#                     text_feats, 
#                     video_feats, 
#                     audio_feats, 
#                     cons_text_feats, 
#                     condition_idx,
#                     quality_features=quality_features,  # 传递质量特征
#                     is_train=False  # 标识推理模式
#                 )
#                 total_logits = torch.cat((total_logits, logits))
#                 total_labels = torch.cat((total_labels, label_ids))
#                 total_features = torch.cat((total_features, features))

#         total_probs = F.softmax(total_logits.detach(), dim=1)
#         total_maxprobs, total_preds = total_probs.max(dim = 1)

#         y_logit = total_logits.cpu().numpy()
#         y_pred = total_preds.cpu().numpy()
#         y_true = total_labels.cpu().numpy()
#         y_prob = total_maxprobs.cpu().numpy()
#         y_feat = total_features.cpu().numpy()
        
#         outputs = self.metrics(y_true, y_pred, show_results = show_results)
        
#         # if args.save_pred and show_results:
#         #     np.save('y_true_' + str(args.seed) + '.npy', y_true)
#         #     np.save('y_pred_' + str(args.seed) + '.npy', y_pred)

#         outputs.update(
#             {
#                 'y_prob': y_prob,
#                 'y_logit': y_logit,
#                 'y_true': y_true,
#                 'y_pred': y_pred,
#                 'y_feat': y_feat
#             }
#         )

#         return outputs
    

#     def _test(self, args):
        
#         test_results = {}
#         # 1. 收集所有样本的质量分数
#         print("正在收集各模态质量分数...")
#         quality_data = collect_all_quality_scores(self.test_dataloader)  # 调用新增的收集函数
    
#          # 2. 生成并保存可视化图像
#         print("正在生成质量分布可视化...")
#         plot_quality_distribution(quality_data)  # 调用新增的可视化函数
        
#         ind_outputs = self._get_outputs(args, self.test_dataloader, show_results = True)
#         if args.train:
#             ind_outputs['best_eval_score'] = round(self.best_eval_score, 4)
        
#         test_results.update(ind_outputs)
        
#         return test_results
import os
import torch
import torch.nn.functional as F
import logging
from torch import nn
from utils.functions import restore_model, save_model, EarlyStopping
from tqdm import trange, tqdm
from data.utils import collect_all_quality_scores, get_dataloader, plot_quality_distribution
from utils.metrics import AverageMeter, Metrics
from transformers import AdamW, get_linear_schedule_with_warmup
from .model import MCWP_Wrapper  # 修改类名引用
from .loss import SupConLoss
import numpy as np


__all__ = ['MCWP_Manager']

class MCWP_Manager:

    def __init__(self, args, data, model):
         
        self.logger = logging.getLogger(args.logger_name)
        self.device = torch.device(f'cuda:{args.gpu_id}')
        args.device = self.device
        self.model = MCWP_Wrapper(args)  # 修改类名引用
        self.model.to(self.device)

        self.optimizer, self.scheduler = self._set_optimizer(args, self.model)
        mm_dataloader = get_dataloader(args, data.mm_data)
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = \
            mm_dataloader['train'], mm_dataloader['dev'], mm_dataloader['test']
                    
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.cons_criterion = SupConLoss(temperature=args.temperature)
        # 新增：模态一致性损失（KL散度）
        self.modal_consistency_criterion = nn.KLDivLoss(reduction='batchmean')
        self.metrics = Metrics(args)
        
        # 损失权重参数
        self.meta_loss_weight = args.meta_loss_weight if hasattr(args, 'meta_loss_weight') else 0.1
        # 新增：模态一致性损失权重（默认0.2，可从配置读取）
        self.modal_consistency_weight = args.modal_consistency_weight if hasattr(args, 'modal_consistency_weight') else 0.2
        
        if args.train:
            self.best_eval_score = 0
        else:
            self.model = restore_model(self.model, args.model_output_path, self.device)
            
    def _set_optimizer(self, args, model):
        # 收集所有参数ID用于去重
        all_param_ids = set()
        optimizer_grouped_parameters = []
        
        # 基础参数分组（不含元学习参数）
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        
        # 第一组：带权重衰减的普通参数
        normal_params = [
            p for n, p in param_optimizer 
            if not any(nd in n for nd in no_decay) 
            and 'meta_predictor' not in n  # 排除元学习参数
        ]
        if normal_params:
            optimizer_grouped_parameters.append({
                'params': normal_params,
                'weight_decay': args.weight_decay
            })
            all_param_ids.update(id(p) for p in normal_params)
        
        # 第二组：不带权重衰减的普通参数
        no_decay_params = [
            p for n, p in param_optimizer 
            if any(nd in n for nd in no_decay) 
            and 'meta_predictor' not in n  # 排除元学习参数
        ]
        if no_decay_params:
            optimizer_grouped_parameters.append({
                'params': no_decay_params,
                'weight_decay': 0.0
            })
            all_param_ids.update(id(p) for p in no_decay_params)
        
        # 第三组：元学习预测器参数（单独设置学习率）
        if hasattr(model.model.bert.MAG, 'meta_predictor'):
            # 收集元学习参数并计算正确的weight_decay
            meta_params = []
            meta_weight_decay = []
            for n, p in model.model.bert.MAG.meta_predictor.named_parameters():
                if id(p) not in all_param_ids:  # 确保不重复
                    meta_params.append(p)
                    # 根据参数名判断是否需要weight_decay
                    meta_weight_decay.append(args.weight_decay if 'bias' not in n else 0.0)
                    all_param_ids.add(id(p))
            
            if meta_params:
                # 处理权重衰减（如果所有参数衰减相同则简化，否则使用参数组列表）
                if len(set(meta_weight_decay)) == 1:
                    optimizer_grouped_parameters.append({
                        'params': meta_params,
                        'lr': args.meta_lr if hasattr(args, 'meta_lr') else args.lr * 5,
                        'weight_decay': meta_weight_decay[0]
                    })
                else:
                    # 不同参数不同衰减率的情况
                    for p, wd in zip(meta_params, meta_weight_decay):
                        optimizer_grouped_parameters.append({
                            'params': [p],
                            'lr': args.meta_lr if hasattr(args, 'meta_lr') else args.lr * 5,
                            'weight_decay': wd
                        })
                        
        # 实例化优化器
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, correct_bias=False)
    
        # 实例化学习率调度器
        num_train_optimization_steps = int(args.num_train_examples / args.train_batch_size) * args.num_train_epochs
        num_warmup_steps = int(args.num_train_examples * args.num_train_epochs * args.warmup_proportion / args.train_batch_size)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_optimization_steps
        )
        return optimizer, scheduler
    
    def _train(self, args): 
        
        early_stopping = EarlyStopping(args)
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            loss_record = AverageMeter()
            cons_loss_record = AverageMeter()
            cls_loss_record = AverageMeter()
            meta_loss_record = AverageMeter()
            # 新增：模态一致性损失记录
            modal_consistency_loss_record = AverageMeter()
            
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):
                # 获取模态质量特征
                quality_features = batch['quality_features'].to(self.device)
                
                # 原有特征加载
                text_feats = batch['text_feats'].to(self.device)
                cons_text_feats = batch['cons_text_feats'].to(self.device)
                condition_idx = batch['condition_idx'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)
                

                with torch.set_grad_enabled(True):
                    # 模型输出新增单模态logits
                    outputs = self.model(
                        text_feats, 
                        video_feats, 
                        audio_feats, 
                        cons_text_feats, 
                        condition_idx,
                        quality_features=quality_features,
                        is_train=True,
                        return_single_modal_logits=True  # 新增：请求返回单模态预测
                    )
                    
                    # # 解析输出（包含新增的单模态logits）
                    logits = outputs[0]
                    # pooled_output = outputs[1]    
                    condition = outputs[2]
                    cons_condition = outputs[3]
                    # # 新增：单模态logits
                    meta_loss = outputs[4]
                    text_logits = outputs[4]
                    video_logits = outputs[5]
                    audio_logits = outputs[6]
                    
                    # 原有损失计算
                    cons_feature = torch.cat((condition.unsqueeze(1), cons_condition.unsqueeze(1)), dim=1)
                    cons_loss = self.cons_criterion(cons_feature)
                    cls_loss = self.criterion(logits, label_ids)
                    # 新增：模态一致性损失计算
                    # 1. 转换为概率分布（logits -> log_softmax和softmax）
                    text_dist = F.log_softmax(text_logits, dim=-1)  # KL散度的输入需要log概率
                    video_dist = F.softmax(video_logits, dim=-1)
                    audio_dist = F.softmax(audio_logits, dim=-1)
                    
                    # 2. 计算视频和音频分布与文本分布的KL散度
                    video_text_consistency = self.modal_consistency_criterion(text_dist, video_dist)
                    audio_text_consistency = self.modal_consistency_criterion(text_dist, audio_dist)
                    modal_consistency_loss = (video_text_consistency + audio_text_consistency) * self.modal_consistency_weight
                    
                    # 修正：元学习损失
                    meta_loss = self.model.model.meta_loss if hasattr(self.model.model, 'meta_loss') else 0.5
                    
                    # 总损失 = 原有损失 + 模态一致性损失
                    loss = cls_loss + cons_loss + self.meta_loss_weight * meta_loss + modal_consistency_loss
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    # 更新损失记录
                    loss_record.update(loss.item(), label_ids.size(0))
                    cons_loss_record.update(cons_loss.item(), label_ids.size(0))
                    cls_loss_record.update(cls_loss.item(), label_ids.size(0))
                    meta_loss_record.update(meta_loss.item() if isinstance(meta_loss, torch.Tensor) else 0, label_ids.size(0))
                    # 新增：模态一致性损失记录
                    modal_consistency_loss_record.update(modal_consistency_loss.item(), label_ids.size(0))

                    if args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], args.grad_clip)

                    self.optimizer.step()
                    self.scheduler.step()
            
            # 评估过程
            outputs = self._get_outputs(args, self.eval_dataloader)
            eval_score = outputs[args.eval_monitor]

            # 日志新增模态一致性损失
            eval_results = {
                'train_loss': round(loss_record.avg, 4),
                'cons_loss': round(cons_loss_record.avg, 4),
                'cls_loss': round(cls_loss_record.avg, 4),
                'meta_loss': round(meta_loss_record.avg, 4),
                'modal_consistency_loss': round(modal_consistency_loss_record.avg, 4),  # 新增
                'eval_score': round(eval_score, 4),
                'best_eval_score': round(early_stopping.best_score, 4),
            }


            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in eval_results.keys():
                self.logger.info("  %s = %s", key, str(eval_results[key]))
            
            early_stopping(eval_score, self.model)

            if early_stopping.early_stop:
                self.logger.info(f'EarlyStopping at epoch {epoch + 1}')
                break

        self.best_eval_score = early_stopping.best_score
        self.model = early_stopping.best_model  
        
        if args.save_model:
            self.logger.info('Trained models are saved in %s', args.model_output_path)
            save_model(self.model, args.model_output_path)   

    def _get_outputs(self, args, dataloader, show_results = False):

        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)
        total_features = torch.empty((0, args.feat_size)).to(self.device)
        
        for batch in tqdm(dataloader, desc="Iteration"):
            # 获取模态质量特征
            quality_features = batch['quality_features'].to(self.device)
            
            # 原有特征加载
            text_feats = batch['text_feats'].to(self.device)
            cons_text_feats = batch['cons_text_feats'].to(self.device)
            condition_idx = batch['condition_idx'].to(self.device)
            video_feats = batch['video_feats'].to(self.device)
            audio_feats = batch['audio_feats'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)
                
            with torch.set_grad_enabled(False):
                # 推理时也获取多模态输出
                outputs = self.model(
                    text_feats, 
                    video_feats, 
                    audio_feats, 
                    cons_text_feats, 
                    condition_idx,
                    quality_features=quality_features,
                    is_train=False,
                    return_single_modal_logits=False  # 评估时可选是否返回
                )
                logits = outputs[0]
                features = outputs[1]
                
                total_logits = torch.cat((total_logits, logits))
                total_labels = torch.cat((total_labels, label_ids))
                total_features = torch.cat((total_features, features))

        total_probs = F.softmax(total_logits.detach(), dim=1)
        total_maxprobs, total_preds = total_probs.max(dim = 1)

        y_logit = total_logits.cpu().numpy()
        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        y_prob = total_maxprobs.cpu().numpy()
        y_feat = total_features.cpu().numpy()
        
        outputs = self.metrics(y_true, y_pred, show_results = show_results)
        
        outputs.update(
            {
                'y_prob': y_prob,
                'y_logit': y_logit,
                'y_true': y_true,
                'y_pred': y_pred,
                'y_feat': y_feat
            }
        )

        return outputs
    

    def _test(self, args):
        
        test_results = {}
        # 1. 收集所有样本的质量分数
        print("正在收集各模态质量分数...")
        quality_data = collect_all_quality_scores(self.test_dataloader)
        
        # 2. 生成并保存可视化图像
        print("正在生成质量分布可视化...")
        plot_quality_distribution(quality_data)
        
        ind_outputs = self._get_outputs(args, self.test_dataloader, show_results = True)
        if args.train:
            ind_outputs['best_eval_score'] = round(self.best_eval_score, 4)
        
        test_results.update(ind_outputs)
        
        return test_results
