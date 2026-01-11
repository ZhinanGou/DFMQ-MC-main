# import torch.nn.functional as F
# import torch
# import torch.utils.checkpoint
# from torch import nn
# from torch.nn import CrossEntropyLoss, MSELoss
# from transformers import BertPreTrainedModel
# from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler
# from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
# from torchmeta.modules import MetaModule, MetaLinear  # 元学习模块
# from .SubNets.transformers_encoder.transformer import TransformerEncoder
# from .AlignNets import AlignSubNet


# class MetaWeightPredictor(MetaModule):
#     """元学习权重预测器：根据模态质量特征动态生成融合权重"""
#     def __init__(self, quality_dim=3, hidden_dim=64):
#         super().__init__()
#         # 元学习网络（显式指定float32类型）
#         self.layers = nn.Sequential(
#             MetaLinear(quality_dim, hidden_dim, dtype=torch.float32),
#             nn.ReLU(),
#             MetaLinear(hidden_dim, hidden_dim, dtype=torch.float32),
#             nn.ReLU(),
#             MetaLinear(hidden_dim, 2, dtype=torch.float32)  # 输出视频和音频的动态权重
#         )
#         self.softmax = nn.Softmax(dim=-1)  # 权重归一化

#     def forward(self, quality_features, params=None):
#         """
#         Args:
#             quality_features: 模态质量特征 [batch_size, 3]（文本、音频、视频质量）
#             params: 元学习快速权重（内循环更新后）
#         Returns:
#             动态权重 [batch_size, 2]（视频权重、音频权重）
#         """
#         # 确保输入特征为float32
#         quality_features = quality_features.to(torch.float32)
        
#         if params is None:
#             x = self.layers(quality_features)
#         else:
#             # 确保快速权重为float32
#             params = [p.to(torch.float32) for p in params]
#             x = F.linear(quality_features, params[0], params[1])
#             x = F.relu(x)
#             x = F.linear(x, params[2], params[3])
#             x = F.relu(x)
#             x = F.linear(x, params[4], params[5])
#         return self.softmax(x)


# class MAG(nn.Module):
#     def __init__(self, config, args):
#         super(MAG, self).__init__()
#         self.args = args

#         if self.args.need_aligned:
#             self.alignNet = AlignSubNet(args, args.mag_aligned_method)

#         text_feat_dim, audio_feat_dim, video_feat_dim = args.text_feat_dim, args.audio_feat_dim, args.video_feat_dim
        
#         # 基础投影层（显式初始化float32参数）
#         self.W_v = nn.Linear(video_feat_dim, text_feat_dim, dtype=torch.float32)
#         self.W_a = nn.Linear(audio_feat_dim, text_feat_dim, dtype=torch.float32)

#         # 元学习权重预测器
#         self.meta_predictor = MetaWeightPredictor(
#             hidden_dim=args.meta_hidden_dim
#         )

#         self.beta_shift = args.beta_shift
#         self.LayerNorm = nn.LayerNorm(config.hidden_size, dtype=torch.float32)
#         self.dropout = nn.Dropout(args.dropout_prob)

#     def forward(self, text_embedding, visual, acoustic, quality_features, is_train=False):
#         """
#         新增参数：
#             quality_features: 模态质量特征 [batch_size, 3]
#             is_train: 是否训练模式（训练时启用元学习内循环）
#         """
#         eps = 1e-6

#         # 统一输入张量类型为float32
#         text_embedding = text_embedding.to(torch.float32)
#         visual = visual.to(torch.float32)
#         acoustic = acoustic.to(torch.float32)
#         quality_features = quality_features.to(torch.float32)

#         if self.args.need_aligned:
#             text_embedding, visual, acoustic = self.alignNet(text_embedding, visual, acoustic)
        
#         # 1. 元学习动态权重预测
#         meta_loss = None
#         if is_train:
#             # 训练时：划分支持集和查询集进行元学习内循环
#             batch_size = quality_features.size(0)
#             support_size = batch_size // 2
#             support_quality = quality_features[:support_size]  # 支持集质量特征
#             query_quality = quality_features[support_size:]    # 查询集质量特征

#             # 内循环：支持集快速适应
#             fast_weights = list(self.meta_predictor.parameters())  # 复制当前参数作为快速权重
#             support_weights = self.meta_predictor(support_quality, fast_weights)  # 支持集权重
#             # 内循环损失（权重分布伪标签损失）
#             support_loss = F.mse_loss(support_weights, torch.ones_like(support_weights)/2)
#             # 计算梯度并更新快速权重
#             grads = torch.autograd.grad(support_loss, fast_weights, create_graph=True)
#             fast_weights = [w - 0.01 * g for w, g in zip(fast_weights, grads)]

#             # 计算查询集元损失
#             query_weights = self.meta_predictor(query_quality, fast_weights)
#             meta_loss = F.mse_loss(query_weights, torch.ones_like(query_weights)/2)
#             # 合并支持集和查询集权重
#             weights = torch.cat([support_weights, query_weights], dim=0)
#         else:
#             # 推理时：直接用基础权重预测
#             weights = self.meta_predictor(quality_features)  # [batch_size, 2]

#         # 2. 应用动态权重融合模态特征
#         video_weight = weights[:, 0].unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1]
#         audio_weight = weights[:, 1].unsqueeze(1).unsqueeze(2)   # [batch_size, 1, 1]

#         # 加权融合视频和音频特征
#         h_m = video_weight * self.W_v(visual) + audio_weight * self.W_a(acoustic)

#         # 3. 残差连接和归一化
#         em_norm = text_embedding.norm(2, dim=-1)
#         hm_norm = h_m.norm(2, dim=-1)
#         hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True, device=text_embedding.device, dtype=torch.float32)
#         hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

#         thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift
#         ones = torch.ones(thresh_hold.shape, requires_grad=True, device=text_embedding.device, dtype=torch.float32)
#         alpha = torch.min(thresh_hold, ones).unsqueeze(dim=-1)

#         acoustic_vis_embedding = alpha * h_m
#         embedding_output = self.dropout(self.LayerNorm(acoustic_vis_embedding + text_embedding))

#         return embedding_output, meta_loss  # 返回融合结果和元损失（训练时）


# class MCWP(BertPreTrainedModel):
#     def __init__(self, config, args):
#         super().__init__(config)
#         self.config = config

#         self.embeddings = BertEmbeddings(config)
#         self.encoder = BertEncoder(config)
#         self.pooler = BertPooler(config)
#         # 融合层：使用修改后的MAG（支持动态权重）
#         self.MAG = MAG(config, args)
#         self.args = args

#         # MAP模块原有逻辑
#         self.alignNet = AlignSubNet(args, args.aligned_method)
#         self.embed_dim = args.text_feat_dim
#         self.num_heads = args.nheads
#         self.layers = args.n_levels
#         self.attn_dropout = args.attn_dropout
#         self.relu_dropout = args.relu_dropout
#         self.res_dropout = args.res_dropout
#         self.embed_dropout = args.embed_dropout
#         self.attn_mask = args.attn_mask

#         # 投影层显式指定float32
#         self.audio_proj = nn.Sequential(
#             nn.LayerNorm(args.audio_feat_dim, dtype=torch.float32),
#             nn.Linear(args.audio_feat_dim, self.embed_dim, dtype=torch.float32),
#             nn.LayerNorm(self.embed_dim, dtype=torch.float32),
#         )

#         self.video_proj = nn.Sequential(
#             nn.LayerNorm(args.video_feat_dim, dtype=torch.float32),
#             nn.Linear(args.video_feat_dim, self.embed_dim, dtype=torch.float32),
#             nn.LayerNorm(self.embed_dim, dtype=torch.float32),
#         )

#         self.text_proj = nn.Sequential(
#             nn.LayerNorm(args.text_feat_dim, dtype=torch.float32),
#             nn.Linear(args.text_feat_dim, self.embed_dim, dtype=torch.float32),
#         )

#         self.out_proj = nn.Sequential(
#             nn.LayerNorm(self.embed_dim, dtype=torch.float32),
#             nn.Linear(self.embed_dim, args.text_feat_dim, dtype=torch.float32)
#         )
#         self.trans_a_with_l = TransformerEncoder(embed_dim=self.embed_dim,
#                                 num_heads=self.num_heads,
#                                 layers=self.layers,
#                                 attn_dropout=self.attn_dropout,
#                                 relu_dropout=self.relu_dropout,
#                                 res_dropout=self.res_dropout,
#                                 embed_dropout=self.embed_dropout,
#                                 attn_mask=self.attn_mask)
        
#         self.gamma = nn.Parameter(torch.ones(args.text_feat_dim, dtype=torch.float32) * 1e-4)

#         self.init_weights()

#     def get_input_embeddings(self):
#         return self.embeddings.word_embeddings

#     def set_input_embeddings(self, value):
#         self.embeddings.word_embeddings = value

#     def _prune_heads(self, heads_to_prune):
#         for layer, heads in heads_to_prune.items():
#             self.encoder.layer[layer].attention.prune_heads(heads)

#     def forward(
#         self,
#         input_ids,
#         visual,
#         acoustic,
#         condition_idx,
#         ctx,
#         quality_features,  # 模态质量特征
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         encoder_hidden_states=None,
#         encoder_attention_mask=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         is_train=False,  # 训练模式标识
        
#     ):
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

#         if input_ids is not None and inputs_embeds is not None:
#             raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
#         elif input_ids is not None:
#             input_shape = input_ids.size()
#         elif inputs_embeds is not None:
#             input_shape = inputs_embeds.size()[:-1]
#         else:
#             raise ValueError("You have to specify either input_ids or inputs_embeds")

#         device = input_ids.device if input_ids is not None else inputs_embeds.device
#         if attention_mask is None:
#             attention_mask = torch.ones(input_shape, device=device, dtype=torch.float32)
#         if token_type_ids is None:
#             token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

#         # 扩展注意力掩码
#         extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
#         encoder_extended_attention_mask = None
#         if self.config.is_decoder and encoder_hidden_states is not None:
#             encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
#             encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
#             if encoder_attention_mask is None:
#                 encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device, dtype=torch.float32)
#             encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)

#         head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

#         # 获取文本嵌入（确保float32）
#         embedding_output = self.embeddings(
#             input_ids=input_ids,
#             position_ids=position_ids,
#             token_type_ids=token_type_ids,
#             inputs_embeds=inputs_embeds,
#         ).to(torch.float32)

#         # 生成模态感知提示
#         batch_ctx = ctx.unsqueeze(0).repeat(acoustic.shape[0], 1, 1).to(torch.float32)
#         _, aligned_visual, aligned_acoustic = self.alignNet(batch_ctx, visual, acoustic)
#         aligned_acoustic = self.audio_proj(aligned_acoustic.to(torch.float32))
#         aligned_visual = self.video_proj(aligned_visual.to(torch.float32))
#         batch_ctx = self.text_proj(batch_ctx)
#         generated_ctx = self.trans_a_with_l(batch_ctx.permute(1, 0, 2), aligned_visual.permute(1, 0, 2), aligned_acoustic.permute(1, 0, 2)).permute(1, 0, 2)
#         generated_ctx = batch_ctx + self.out_proj(generated_ctx) * self.gamma
#         for i in range(embedding_output.shape[0]):
#             embedding_output[i, condition_idx[i] - self.args.prompt_len : condition_idx[i], :] = generated_ctx[i]

#         # 融合模态特征（传入质量特征和训练模式）
#         fused_embedding, meta_loss = self.MAG(
#             embedding_output, 
#             visual, 
#             acoustic, 
#             quality_features=quality_features,
#             is_train=is_train
#         )

#         # BERT编码器处理融合后的特征
#         encoder_outputs = self.encoder(
#             fused_embedding,
#             attention_mask=extended_attention_mask,
#             head_mask=head_mask,
#             encoder_hidden_states=encoder_hidden_states,
#             encoder_attention_mask=encoder_extended_attention_mask,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#         )
#         sequence_output = encoder_outputs[0].to(torch.float32)
#         pooled_output = self.pooler(sequence_output).to(torch.float32)

#         outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
#         return outputs, generated_ctx, meta_loss  # 返回元损失


# class MCWP_Model(BertPreTrainedModel):
#     def __init__(self, config, args):
#         super().__init__(config)
#         self.num_labels = args.num_labels
#         self.label_len = args.label_len

#         self.bert = MCWP(config, args)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, args.num_labels, dtype=torch.float32)  # 显式float32

#         self.init_weights()

#     def forward(
#         self,
#         text,
#         visual,
#         acoustic,
#         condition_idx,
#         ctx,
#         quality_features,  # 模态质量特征
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         is_train=False,  # 训练模式标识
#     ):
#         input_ids, attention_mask, token_type_ids = text[:, 0], text[:, 1], text[:, 2]

#         # 调用MAP时传入质量特征和训练模式
#         outputs, generated_ctx, meta_loss = self.bert(
#             input_ids,
#             visual,
#             acoustic,
#             condition_idx,
#             ctx,
#             quality_features=quality_features,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             is_train=is_train,
#         )

#         sequence_output = outputs[0].to(torch.float32)
#         condition_tuple = tuple(sequence_output[torch.arange(sequence_output.shape[0]), condition_idx.view(-1) + i, :].unsqueeze(1) for i in range(self.label_len))
#         condition = torch.cat(condition_tuple, dim=1)
        
#         pooled_output = outputs[1].to(torch.float32)
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)

#         outputs = (logits,) + outputs[2:]

#         if labels is not None:
#             if self.num_labels == 1:
#                 loss_fct = MSELoss()
#                 loss = loss_fct(logits.view(-1), labels.view(-1).to(torch.float32))
#             else:
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
#             # 训练时：总损失 = 分类损失 + 元学习损失
#             if is_train and meta_loss is not None:
#                 loss = loss + 0.1 * meta_loss  # 元损失权重可通过配置调整
#             outputs = (loss,) + outputs
            
#         return outputs, pooled_output, condition, generated_ctx


# class Cons_Model(BertPreTrainedModel):
#     """对比学习模型（确保兼容性）"""
#     def __init__(self, config, args, add_pooling_layer=True):
#         super().__init__(config)
#         self.config = config

#         self.embeddings = BertEmbeddings(config)
#         self.encoder = BertEncoder(config)

#         self.pooler = BertPooler(config) if add_pooling_layer else None
#         self.args = args
#         self.post_init()

#     def get_input_embeddings(self):
#         return self.embeddings.word_embeddings

#     def set_input_embeddings(self, value):
#         self.embeddings.word_embeddings = value

#     def _prune_heads(self, heads_to_prune):
#         for layer, heads in heads_to_prune.items():
#             self.encoder.layer[layer].attention.prune_heads(heads)

#     def forward(
#         self,
#         condition_idx,
#         ctx,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         encoder_hidden_states=None,
#         encoder_attention_mask=None,
#         past_key_values=None,
#         use_cache=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if self.config.is_decoder:
#             use_cache = use_cache if use_cache is not None else self.config.use_cache
#         else:
#             use_cache = False

#         if input_ids is not None and inputs_embeds is not None:
#             raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
#         elif input_ids is not None:
#             input_shape = input_ids.size()
#         elif inputs_embeds is not None:
#             input_shape = inputs_embeds.size()[:-1]
#         else:
#             raise ValueError("You have to specify either input_ids or inputs_embeds")

#         batch_size, seq_length = input_shape
#         device = input_ids.device if input_ids is not None else inputs_embeds.device
#         past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

#         if attention_mask is None:
#             attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
#         if token_type_ids is None:
#             if hasattr(self.embeddings, "token_type_ids"):
#                 buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
#                 buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
#                 token_type_ids = buffered_token_type_ids_expanded
#             else:
#                 token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

#         extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
#         encoder_extended_attention_mask = None
#         if self.config.is_decoder and encoder_hidden_states is not None:
#             encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
#             encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
#             if encoder_attention_mask is None:
#                 encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
#             encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)

#         head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

#         embedding_output = self.embeddings(
#             input_ids=input_ids,
#             position_ids=position_ids,
#             token_type_ids=token_type_ids,
#             inputs_embeds=inputs_embeds,
#             past_key_values_length=past_key_values_length,
#         )

#         for i in range(embedding_output.shape[0]):
#             embedding_output[i, condition_idx[i] - self.args.prompt_len : condition_idx[i], :] = ctx[i]

#         encoder_outputs = self.encoder(
#             embedding_output,
#             attention_mask=extended_attention_mask,
#             head_mask=head_mask,
#             encoder_hidden_states=encoder_hidden_states,
#             encoder_attention_mask=encoder_extended_attention_mask,
#             past_key_values=past_key_values,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         sequence_output = encoder_outputs[0]
#         pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

#         if not return_dict:
#             return (sequence_output, pooled_output) + encoder_outputs[1:]

#         return BaseModelOutputWithPoolingAndCrossAttentions(
#             last_hidden_state=sequence_output,
#             pooler_output=pooled_output,
#             past_key_values=encoder_outputs.past_key_values,
#             hidden_states=encoder_outputs.hidden_states,
#             attentions=encoder_outputs.attentions,
#             cross_attentions=encoder_outputs.cross_attentions,
#         )


# # 包装类避免类名冲突
# class MCWP_Wrapper(nn.Module):
#     def __init__(self, args):
#         super(MCWP_Wrapper, self).__init__()
        
#         # 加载预训练模型（路径根据实际情况调整）
#         self.model = MCWP_Model.from_pretrained(
#             '/home/jiamengyao/MIntRec-main/bert-base-uncased', 
#             # cache_dir=args.cache_path, 
#             cache_dir="./cache",
#             args=args
#         ).float()  # 确保模型为float32
#         self.cons_model = Cons_Model.from_pretrained(
#             '/home/jiamengyao/MIntRec-main/bert-base-uncased', 
#             # cache_dir=args.cache_path, 
#             cache_dir="./cache",
#             args=args
#         ).float()  # 确保模型为float32

#         self.ctx_vectors = self._init_ctx(args)
#         self.ctx = nn.Parameter(self.ctx_vectors)

#         self.label_len = args.label_len
#         args.feat_size = args.text_feat_dim
#         args.video_feat_size = args.video_feat_dim
#         args.audio_feat_size = args.audio_feat_dim

#     def _init_ctx(self, args):
#         ctx = torch.empty(args.prompt_len, args.text_feat_dim, dtype=torch.float32)
#         nn.init.trunc_normal_(ctx)
#         return ctx

    
#     def forward(self, text_feats, video_feats, audio_feats, cons_text_feats, condition_idx, quality_features, is_train=False):
#         """新增参数：
#             quality_features: 模态质量特征 [batch_size, 3]
#             is_train: 是否训练模式
#         """
#         # 确保所有输入特征为float32
#         video_feats = video_feats.float()
#         audio_feats = audio_feats.float()
#         quality_features = quality_features.float()

#         # 处理正常样本（传入质量特征和训练模式）
#         outputs, pooled_output, condition, generated_ctx = self.model(
#             text=text_feats,
#             visual=video_feats,
#             acoustic=audio_feats,
#             condition_idx=condition_idx, 
#             ctx=self.ctx,
#             quality_features=quality_features,  # 传入质量特征
#             is_train=is_train  # 传入训练模式
#         )

#         # 处理增强样本
#         cons_input_ids, cons_input_mask, cons_segment_ids = cons_text_feats[:, 0], cons_text_feats[:, 1], cons_text_feats[:, 2]
#         cons_outputs = self.cons_model(
#             input_ids=cons_input_ids, 
#             condition_idx=condition_idx,
#             ctx=generated_ctx,
#             token_type_ids=cons_segment_ids, 
#             attention_mask=cons_input_mask
#         )
#         last_hidden_state = cons_outputs.last_hidden_state
#         cons_condition_tuple = tuple(last_hidden_state[torch.arange(last_hidden_state.shape[0]), condition_idx.view(-1) + i, :].unsqueeze(1) for i in range(self.label_len))
#         cons_condition = torch.cat(cons_condition_tuple, dim=1)

#         return outputs[0], pooled_output, condition.mean(dim=1), cons_condition.mean(dim=1)
import torch.nn.functional as F
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from torchmeta.modules import MetaModule, MetaLinear  # 元学习模块
from .SubNets.transformers_encoder.transformer import TransformerEncoder
from .AlignNets import AlignSubNet


class MetaWeightPredictor(MetaModule):
    """元学习权重预测器：根据模态质量特征动态生成融合权重"""
    def __init__(self, quality_dim=3, hidden_dim=64):
        super().__init__()
        # 元学习网络（显式指定float32类型）
        self.layers = nn.Sequential(
            MetaLinear(quality_dim, hidden_dim, dtype=torch.float32),
            nn.ReLU(),
            MetaLinear(hidden_dim, hidden_dim, dtype=torch.float32),
            nn.ReLU(),
            MetaLinear(hidden_dim, 2, dtype=torch.float32)  # 输出视频和音频的动态权重
        )
        self.softmax = nn.Softmax(dim=-1)  # 权重归一化

    def forward(self, quality_features, params=None):
        """
        Args:
            quality_features: 模态质量特征 [batch_size, 3]（文本、音频、视频质量）
            params: 元学习快速权重（内循环更新后）
        Returns:
            动态权重 [batch_size, 2]（视频权重、音频权重）
        """
        # 确保输入特征为float32
        quality_features = quality_features.to(torch.float32)
        
        if params is None:
            x = self.layers(quality_features)
        else:
            # 确保快速权重为float32
            params = [p.to(torch.float32) for p in params]
            x = F.linear(quality_features, params[0], params[1])
            x = F.relu(x)
            x = F.linear(x, params[2], params[3])
            x = F.relu(x)
            x = F.linear(x, params[4], params[5])
        return self.softmax(x)


class MAG(nn.Module):
    def __init__(self, config, args):
        super(MAG, self).__init__()
        self.args = args

        if self.args.need_aligned:
            self.alignNet = AlignSubNet(args, args.mag_aligned_method)

        text_feat_dim, audio_feat_dim, video_feat_dim = args.text_feat_dim, args.audio_feat_dim, args.video_feat_dim
        
        # 基础投影层（显式初始化float32参数）
        self.W_v = nn.Linear(video_feat_dim, text_feat_dim, dtype=torch.float32)
        self.W_a = nn.Linear(audio_feat_dim, text_feat_dim, dtype=torch.float32)

        # 单模态分类器（新增：用于生成单模态logits）
        self.text_classifier = nn.Linear(text_feat_dim, args.num_labels, dtype=torch.float32)
        self.video_classifier = nn.Linear(text_feat_dim, args.num_labels, dtype=torch.float32)
        self.audio_classifier = nn.Linear(text_feat_dim, args.num_labels, dtype=torch.float32)

        # 元学习权重预测器
        self.meta_predictor = MetaWeightPredictor(
            hidden_dim=args.meta_hidden_dim
        )

        self.beta_shift = args.beta_shift
        self.LayerNorm = nn.LayerNorm(config.hidden_size, dtype=torch.float32)
        self.dropout = nn.Dropout(args.dropout_prob)

    def forward(self, text_embedding, visual, acoustic, quality_features, is_train=False, return_single_modal_logits=False):
        """
        新增参数：
            quality_features: 模态质量特征 [batch_size, 3]
            is_train: 是否训练模式（训练时启用元学习内循环）
            return_single_modal_logits: 是否返回单模态预测logits
        """
        eps = 1e-6

        # 统一输入张量类型为float32
        text_embedding = text_embedding.to(torch.float32)
        visual = visual.to(torch.float32)
        acoustic = acoustic.to(torch.float32)
        quality_features = quality_features.to(torch.float32)

        if self.args.need_aligned:
            text_embedding, visual, acoustic = self.alignNet(text_embedding, visual, acoustic)
        
        # 1. 元学习动态权重预测
        meta_loss = None
        if is_train:
            # 训练时：划分支持集和查询集进行元学习内循环
            batch_size = quality_features.size(0)
            support_size = batch_size // 2
            support_quality = quality_features[:support_size]  # 支持集质量特征
            query_quality = quality_features[support_size:]    # 查询集质量特征

            # 内循环：支持集快速适应
            fast_weights = list(self.meta_predictor.parameters())  # 复制当前参数作为快速权重
            support_weights = self.meta_predictor(support_quality, fast_weights)  # 支持集权重
            # 内循环损失（权重分布伪标签损失）
            support_loss = F.mse_loss(support_weights, torch.ones_like(support_weights)/2)
            # 计算梯度并更新快速权重
            grads = torch.autograd.grad(support_loss, fast_weights, create_graph=True)
            fast_weights = [w - 0.01 * g for w, g in zip(fast_weights, grads)]

            # 计算查询集元损失
            query_weights = self.meta_predictor(query_quality, fast_weights)
            meta_loss = F.mse_loss(query_weights, torch.ones_like(query_weights)/2)
            # 合并支持集和查询集权重
            weights = torch.cat([support_weights, query_weights], dim=0)
        else:
            # 推理时：直接用基础权重预测
            weights = self.meta_predictor(quality_features)  # [batch_size, 2]

        # 2. 应用动态权重融合模态特征
        video_weight = weights[:, 0].unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1]
        audio_weight = weights[:, 1].unsqueeze(1).unsqueeze(2)   # [batch_size, 1, 1]

        # 加权融合视频和音频特征
        h_m = video_weight * self.W_v(visual) + audio_weight * self.W_a(acoustic)

        # 3. 残差连接和归一化
        em_norm = text_embedding.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)
        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True, device=text_embedding.device, dtype=torch.float32)
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift
        ones = torch.ones(thresh_hold.shape, requires_grad=True, device=text_embedding.device, dtype=torch.float32)
        alpha = torch.min(thresh_hold, ones).unsqueeze(dim=-1)

        acoustic_vis_embedding = alpha * h_m
        embedding_output = self.dropout(self.LayerNorm(acoustic_vis_embedding + text_embedding))

        # 计算单模态logits（新增逻辑）
        if return_single_modal_logits:
            # 文本单模态logits（使用池化后的文本特征）
            text_pooled = torch.mean(text_embedding, dim=1)  # 简单池化，可根据实际调整
            text_logits = self.text_classifier(text_pooled)
            
            # 视频单模态logits（使用投影后的视频特征）
            video_proj = self.W_v(visual)
            video_pooled = torch.mean(video_proj, dim=1)
            video_logits = self.video_classifier(video_pooled)
            
            # 音频单模态logits（使用投影后的音频特征）
            audio_proj = self.W_a(acoustic)
            audio_pooled = torch.mean(audio_proj, dim=1)
            audio_logits = self.audio_classifier(audio_pooled)
            
            return embedding_output, meta_loss, text_logits, video_logits, audio_logits
        else:
            return embedding_output, meta_loss  # 返回融合结果和元损失（训练时）


class MCWP(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        # 融合层：使用修改后的MAG（支持动态权重）
        self.MAG = MAG(config, args)
        self.args = args

        # MAP模块原有逻辑
        self.alignNet = AlignSubNet(args, args.aligned_method)
        self.embed_dim = args.text_feat_dim
        self.num_heads = args.nheads
        self.layers = args.n_levels
        self.attn_dropout = args.attn_dropout
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout
        self.embed_dropout = args.embed_dropout
        self.attn_mask = args.attn_mask

        # 投影层显式指定float32
        self.audio_proj = nn.Sequential(
            nn.LayerNorm(args.audio_feat_dim, dtype=torch.float32),
            nn.Linear(args.audio_feat_dim, self.embed_dim, dtype=torch.float32),
            nn.LayerNorm(self.embed_dim, dtype=torch.float32),
        )

        self.video_proj = nn.Sequential(
            nn.LayerNorm(args.video_feat_dim, dtype=torch.float32),
            nn.Linear(args.video_feat_dim, self.embed_dim, dtype=torch.float32),
            nn.LayerNorm(self.embed_dim, dtype=torch.float32),
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(args.text_feat_dim, dtype=torch.float32),
            nn.Linear(args.text_feat_dim, self.embed_dim, dtype=torch.float32),
        )

        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.embed_dim, dtype=torch.float32),
            nn.Linear(self.embed_dim, args.text_feat_dim, dtype=torch.float32)
        )
        self.trans_a_with_l = TransformerEncoder(embed_dim=self.embed_dim,
                                num_heads=self.num_heads,
                                layers=self.layers,
                                attn_dropout=self.attn_dropout,
                                relu_dropout=self.relu_dropout,
                                res_dropout=self.res_dropout,
                                embed_dropout=self.embed_dropout,
                                attn_mask=self.attn_mask)
        
        self.gamma = nn.Parameter(torch.ones(args.text_feat_dim, dtype=torch.float32) * 1e-4)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids,
        visual,
        acoustic,
        condition_idx,
        ctx,
        quality_features,  # 模态质量特征
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        is_train=False,  # 训练模式标识
        return_single_modal_logits=False,  # 新增参数：是否返回单模态logits
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device, dtype=torch.float32)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # 扩展注意力掩码
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        encoder_extended_attention_mask = None
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device, dtype=torch.float32)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # 获取文本嵌入（确保float32）
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        ).to(torch.float32)

        # 生成模态感知提示
        batch_ctx = ctx.unsqueeze(0).repeat(acoustic.shape[0], 1, 1).to(torch.float32)
        _, aligned_visual, aligned_acoustic = self.alignNet(batch_ctx, visual, acoustic)
        aligned_acoustic = self.audio_proj(aligned_acoustic.to(torch.float32))
        aligned_visual = self.video_proj(aligned_visual.to(torch.float32))
        batch_ctx = self.text_proj(batch_ctx)
        generated_ctx = self.trans_a_with_l(batch_ctx.permute(1, 0, 2), aligned_visual.permute(1, 0, 2), aligned_acoustic.permute(1, 0, 2)).permute(1, 0, 2)
        generated_ctx = batch_ctx + self.out_proj(generated_ctx) * self.gamma
        for i in range(embedding_output.shape[0]):
            embedding_output[i, condition_idx[i] - self.args.prompt_len : condition_idx[i], :] = generated_ctx[i]

        # 融合模态特征（传入新增参数）
        if return_single_modal_logits:
            fused_embedding, meta_loss, text_logits, video_logits, audio_logits = self.MAG(
                embedding_output, 
                visual, 
                acoustic, 
                quality_features=quality_features,
                is_train=is_train,
                return_single_modal_logits=True
            )
        else:
            fused_embedding, meta_loss = self.MAG(
                embedding_output, 
                visual, 
                acoustic, 
                quality_features=quality_features,
                is_train=is_train,
                return_single_modal_logits=False
            )
            text_logits = video_logits = audio_logits = None  # 未启用时置空

        # BERT编码器处理融合后的特征
        encoder_outputs = self.encoder(
            fused_embedding,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0].to(torch.float32)
        pooled_output = self.pooler(sequence_output).to(torch.float32)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        # 根据参数决定返回值
        if return_single_modal_logits:
            return outputs, generated_ctx, meta_loss, text_logits, video_logits, audio_logits
        else:
            return outputs, generated_ctx, meta_loss


class MCWP_Model(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.label_len = args.label_len

        self.bert = MCWP(config, args)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, args.num_labels, dtype=torch.float32)  # 显式float32

        self.init_weights()

    def forward(
        self,
        text,
        visual,
        acoustic,
        condition_idx,
        ctx,
        quality_features,  # 模态质量特征
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        is_train=False,  # 训练模式标识
        return_single_modal_logits=False,  # 新增参数
    ):
        input_ids, attention_mask, token_type_ids = text[:, 0], text[:, 1], text[:, 2]

        # 调用时传递新增参数
        if return_single_modal_logits:
            outputs, generated_ctx, meta_loss, text_logits, video_logits, audio_logits = self.bert(
                input_ids,
                visual,
                acoustic,
                condition_idx,
                ctx,
                quality_features=quality_features,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                is_train=is_train,
                return_single_modal_logits=True
            )
        else:
            outputs, generated_ctx, meta_loss = self.bert(
                input_ids,
                visual,
                acoustic,
                condition_idx,
                ctx,
                quality_features=quality_features,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                is_train=is_train,
                return_single_modal_logits=False
            )
            text_logits = video_logits = audio_logits = None

        sequence_output = outputs[0].to(torch.float32)
        condition_tuple = tuple(sequence_output[torch.arange(sequence_output.shape[0]), condition_idx.view(-1) + i, :].unsqueeze(1) for i in range(self.label_len))
        condition = torch.cat(condition_tuple, dim=1)
        
        pooled_output = outputs[1].to(torch.float32)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1).to(torch.float32))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
            # 训练时：总损失 = 分类损失 + 元学习损失
            if is_train and meta_loss is not None:
                loss = loss + 0.1 * meta_loss  # 元损失权重可通过配置调整
            outputs = (loss,) + outputs
        
        # 根据参数决定返回值
        if return_single_modal_logits:
            return outputs, pooled_output, condition, generated_ctx, text_logits, video_logits, audio_logits
        else:
            return outputs, pooled_output, condition, generated_ctx


class Cons_Model(BertPreTrainedModel):
    """对比学习模型（确保兼容性）"""
    def __init__(self, config, args, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None
        self.args = args
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        condition_idx,
        ctx,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        encoder_extended_attention_mask = None
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        for i in range(embedding_output.shape[0]):
            embedding_output[i, condition_idx[i] - self.args.prompt_len : condition_idx[i], :] = ctx[i]

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


# 包装类避免类名冲突
class MCWP_Wrapper(nn.Module):
    def __init__(self, args):
        super(MCWP_Wrapper, self).__init__()
        
        # 加载预训练模型（路径根据实际情况调整）
        self.model = MCWP_Model.from_pretrained(
            '/home/jiamengyao/MIntRec-main/bert-base-uncased', 
            # cache_dir=args.cache_path, 
            cache_dir="./cache",
            args=args
        ).float()  # 确保模型为float32
        self.cons_model = Cons_Model.from_pretrained(
            '/home/jiamengyao/MIntRec-main/bert-base-uncased', 
            # cache_dir=args.cache_path, 
            cache_dir="./cache",
            args=args
        ).float()  # 确保模型为float32

        self.ctx_vectors = self._init_ctx(args)
        self.ctx = nn.Parameter(self.ctx_vectors)

        self.label_len = args.label_len
        args.feat_size = args.text_feat_dim
        args.video_feat_size = args.video_feat_dim
        args.audio_feat_size = args.audio_feat_dim

    def _init_ctx(self, args):
        ctx = torch.empty(args.prompt_len, args.text_feat_dim, dtype=torch.float32)
        nn.init.trunc_normal_(ctx)
        return ctx

    
    def forward(self, text_feats, video_feats, audio_feats, cons_text_feats, condition_idx, quality_features, is_train=False, return_single_modal_logits=False):
        """新增参数：
            quality_features: 模态质量特征 [batch_size, 3]
            is_train: 是否训练模式
            return_single_modal_logits: 是否返回单模态预测logits
        """
        # 确保所有输入特征为float32
        video_feats = video_feats.float()
        audio_feats = audio_feats.float()
        quality_features = quality_features.float()

        # 处理正常样本（传入新增参数）
        if return_single_modal_logits:
            outputs, pooled_output, condition, generated_ctx, text_logits, video_logits, audio_logits = self.model(
                text=text_feats,
                visual=video_feats,
                acoustic=audio_feats,
                condition_idx=condition_idx, 
                ctx=self.ctx,
                quality_features=quality_features,
                is_train=is_train,
                return_single_modal_logits=True
            )
        else:
            outputs, pooled_output, condition, generated_ctx = self.model(
                text=text_feats,
                visual=video_feats,
                acoustic=audio_feats,
                condition_idx=condition_idx, 
                ctx=self.ctx,
                quality_features=quality_features,
                is_train=is_train,
                return_single_modal_logits=False
            )
            text_logits = video_logits = audio_logits = None

        # 处理增强样本
        cons_input_ids, cons_input_mask, cons_segment_ids = cons_text_feats[:, 0], cons_text_feats[:, 1], cons_text_feats[:, 2]
        cons_outputs = self.cons_model(
            input_ids=cons_input_ids, 
            condition_idx=condition_idx,
            ctx=generated_ctx,
            token_type_ids=cons_segment_ids, 
            attention_mask=cons_input_mask
        )
        last_hidden_state = cons_outputs.last_hidden_state
        cons_condition_tuple = tuple(last_hidden_state[torch.arange(last_hidden_state.shape[0]), condition_idx.view(-1) + i, :].unsqueeze(1) for i in range(self.label_len))
        cons_condition = torch.cat(cons_condition_tuple, dim=1)

        # 根据参数决定返回值
        if return_single_modal_logits:
            return outputs[0], pooled_output, condition.mean(dim=1), cons_condition.mean(dim=1), text_logits, video_logits, audio_logits
        else:
            return outputs[0], pooled_output, condition.mean(dim=1), cons_condition.mean(dim=1)
