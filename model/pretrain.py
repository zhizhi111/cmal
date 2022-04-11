"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER for pretraining
"""
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm

from .layer import GELU, BertOnlyMLMHead
from .model import UniterModel, UniterPreTrainedModel
from .ot import optimal_transport_dist


class RegionFeatureRegression(nn.Module):
    " for MRM"

    def __init__(self, hidden_size, feat_dim, img_linear_weight):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 GELU(), )
        self.norm = LayerNorm(hidden_size, eps=1e-12)

        self.weight = img_linear_weight
        self.bias = nn.Parameter(torch.zeros(feat_dim))

    def forward(self, input_):
        hidden1 = self.net(input_)
        hidden = self.norm(hidden1.float())
        output = F.linear(hidden, self.weight.t(), self.bias)
        return output


class RegionClassification(nn.Module):
    " for MRC(-kl)"

    def __init__(self, hidden_size, label_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 GELU(), )
        self.norm = LayerNorm(hidden_size, eps=1e-12)
        self.liner = nn.Linear(hidden_size, label_dim)

    def forward(self, input_):
        net_out = self.net(input_)
        norm_out = self.norm(net_out.float())
        output = self.liner(norm_out)
        return output


class UniterForPretraining(UniterPreTrainedModel):
    """ UNITER pretraining """

    def __init__(self, config, img_dim, img_label_dim):
        super().__init__(config)
        self.uniter = UniterModel(config, img_dim)
        self.cls = BertOnlyMLMHead(
            config, self.uniter.embeddings.word_embeddings.weight)
        self.feat_regress = RegionFeatureRegression(
            config.hidden_size, img_dim,
            self.uniter.img_embeddings.img_linear.weight)
        self.region_classifier = RegionClassification(
            config.hidden_size, img_label_dim)
        self.itm_output = nn.Linear(config.hidden_size, 2)
        self.temp = nn.Parameter(torch.ones([]).cuda() * 0.07)
        self.apply(self.init_weights)

    def forward(self, batch, task, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attention_mask = batch['attn_masks']
        gather_index = batch['gather_index']
        word_region_maps = batch['word_region_maps']
        if task == 'mlm':
            txt_labels = batch['txt_labels']
            return self.forward_mlm(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index, word_region_maps,
                                    txt_labels, compute_loss)
        elif task == 'mrfr':
            img_mask_tgt = batch['img_mask_tgt']
            img_masks = batch['img_masks']
            mrfr_feat_target = batch['feat_targets']
            return self.forward_mrfr(input_ids, position_ids,
                                     img_feat, img_pos_feat,
                                     attention_mask, gather_index, word_region_maps,
                                     img_masks, img_mask_tgt,
                                     mrfr_feat_target, compute_loss)
        elif task == 'itm':
            targets = batch['targets']
            ot_inputs = batch['ot_inputs']
            return self.forward_itm(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index,
                                    targets, ot_inputs, compute_loss)
        elif task.startswith('mrc'):
            img_mask_tgt = batch['img_mask_tgt']
            img_masks = batch['img_masks']
            mrc_label_target = batch['label_targets']
            return self.forward_mrc(input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    attention_mask, gather_index, word_region_maps,
                                    img_masks, img_mask_tgt,
                                    mrc_label_target, task, compute_loss)
        elif task.startswith('wrc'):
            return self.forward_wrc(input_ids, position_ids, img_feat, img_pos_feat,
                                    attention_mask, gather_index, word_region_maps,
                                    compute_loss)
        elif task.startswith('alm'):
            txt_labels = batch['txt_labels']
            img_mask_tgt = batch['img_mask_tgt']
            img_masks = batch['img_masks']
            mrc_label_target = batch['label_targets']
            return self.forward_alm(input_ids, position_ids, img_feat, img_pos_feat,
                                    attention_mask, gather_index, word_region_maps,
                                    txt_labels, img_mask_tgt, img_masks, mrc_label_target,
                                    compute_loss)
        else:
            raise ValueError('invalid task')

    def forward_mlm(self, input_ids, position_ids, img_feat, img_pos_feat,
                    attention_mask, gather_index, word_region_maps,
                    txt_labels, compute_loss=True):
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index, word_region_maps=word_region_maps, swap_it=1,
                                      output_all_encoded_layers=False)
        # get only the text part
        sequence_output = sequence_output[:, :input_ids.size(1), :]
        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    txt_labels != -1)
        prediction_scores = self.cls(masked_output)

        if compute_loss:
            masked_lm_loss = F.cross_entropy(prediction_scores,
                                             txt_labels[txt_labels != -1],
                                             reduction='none')
            return masked_lm_loss
        else:
            return prediction_scores

    def _compute_masked_hidden(self, hidden, mask):
        """ get only the masked region (don't compute unnecessary hiddens) """
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked

    def forward_mrfr(self, input_ids, position_ids, img_feat, img_pos_feat,
                     attention_mask, gather_index, word_region_maps,
                     img_masks, img_mask_tgt,
                     feat_targets, compute_loss=True):
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index, word_region_maps=word_region_maps, swap_it=2,
                                      output_all_encoded_layers=False,
                                      img_masks=img_masks)

        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    img_mask_tgt)
        prediction_feat = self.feat_regress(masked_output)

        if compute_loss:
            mrfr_loss = F.mse_loss(prediction_feat, feat_targets,
                                   reduction='none')
            return mrfr_loss
        else:
            return prediction_feat

    def forward_itm(self, input_ids, position_ids, img_feat, img_pos_feat,
                    attention_mask, gather_index, targets, ot_inputs,
                    compute_loss=True):
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False)
        pooled_output = self.uniter.pooler(sequence_output)
        itm_scores = self.itm_output(pooled_output)

        # 取消 OT WRA
        # # OT loss
        # if ot_inputs is not None:
        #     ot_scatter = ot_inputs['ot_scatter']
        #
        #     b = sequence_output.size(0)
        #     tl = input_ids.size(1)
        #     il = img_feat.size(1)
        #     max_l = max(ot_inputs['scatter_max'] + 1, tl + il)
        #
        #     ot_scatter = ot_scatter.unsqueeze(-1).expand_as(sequence_output)
        #     ctx_emb = torch.zeros(b, max_l, self.config.hidden_size,
        #                           dtype=sequence_output.dtype,
        #                           device=sequence_output.device
        #                           ).scatter_(dim=1, index=ot_scatter,
        #                                      src=sequence_output)
        #     txt_emb = ctx_emb[:, :tl, :]
        #     img_emb = ctx_emb[:, tl:tl + il, :]
        #
        #     txt_pad = ot_inputs['txt_pad']
        #     img_pad = ot_inputs['img_pad']
        #     # NOTE: run in fp32 for stability
        #     ot_dist = optimal_transport_dist(txt_emb.float(), img_emb.float(),
        #                                      txt_pad, img_pad).to(txt_emb)
        #     ot_pos_dist = ot_dist.masked_select(targets == 1)
        #     ot_neg_dist = ot_dist.masked_select(targets == 0)
        #     ot_loss = (ot_pos_dist, ot_neg_dist)
        # else:
        #     ot_loss = None
        ot_loss = None

        if compute_loss:
            itm_loss = F.cross_entropy(itm_scores, targets, reduction='none')
            return itm_loss, ot_loss
        else:
            return itm_scores, ot_loss

    def forward_mrc(self, input_ids, position_ids, img_feat, img_pos_feat,
                    attention_mask, gather_index, word_region_maps,
                    img_masks, img_mask_tgt,
                    label_targets, task, compute_loss=True):
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index, word_region_maps=word_region_maps, swap_it=2,
                                      output_all_encoded_layers=False,
                                      img_masks=img_masks)

        # only compute masked regions for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output,
                                                    img_mask_tgt)
        prediction_soft_label = self.region_classifier(masked_output)

        if compute_loss:
            if "kl" in task:
                prediction_soft_label = F.log_softmax(
                    prediction_soft_label, dim=-1)
                mrc_loss = F.kl_div(
                    prediction_soft_label, label_targets, reduction='none')
            else:
                # background class should not be the target
                label_targets = torch.max(label_targets[:, 1:], dim=-1)[1] + 1
                mrc_loss = F.cross_entropy(
                    prediction_soft_label, label_targets,
                    ignore_index=0, reduction='none')
            return mrc_loss
        else:
            return prediction_soft_label

    def forward_wrc(self, input_ids, position_ids, img_feat, img_pos_feat,
                    attention_mask, gather_index, word_region_maps,
                    compute_loss=True):
        # 将 self.temp 范围限制到 0.001 到 0.5 之间，该操作不计入 track 梯度
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False)  # sequence_output.size(): torch.Size([112, 44, 768])
        batch_size = sequence_output.size(0)
        max_txt_len = input_ids.size(1)
        max_img_len = img_feat.size(1)

        # 先将 sequence_output 形状还原
        index = gather_index.unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
        txt_img_output = torch.zeros(batch_size, max_txt_len + max_img_len, self.config.hidden_size,
                                     dtype=torch.float32).cuda().scatter_(1, index, sequence_output)

        # 获取图像和文本对应的输出
        txt_output = txt_img_output[:, :max_txt_len, :]
        img_output = txt_img_output[:, max_txt_len:, :]

        # 进行归一化
        txt_output = F.normalize(txt_output, p=2, dim=2)
        img_output = F.normalize(img_output, p=2, dim=2)

        # 对 img_output 进行转置
        img_output = torch.transpose(img_output, 1, 2)

        # 计算向量乘积
        mat = torch.bmm(txt_output, img_output) / self.temp

        # 选择正样本对
        mat_mask = torch.zeros_like(mat)
        for i in range(batch_size):
            for k, v in word_region_maps[i].items():
                for j in v:
                    mat_mask[i][k][j] = 1
        # 对 mat 和 mat_mask 进行 flatten
        mat = mat.flatten(1)
        mat_mask = mat_mask.flatten(1)

        wrc_loss = -torch.mean(F.log_softmax(mat, dim=1) * mat_mask, dim=1)

        return wrc_loss

    def forward_alm(self, input_ids, position_ids, img_feat, img_pos_feat,
                    attention_mask, gather_index, word_region_maps,
                    txt_labels, img_mask_tgt, img_masks, label_targets, compute_loss=True):
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index, word_region_maps=word_region_maps, swap_it=3,
                                      output_all_encoded_layers=False, img_masks=img_masks)
        batch_size = sequence_output.size(0)
        # get only the text part
        txt_output = sequence_output[:, :input_ids.size(1), :]
        # only compute masked tokens for better efficiency
        txt_masked_output = self._compute_masked_hidden(txt_output,
                                                        txt_labels != -1)
        txt_prediction_scores = self.cls(txt_masked_output)

        # only compute masked regions for better efficiency
        img_masked_output = self._compute_masked_hidden(sequence_output,
                                                        img_mask_tgt)
        img_prediction_soft_label = self.region_classifier(img_masked_output)

        if compute_loss:
            # mlm
            masked_lm_loss = F.cross_entropy(txt_prediction_scores,
                                             txt_labels[txt_labels != -1])
            # mrc
            # background class should not be the target
            label_targets = torch.max(label_targets[:, 1:], dim=-1)[1] + 1
            mrc_loss = F.cross_entropy(
                img_prediction_soft_label, label_targets,
                ignore_index=0)

            return (masked_lm_loss + mrc_loss).repeat(batch_size)
        else:
            return txt_prediction_scores, img_prediction_soft_label
