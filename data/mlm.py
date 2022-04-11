"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

MLM datasets
"""
import random
import sys

import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip

from .data import (DetectFeatTxtTokDataset, TxtTokLmdb,
                   pad_tensors, get_gather_index)
from .word_region_util import obj2bert
import numpy as np


# KevinHwang@220306
def _get_img_and_txt_swap(input_ids, img_soft_labels, num_bb, mask_prob=0.25):
    # 使用 argmax 取得每一个 region 的实体标签序号
    # background class should not be the target
    argmax_soft_labels = torch.argmax(img_soft_labels[:, 1:-1], dim=1).tolist()
    # 将实体标签序号转化为 token（每一个实体标签序号可能对应多个 token）
    converted_argmax = [obj2bert[obj_label] if obj_label in obj2bert.keys() else [] for obj_label in
                        argmax_soft_labels]
    # 获取所有的实体 token 并去重
    obj_tokens = set(tk for tokens in converted_argmax for tk in tokens)
    word_region_map = {}  # 保存需要进行交换的 word 和 region ，word 为键（NOTE: 注意 +1，要考虑到填充 [CLS] 后的情况）， region 为值
    txt_len = len(input_ids)
    txt_swap = [False] * txt_len
    img_swap = [False] * num_bb
    img_mask = [False] * num_bb

    if len(obj_tokens) > 0:
        # 选择出一种 token，将所有的该 token 与图片中实体替换
        # tar_token = random.sample(no_repeat, 1)[0]

        # 遍历所有的实体 token

        # 如果 input_ids 中有实体 token ，就查找图片中对应的实体 region 进行替换，如果有多个对应的 region 就随机选一个
        for i, tk in enumerate(input_ids):
            if tk in obj_tokens:
                txt_swap[i] = True
                tar_region_idxs = []
                for j, rg in enumerate(converted_argmax):
                    if tk in rg:
                        tar_region_idxs.append(j)
                if len(tar_region_idxs) > 0:
                    # NOTE: 注意 + 1，要考虑到填充[CLS] 后的情况
                    tar_index = random.choice(tar_region_idxs)
                    img_swap[tar_index] = True
                    word_region_map[i + 1] = tar_index

    chosen_regions_idx = set(word_region_map.values())
    # 获取需要真正 mask 掉的 region
    for i in range(num_bb):
        if i not in chosen_regions_idx and random.random() < mask_prob:
            img_mask[i] = True
    if not any(img_swap) and not any(img_mask):
        # at least mask 1
        img_mask[random.choice(range(num_bb))] = True
    '''
    # -------------------------------------------------[调试代码]-----------------------------------------------------
    # img_swap_tgt:
    print('obj_tokens:', obj_tokens)
    # obj_tokens: [4355, 4876, 16024, 8090, 17180, 5020, 4637, 24998, 2095, 3504, 13624, 4282, 1339, 3392, 7366,
    # 5324, 4694, 4705, 2146, 7404, 3439, 2928, 1520]
    print('argmax_soft_labels:', argmax_soft_labels)
    # argmax_soft_labels: [118, 248, 1330, 1066, 959, 1281, 1330, 959, 959, 959, 1010, 800, 327, 598, 909, 959, 959,
    # 248, 231, 395, 514, 1330, 345, 1414, 959, 919, 231]
    print('img_mask:', img_mask)
    print('word_region_maps:', word_region_maps)
    # -----------------------------------------------[调试代码 END]---------------------------------------------------
    '''
    return img_swap, txt_swap, img_mask, word_region_map


def random_word(tokens, vocab_range, mask, txt_swap):
    """
    Masking some random tokens for Language Model task with probabilities as in
        the original BERT paper.
    :param tokens: list of int, tokenized sentence.
    :param vocab_range: for choosing a random word
    :return: (list of int, list of int), masked tokens and related labels for
        LM prediction
    """
    output_label = [-1] * len(tokens)

    for i, token in enumerate(tokens):
        # 选中的实体 token 全部掩码，其余以 15% 概率掩码
        if txt_swap[i]:
            tokens[i] = mask
            output_label[i] = token
            continue

        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = mask

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(range(*vocab_range)))

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            output_label[i] = token
        else:
            # no masking token (will be ignored by loss function later)
            output_label[i] = -1
    if all(o == -1 for o in output_label):
        # at least mask 1
        output_label[0] = tokens[0]
        tokens[0] = mask

    return tokens, output_label


class MlmDataset(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db):
        assert isinstance(txt_db, TxtTokLmdb)
        super().__init__(txt_db, img_db)

    # KevinHwang@220223：重写 _get_img_feat 函数，取得 img_soft_label
    def _get_img_feat(self, fname):
        img_dump = self.img_db.get_dump(fname)
        num_bb = self.img_db.name2nbb[fname]
        img_feat = torch.tensor(img_dump['features'])
        bb = torch.tensor(img_dump['norm_bb'])
        img_pos_feat = torch.cat([bb, bb[:, 4:5] * bb[:, 5:]], dim=-1)
        img_soft_label = torch.tensor(img_dump['soft_labels'])
        return img_feat, img_pos_feat, img_soft_label, num_bb

    def __getitem__(self, i):
        """
        Return:
        - input_ids    : (L, ), i.e., [cls, wd, wd, ..., sep, 0, 0], 0s padded
        - img_feat     : (num_bb, d)
        - img_pos_feat : (num_bb, 7)
        - attn_masks   : (L + num_bb, ), ie., [1, 1, ..., 0, 0, 1, 1]
        - txt_labels   : (L, ), [-1, -1, wid, -1, -1, -1]
        0's padded so that (L + num_bb) % 8 == 0
        """

        # example 类型为 dara.data.TxtTokLmdb
        example = super().__getitem__(i)

        # KevinHwang@220223: 改进后的 img input ，可以取得 img_soft_labels
        img_feat, img_pos_feat, img_soft_labels, num_bb = self._get_img_feat(example['img_fname'])

        # text input
        input_ids = example['input_ids']

        # KevinHwang: get txt_swap
        _, txt_swap, _, word_region_map = _get_img_and_txt_swap(input_ids, img_soft_labels, num_bb)

        # text input
        input_ids, txt_labels = self.create_mlm_io(example['input_ids'], txt_swap)

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)
        '''
        # -------------------------------------------------[调试代码]-----------------------------------------------------
        print('input_ids: ', input_ids)
        # input_ids:  tensor([101, 170, 13559, 1199, 17655, 1116, 1105, 1199, 103, 8483, 102])
        from data import bert_base_cased_vocab
        tokens = [bert_base_cased_vocab.vocab[str(input_id.item())] for input_id in input_ids]
        print('tokens: ', tokens)
        # tokens:  ['[CLS]', 'a', 'doll', 'some', 'suitcase', '@@##s', 'and', 'some', '[MASK]', 'bags', '[SEP]']
        print('img_feat: ', img_feat)
        # img_feat:  tensor([[0.0000, 0.0000, 0.4736,  ..., 0.0000, 0.0000, 0.0000],
        #                    [0.1102, 0.0607, 4.6328,  ..., 0.0000, 1.1123, 1.8496],
        #                    [0.8599, 0.3428, 1.1650,  ..., 0.0000, 0.0000, 0.2844],
        #                    ...,
        #                    [1.1113, 0.3022, 0.0202,  ..., 0.0000, 0.0000, 0.0319],
        #                    [0.4514, 0.3994, 0.1245,  ..., 0.2024, 3.4238, 1.7256],
        #                    [2.3730, 5.9336, 0.1826,  ..., 0.0000, 1.4072, 0.1669]])
        print('img_pos_feat: ', img_pos_feat)
        # img_pos_feat:  tensor([[7.0374e-02, 7.8369e-01, 1.5503e-01, 9.2822e-01, 8.4595e-02, 1.4441e-01, 1.2216e-02],
        #                        [9.3066e-01, 0.0000e+00, 9.9854e-01, 9.8999e-02, 6.8237e-02, 9.8999e-02, 6.7554e-03],
        #                        [3.4717e-01, 6.5820e-01, 6.7969e-01, 9.9854e-01, 3.3228e-01, 3.4009e-01, 1.1300e-01],
        #                        [2.6718e-02, 6.2891e-01, 3.8574e-01, 9.9854e-01, 3.5913e-01, 3.6963e-01, 1.3275e-01],
        #                        [1.9226e-01, 0.0000e+00, 6.3623e-01, 9.3018e-01, 4.4409e-01, 9.3018e-01, 4.1308e-01],
        #                        [3.5425e-01, 4.4214e-01, 9.9854e-01, 9.9854e-01, 6.4453e-01, 5.5615e-01, 3.5846e-01],
        #                        [8.7280e-02, 4.5605e-01, 3.6255e-01, 9.7705e-01, 2.7539e-01, 5.2100e-01, 1.4348e-01],
        #                        [1.2091e-01, 3.8483e-02, 4.5581e-01, 4.4336e-01, 3.3472e-01, 4.0479e-01, 1.3549e-01],
        #                        [7.9443e-01, 5.3320e-01, 8.8867e-01, 6.6553e-01, 9.4360e-02, 1.3196e-01, 1.2452e-02],
        #                        [0.0000e+00, 6.0699e-02, 1.3403e-01, 9.9854e-01, 1.3403e-01, 9.3750e-01, 1.2566e-01],
        #                        [6.0059e-01, 2.5146e-01, 7.4463e-01, 4.4922e-01, 1.4417e-01, 1.9763e-01, 2.8492e-02],
        #                        [4.4434e-01, 2.5421e-02, 5.8398e-01, 2.1252e-01, 1.3977e-01, 1.8713e-01, 2.6156e-02],
        #                        [1.2842e-01, 0.0000e+00, 5.3662e-01, 8.3252e-01, 4.0820e-01, 8.3252e-01, 3.3984e-01],
        #                        [0.0000e+00, 3.1152e-01, 6.8799e-01, 9.9854e-01, 6.8799e-01, 6.8652e-01, 4.7232e-01],
        #                        [7.4023e-01, 8.5205e-01, 9.7510e-01, 9.9854e-01, 2.3499e-01, 1.4636e-01, 3.4393e-02],
        #                        [6.1475e-01, 2.8101e-01, 7.0605e-01, 4.2480e-01, 9.1370e-02, 1.4404e-01, 1.3161e-02],
        #                        [0.0000e+00, 4.6899e-01, 8.1104e-01, 9.9854e-01, 8.1104e-01, 5.2930e-01, 4.2928e-01],
        #                        [6.3281e-01, 0.0000e+00, 9.9854e-01, 8.0762e-01, 3.6597e-01, 8.0762e-01, 2.9556e-01],
        #                        [2.8394e-01, 7.1875e-01, 7.2803e-01, 9.9854e-01, 4.4434e-01, 2.7979e-01, 1.2432e-01],
        #                        [3.2666e-01, 7.9346e-01, 7.3047e-01, 9.9854e-01, 4.0356e-01, 2.0508e-01, 8.2762e-02],
        #                        [4.6484e-01, 2.3352e-01, 8.0518e-01, 7.2998e-01, 3.4009e-01, 4.9658e-01, 1.6888e-01],
        #                        [7.3862e-04, 5.3125e-01, 9.4727e-02, 8.7012e-01, 9.3994e-02, 3.3887e-01, 3.1852e-02]])
        print('attn_masks: ', attn_masks)
        # attn_masks:  tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #                      1, 1, 1, 1])
        print('txt_labels: ', txt_labels)
        # txt_labels:  tensor([-1, -1, -1, -1, -1, -1, -1, -1, 1168, -1, -1])
        exit(1)
        # -----------------------------------------------[调试代码 END]---------------------------------------------------
        '''
        return input_ids, img_feat, img_pos_feat, attn_masks, txt_labels, word_region_map

    def create_mlm_io(self, input_ids, txt_swap):
        input_ids, txt_labels = random_word(input_ids,
                                            self.txt_db.v_range,
                                            self.txt_db.mask, txt_swap)
        input_ids = torch.tensor([self.txt_db.cls_]
                                 + input_ids
                                 + [self.txt_db.sep])
        txt_labels = torch.tensor([-1] + txt_labels + [-1])
        return input_ids, txt_labels


def mlm_collate(inputs):
    """
    Return:
    :input_ids    (n, max_L) padded with 0
    :position_ids (n, max_L) padded with 0
    :txt_lens     list of [txt_len]
    :img_feat     (n, max_num_bb, feat_dim)
    :img_pos_feat (n, max_num_bb, 7)
    :num_bbs      list of [num_bb]
    :attn_masks   (n, max_{L + num_bb}) padded with 0
    :txt_labels   (n, max_L) padded with -1
    """
    (input_ids, img_feats, img_pos_feats, attn_masks, txt_labels, word_region_maps
     ) = map(list, unzip(inputs))

    # text batches
    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    # image batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'word_region_maps': word_region_maps,
             'txt_labels': txt_labels}
    return batch
