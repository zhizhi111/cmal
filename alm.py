"""
alm dataset
"""
import random

import numpy as np
import torch
from toolz.sandbox import unzip
from torch.nn.utils.rnn import pad_sequence

from .data import (DetectFeatTxtTokDataset, DetectFeatLmdb, TxtTokLmdb,
                   pad_tensors, get_gather_index)
from .word_region_util import obj2bert


def _get_img_mask(mask_prob, num_bb):
    img_mask = [random.random() < mask_prob for _ in range(num_bb)]
    if not any(img_mask):
        # at least mask 1
        img_mask[random.choice(range(num_bb))] = True
    return img_mask


def _get_img_tgt_mask(img_mask, txt_len):
    z = torch.zeros(txt_len, dtype=torch.bool)
    img_mask_tgt = torch.cat([z, img_mask], dim=0)
    return img_mask_tgt


def _get_feat_target(img_feat, img_masks):
    img_masks_ext = img_masks.unsqueeze(-1).expand_as(img_feat)  # (n, m, d)
    feat_dim = img_feat.size(-1)
    feat_targets = img_feat[img_masks_ext].contiguous().view(
        -1, feat_dim)  # (s, d)
    return feat_targets


def _mask_img_feat(img_feat, img_masks):
    img_masks_ext = img_masks.unsqueeze(-1).expand_as(img_feat)
    img_feat_masked = img_feat.data.masked_fill(img_masks_ext, 0)
    return img_feat_masked


# KevinHwang@220306
def _get_mask_and_map(input_ids, img_soft_label, num_bb):
    # 使用 argmax 取得每一个 region 的实体标签序号
    # background class should not be the target
    argmax_soft_labels = torch.argmax(img_soft_label[:, 1:-1], dim=1).tolist()
    # 将实体标签序号转化为 token（每一个实体标签序号可能对应多个 token）
    converted_argmax = [obj2bert[obj_label] if obj_label in obj2bert.keys() else [] for obj_label in
                        argmax_soft_labels]
    obj_tokens = list(set(tk for tokens in converted_argmax for tk in tokens))  # 获取所有的实体 token 并去重
    txt_len = len(input_ids)
    obj_txt_mask = [False] * txt_len  # input_ids 的 mask 遮罩，True 为需要 mask 的
    obj_img_mask = [False] * num_bb  # region 的 mask 遮罩，True 为需要 mask 的
    word_region_map = {}  # 获取 word 和 region 的对应关系，key是 word 的位置索引， region 是区域的索引列表

    if len(obj_tokens) > 0:
        # 遍历所有的实体 token
        for tar_token in obj_tokens:
            for i, tk in enumerate(input_ids):
                if tk == tar_token:
                    obj_txt_mask[i] = True
                    tar_region_idxs = []
                    for j, rg in enumerate(converted_argmax):
                        # 不重复选择 region 进行交换
                        if j not in word_region_map.values() and tar_token in rg:
                            tar_region_idxs.append(j)
                    if len(tar_region_idxs) > 0:
                        tar_index = random.choice(tar_region_idxs)
                        # 考虑到后续需要加 [CLS] ，这里是 i+1 而不是 i
                        word_region_map[i + 1] = tar_index
                        obj_img_mask[tar_index] = True
    return obj_txt_mask, obj_img_mask, word_region_map


def random_word(tokens, vocab_range, mask, obj_txt_mask):
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
        if obj_txt_mask[i]:
            # tokens[i] = mask
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


class AlmDataset(DetectFeatTxtTokDataset):
    """ NOTE this Dataset handles distributed training itself
    (for more efficient negative sampling) """

    def __init__(self, txt_db, img_db):
        assert isinstance(txt_db, TxtTokLmdb)
        assert isinstance(img_db, DetectFeatLmdb)
        super().__init__(txt_db, img_db)

        self.txt_db = txt_db
        self.img_db = img_db

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
        example = super().__getitem__(i)
        img_fname = example['img_fname']
        img_feat, img_pos_feat, img_soft_label, num_bb = self._get_img_feat(img_fname)

        # text input
        input_ids = example['input_ids']

        obj_txt_mask, obj_img_mask, word_region_map = _get_mask_and_map(input_ids, img_soft_label, num_bb)

        # 对文本进行掩码
        input_ids, txt_labels = self.create_mlm_io(example['input_ids'], obj_txt_mask)

        real_mask = _get_img_mask(self.mask_prob, num_bb)
        img_mask = torch.tensor([(res1 or res2) for res1, res2 in zip(obj_img_mask, real_mask)])  # 合并上原本 15% 的 mask
        # 对图片进行掩码
        img_mask_tgt = _get_img_tgt_mask(img_mask, len(input_ids))

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        '''
        # -------------------------------------------------[调试代码]-----------------------------------------------------
        print('input_ids: ', input_ids)
        # input_ids:  tensor([  101,  1199,  1894,  2702,  5579,  1200,  8049,  3079,  1105,   170, 17559,  1113,
        #                       170,  9580,   102])
        from data import bert_base_cased_vocab
        tokens = [bert_base_cased_vocab.vocab[str(input_id.item())] for input_id in input_ids]
        print('tokens: ', tokens)
        # tokens:  ['[CLS]', 'some', 'red', 'double', 'deck', '##er', 'buses', 'cars', 'and', 'a', 'policeman', 'on',
        #           'a', 'motorcycle', '[SEP]']
        print('img_feat: ', img_feat)
        # img_feat:  tensor([[2.4768e-01, 1.4758e-01, 9.4482e-01,  ..., 7.6016e+00, 2.3840e-01, 5.6445e+00],
        #                    [1.3847e-02, 6.8176e-02, 5.1758e-01,  ..., 6.3843e-02, 0.0000e+00, 1.8145e+00],
        #                    [7.3828e-01, 1.3367e-01, 2.3105e+00,  ..., 4.1211e+00, 1.1934e+00, 1.5791e+00],
        #                    ...,
        #                    [5.1941e-02, 1.0376e-01, 4.0898e+00,  ..., 3.4414e+00, 2.4817e-01, 3.2598e+00],
        #                    [1.5373e-02, 2.4585e-01, 5.4855e-03,  ..., 1.6553e-01, 9.1553e-02, 0.0000e+00],
        #                    [0.0000e+00, 1.7407e-01, 4.8340e-01,  ..., 1.0664e+00, 0.0000e+00, 7.0859e+00]])
        print('img_pos_feat: ', img_pos_feat)
        # img_pos_feat:  略
        print('word_region_map: ', word_region_map)
        # word_region_map: {7: [3, 14], 13: [5, 16, 36, 50], 6: [0, 2, 10, 48]}
        print('attn_masks: ', attn_masks)
        # tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        exit(1)
        # -----------------------------------------------[调试代码 END]---------------------------------------------------
        '''

        return input_ids, img_feat, img_pos_feat, word_region_map, txt_labels, img_mask, real_mask, img_mask_tgt, attn_masks

    def create_mlm_io(self, input_ids, obj_txt_mask):
        input_ids, txt_labels = random_word(input_ids,
                                            self.txt_db.v_range,
                                            self.txt_db.mask, obj_txt_mask)
        input_ids = torch.tensor([self.txt_db.cls_]
                                 + input_ids
                                 + [self.txt_db.sep])
        txt_labels = torch.tensor([-1] + txt_labels + [-1])
        return input_ids, txt_labels


def alm_collate(inputs):
    (input_ids, img_feats, img_pos_feats, word_region_maps, txt_labels, img_masks, real_masks, img_mask_tgts,
     attn_masks) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    # mask features
    img_masks = pad_sequence(img_masks, batch_first=True, padding_value=0)
    real_masks = pad_sequence(real_masks, batch_first=True, padding_value=0)
    feat_targets = _get_feat_target(img_feat, img_masks)
    img_feat = _mask_img_feat(img_feat, real_masks)  # NOTE:此处只对那 15% 随机掩码的做真正地掩码
    img_mask_tgt = pad_sequence(img_mask_tgts,
                                batch_first=True, padding_value=0)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'word_region_maps': word_region_maps,
             'txt_labels': txt_labels,
             'feat_targets': feat_targets,
             'img_masks': img_masks,
             'img_mask_tgt': img_mask_tgt,
             'attn_masks': attn_masks,
             'gather_index': gather_index}
    return batch
