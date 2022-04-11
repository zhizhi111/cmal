"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

MRM Datasets
"""
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from .data import DetectFeatTxtTokDataset, pad_tensors, get_gather_index
from .word_region_util import obj2bert
import numpy as np


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


class MrfrDataset(DetectFeatTxtTokDataset):
    # KevinHwang@220223：重写 _get_img_feat 函数，取得 img_soft_label
    def _get_img_feat(self, fname):
        img_dump = self.img_db.get_dump(fname)
        num_bb = self.img_db.name2nbb[fname]
        img_feat = torch.tensor(img_dump['features'])
        bb = torch.tensor(img_dump['norm_bb'])
        img_pos_feat = torch.cat([bb, bb[:, 4:5] * bb[:, 5:]], dim=-1)
        img_soft_label = torch.tensor(img_dump['soft_labels'])
        return img_feat, img_pos_feat, img_soft_label, num_bb

    def __init__(self, mask_prob, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_prob = mask_prob

    def __getitem__(self, i):
        """
        Return:
        - input_ids    : (L, ), i.e., [cls, wd, wd, ..., sep, 0, 0], 0s padded
        - img_feat     : (num_bb, d)
        - img_pos_feat : (num_bb, 7)
        - attn_masks   : (L + num_bb, ), ie., [1, 1, ..., 0, 0, 1, 1]
        - img_mask     : (num_bb, ) between {0, 1}
        """
        example = super().__getitem__(i)
        # text input
        input_ids = example['input_ids']

        # KevinHwang@220223: 改进后的 img input ，可以取得 img_soft_labels
        img_feat, img_pos_feat, img_soft_labels, num_bb = self._get_img_feat(example['img_fname'])

        # KevinHwang: get obj_img_mask
        img_swap, _, _, word_region_map = _get_img_and_txt_swap(input_ids, img_soft_labels, num_bb)
        # 合并上原本 15% 的 mask
        img_mask2 = _get_img_mask(self.mask_prob, num_bb)
        img_mask = torch.tensor([(res1 or res2) for res1, res2 in zip(img_swap, img_mask2)])

        # transfer input_ids to tensor from list
        input_ids = torch.tensor([self.txt_db.cls_] + input_ids + [self.txt_db.sep])

        img_mask_tgt = _get_img_tgt_mask(img_mask, len(input_ids))

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)
        '''
        # -------------------------------------------------[调试代码]-----------------------------------------------------
        print('input_ids: ', input_ids)
        # input_ids:  tensor([ 101, 1247, 1132, 1317, 4697, 2033, 2407, 1106, 1886,  119,  102])
        from data import bert_base_cased_vocab
        tokens = [bert_base_cased_vocab.vocab[str(input_id.item())] for input_id in input_ids]
        print('tokens: ', tokens)
        # tokens:  ['[CLS]', 'A', 'couple', 'of', 'traffic', 'lights', 'si', '##llo', '##ue', '##tted', 'by', 'a',
        #           'setting', 'sun', '.', '[SEP]']
        print('img_feat: ', img_feat)
        # img_feat:  tensor([[1.1553, 0.2517, 0.2903,  ..., 0.1884, 0.1270, 0.0876],
        #                    [0.0000, 0.1542, 0.5659,  ..., 0.0000, 0.0000, 0.0193],
        #                    [0.0397, 0.0000, 0.0075,  ..., 0.0000, 0.0000, 0.0000],
        #                    ...,
        #                    [0.6206, 2.5195, 0.0592,  ..., 0.0357, 2.4492, 0.0000],
        #                    [0.0401, 0.0000, 0.0403,  ..., 0.0000, 1.8018, 0.1342],
        #                    [0.5132, 5.3477, 0.4771,  ..., 0.1758, 0.6685, 0.0000]])
        print('img_pos_feat: ', img_pos_feat)
        # img_pos_feat:  tensor([[0.0000, 0.0000, 0.8315, 0.8022, 0.8315, 0.8022, 0.6671],
        #                        [0.8765, 0.7070, 0.9097, 0.7881, 0.0331, 0.0808, 0.0027],
        #                        [0.7314, 0.9019, 0.7686, 0.9805, 0.0374, 0.0785, 0.0029],
        #                        [0.2578, 0.7554, 0.6567, 0.8730, 0.3989, 0.1179, 0.0470],
        #                        [0.4141, 0.7441, 0.4534, 0.8901, 0.0393, 0.1460, 0.0057],
        #                        [0.7705, 0.3967, 0.8086, 0.4443, 0.0382, 0.0477, 0.0018],
        #                        [0.3994, 0.7671, 0.9746, 0.9985, 0.5752, 0.2313, 0.1331],
        #                        [0.6475, 0.7583, 0.9521, 0.8667, 0.3047, 0.1085, 0.0331],
        #                        [0.5874, 0.6553, 0.6748, 0.8813, 0.0871, 0.2262, 0.0197],
        #                        [0.5942, 0.0152, 0.9121, 0.1790, 0.3179, 0.1638, 0.0521],
        #                        [0.7271, 0.4246, 0.8130, 0.6440, 0.0862, 0.2195, 0.0189],
        #                        [0.0626, 0.7627, 0.7671, 0.9639, 0.7046, 0.2013, 0.1418],
        #                        [0.4622, 0.0000, 0.5859, 0.1598, 0.1235, 0.1598, 0.0197],
        #                        [0.1814, 0.5317, 0.9814, 0.7378, 0.7998, 0.2059, 0.1647],
        #                        [0.2927, 0.0412, 0.9321, 0.3728, 0.6396, 0.3315, 0.2121],
        #                        [0.1260, 0.1003, 0.2681, 0.3450, 0.1421, 0.2446, 0.0348],
        #                        [0.0089, 0.0000, 0.2896, 0.1964, 0.2808, 0.1964, 0.0551],
        #                        [0.7461, 0.4685, 0.7998, 0.6138, 0.0538, 0.1453, 0.0078],
        #                        [0.7158, 0.9429, 0.7817, 0.9985, 0.0659, 0.0555, 0.0037],
        #                        [0.5020, 0.0000, 0.9985, 0.2063, 0.4968, 0.2063, 0.1025],
        #                        [0.9043, 0.8442, 0.9414, 0.9102, 0.0370, 0.0657, 0.0024],
        #                        [0.1467, 0.0918, 0.2391, 0.3113, 0.0924, 0.2194, 0.0203],
        #                        [0.4060, 0.0000, 0.9985, 0.4148, 0.5928, 0.4148, 0.2459],
        #                        [0.1385, 0.8848, 0.5889, 0.9985, 0.4507, 0.1135, 0.0511],
        #                        [0.5747, 0.6646, 0.7104, 0.8594, 0.1357, 0.1946, 0.0264],
        #                        [0.0000, 0.7705, 0.2920, 0.8662, 0.2920, 0.0958, 0.0280],
        #                        [0.2288, 0.6523, 0.9487, 0.8906, 0.7202, 0.2386, 0.1719],
        #                        [0.0068, 0.8589, 0.7954, 0.9985, 0.7886, 0.1393, 0.1098],
        #                        [0.0000, 0.0591, 0.2233, 0.9985, 0.2233, 0.9395, 0.2097],
        #                        [0.6729, 0.9102, 0.7065, 0.9829, 0.0336, 0.0729, 0.0024],
        #                        [0.9639, 0.1564, 0.9985, 0.1963, 0.0350, 0.0399, 0.0014],
        #                        [0.1050, 0.7817, 0.5234, 0.8818, 0.4182, 0.0996, 0.0417],
        #                        [0.0648, 0.3560, 0.7725, 0.5757, 0.7075, 0.2198, 0.1555],
        #                        [0.4814, 0.0000, 0.5679, 0.1694, 0.0862, 0.1694, 0.0146],
        #                        [0.1870, 0.7891, 0.2472, 0.8486, 0.0602, 0.0594, 0.0036],
        #                        [0.0000, 0.1050, 0.0287, 0.8965, 0.0287, 0.7915, 0.0227],
        #                        [0.2256, 0.7534, 0.7178, 0.8501, 0.4922, 0.0967, 0.0476],
        #                        [0.4868, 0.0000, 0.5752, 0.1243, 0.0886, 0.1243, 0.0110],
        #                        [0.1501, 0.9248, 0.1835, 0.9805, 0.0333, 0.0558, 0.0019],
        #                        [0.1857, 0.7900, 0.3242, 0.8452, 0.1385, 0.0550, 0.0076],
        #                        [0.0000, 0.7910, 0.1499, 0.9985, 0.1499, 0.2073, 0.0311],
        #                        [0.0142, 0.6250, 0.7695, 0.8081, 0.7554, 0.1831, 0.1383],
        #                        [0.6890, 0.4502, 0.9595, 0.6099, 0.2705, 0.1597, 0.0432],
        #                        [0.0250, 0.7759, 0.4045, 0.8638, 0.3794, 0.0880, 0.0334]])
        print('attn_masks: ', attn_masks)
        # attn_masks:  tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #                      1, 1])
        print('img_mask: ', img_mask)
        # img_mask:  tensor([False, False, False,  True, False, False,  True, False, False, False,
        #                  False, False, False,  True, False, False, False, False, False, False,
        #                  False, False, False, False, False, False, False, False, False,  True,
        #                  False,  True, False, False, False, False,  True, False,  True, False,
        #                  False,  True,  True, False])
        print('img_mask_tgt: ', img_mask_tgt)
        # img_mask_tgt:  tensor([False, False, False, False, False, False, False, False, False, False,
        #                        False, False, False, False, False, False, False, False, False,  True,
        #                        False, False,  True, False, False, False, False, False, False,  True,
        #                        False, False, False, False, False, False, False, False, False, False,
        #                        False, False, False, False, False,  True, False,  True, False, False,
        #                        False, False,  True, False,  True, False, False,  True,  True, False])
        exit(1)
        # -----------------------------------------------[调试代码 END]---------------------------------------------------
        '''

        return (input_ids, img_feat, img_pos_feat,
                attn_masks, img_mask, img_mask_tgt, word_region_map)


def mrfr_collate(inputs):
    """
    Return:
    - input_ids    : (n, max_L), i.e., [cls, wd, wd, ..., sep, 0, 0], 0s padded
    - position_ids : (n, max_L)
    - txt_lens     : list of [input_len]
    - img_feat     : (n, max_num_bb, d)
    - img_pos_feat : (n, max_num_bb, 7)
    - num_bbs      : list of [num_bb]
    - attn_masks   : (n, max_{L + num_bb}), ie., [1, 1, ..., 0, 0, 1, 1]
    - img_masks    : (n, max_num_bb) between {0, 1}
    """
    (input_ids, img_feats, img_pos_feats, attn_masks, img_masks, img_mask_tgts, word_region_maps
     ) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    # mask features
    img_masks = pad_sequence(img_masks, batch_first=True, padding_value=0)
    feat_targets = _get_feat_target(img_feat, img_masks)
    img_feat = _mask_img_feat(img_feat, img_masks)
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
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'feat_targets': feat_targets,
             'img_masks': img_masks,
             'img_mask_tgt': img_mask_tgt,
             'word_region_maps': word_region_maps}
    return batch


def _get_targets(img_masks, img_soft_label):
    soft_label_dim = img_soft_label.size(-1)
    img_masks_ext_for_label = img_masks.unsqueeze(-1).expand_as(img_soft_label)
    label_targets = img_soft_label[img_masks_ext_for_label].contiguous().view(
        -1, soft_label_dim)
    return label_targets


class MrcDataset(DetectFeatTxtTokDataset):
    def __init__(self, mask_prob, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_prob = mask_prob

    def _get_img_feat(self, fname):
        img_dump = self.img_db.get_dump(fname)
        num_bb = self.img_db.name2nbb[fname]
        img_feat = torch.tensor(img_dump['features'])
        bb = torch.tensor(img_dump['norm_bb'])
        img_bb = torch.cat([bb, bb[:, 4:5] * bb[:, 5:]], dim=-1)
        img_soft_label = torch.tensor(img_dump['soft_labels'])
        return img_feat, img_bb, img_soft_label, num_bb

    def __getitem__(self, i):
        example = super().__getitem__(i)
        # text input
        input_ids = example['input_ids']

        # KevinHwang@220223: 改进后的 img input ，可以取得 img_soft_labels
        img_feat, img_pos_feat, img_soft_labels, num_bb = self._get_img_feat(example['img_fname'])

        # KevinHwang: get obj_img_mask
        img_swap, _, _, word_region_map = _get_img_and_txt_swap(input_ids, img_soft_labels, num_bb)
        # 合并上原本 15% 的 mask
        img_mask2 = _get_img_mask(self.mask_prob, num_bb)
        img_mask = torch.tensor([(res1 or res2) for res1, res2 in zip(img_swap, img_mask2)])

        # transfer input_ids to tensor from list
        input_ids = torch.tensor([self.txt_db.cls_] + input_ids + [self.txt_db.sep])

        img_mask_tgt = _get_img_tgt_mask(img_mask, len(input_ids))

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        '''
        # -------------------------------------------------[调试代码]-----------------------------------------------------
        print('input_ids: ', input_ids)
        # input_ids:  tensor([101,  2677,   188,  2772,  4616,  1832,  1107,  1103,  1447,  1105, 20964,   119,   102])
        from data import bert_base_cased_vocab
        tokens = [bert_base_cased_vocab.vocab[str(input_id.item())] for input_id in input_ids]
        print('tokens: ', tokens)
        # tokens:  ['[CLS]', 'Three', 's', '##tor', '##ks', 'rest', 'in', 'the', 'water', 'and', 'algae', '.', '[SEP]']
        print('img_feat: ', img_feat)
        # img_feat:  tensor([[6.9727e-01, 0.0000e+00, 1.5879e+00,  ..., 0.0000e+00, 0.0000e+00, 0.0000e+00],
        #                    [1.1299e+00, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00, 7.9785e-01, 4.7363e-02],
        #                    [9.8450e-02, 0.0000e+00, 1.0583e-01,  ..., 0.0000e+00, 5.6885e-01, 3.9749e-03],
        #                    ...,
        #                    [2.1680e-01, 0.0000e+00, 9.0790e-03,  ..., 0.0000e+00, 3.5309e-02, 0.0000e+00],
        #                    [2.4529e-03, 1.3306e-01, 4.0259e-01,  ..., 2.7368e-01, 1.7102e-01, 1.3725e-02],
        #                    [0.0000e+00, 0.0000e+00, 6.1816e-01,  ..., 0.0000e+00, 1.2121e-03, 1.4336e-02]])
        print('img_pos_feat: ', img_pos_feat)
        # img_pos_feat:  tensor([[0.6460, 0.7251, 0.7461, 0.9106, 0.0997, 0.1855, 0.0185],
        #                        [0.6660, 0.7261, 0.7324, 0.7661, 0.0663, 0.0403, 0.0027],
        #                        [0.2615, 0.0938, 0.3516, 0.1492, 0.0901, 0.0554, 0.0050],
        #                        [0.5776, 0.6660, 0.8486, 0.9932, 0.2708, 0.3274, 0.0886],
        #                        [0.0000, 0.0000, 0.9121, 0.4841, 0.9121, 0.4841, 0.4416],
        #                        [0.6499, 0.7310, 0.7446, 0.7930, 0.0952, 0.0622, 0.0059],
        #                        [0.1974, 0.1874, 0.8862, 0.9839, 0.6890, 0.7964, 0.5487],
        #                        [0.2417, 0.1414, 0.5327, 0.2690, 0.2910, 0.1277, 0.0372],
        #                        [0.3777, 0.1814, 0.4924, 0.2549, 0.1147, 0.0734, 0.0084],
        #                        [0.1893, 0.0826, 0.3909, 0.2151, 0.2014, 0.1324, 0.0267],
        #                        [0.2837, 0.1222, 0.4434, 0.1915, 0.1598, 0.0693, 0.0111],
        #                        [0.0194, 0.6094, 0.4519, 0.9263, 0.4326, 0.3171, 0.1372],
        #                        [0.2286, 0.0786, 0.4412, 0.2050, 0.2126, 0.1263, 0.0269],
        #                        [0.6445, 0.8398, 0.7266, 0.9189, 0.0821, 0.0791, 0.0065],
        #                        [0.6475, 0.7891, 0.7373, 0.9062, 0.0895, 0.1170, 0.0105],
        #                        [0.5283, 0.0701, 0.9751, 0.3965, 0.4465, 0.3264, 0.1458],
        #                        [0.2479, 0.1120, 0.3450, 0.1708, 0.0970, 0.0588, 0.0057],
        #                        [0.0000, 0.3030, 0.6606, 0.9985, 0.6606, 0.6958, 0.4597],
        #                        [0.3220, 0.1459, 0.4563, 0.2344, 0.1343, 0.0884, 0.0119],
        #                        [0.3357, 0.2081, 0.4973, 0.2822, 0.1615, 0.0741, 0.0120],
        #                        [0.1187, 0.3372, 0.9126, 0.6191, 0.7939, 0.2817, 0.2237],
        #                        [0.6694, 0.7588, 0.7759, 0.8286, 0.1063, 0.0699, 0.0074],
        #                        [0.6328, 0.7148, 0.6973, 0.7529, 0.0644, 0.0381, 0.0025],
        #                        [0.2673, 0.3904, 0.9985, 0.9985, 0.7310, 0.6084, 0.4447],
        #                        [0.2700, 0.1154, 0.3516, 0.1641, 0.0816, 0.0486, 0.0040]])
        print('img_soft_labels: ', img_soft_labels)
        # img_soft_labels:  tensor([[1.4334e-03, 5.9605e-08, 5.9605e-08,  ..., 3.3855e-04, 6.9141e-06, 3.3379e-06],
        #                           [7.9529e-02, 2.3842e-07, 1.1921e-07,  ..., 6.1214e-05, 3.3259e-05, 1.8001e-05],
        #                           [8.8989e-02, 6.5565e-07, 8.9407e-07,  ..., 3.9816e-05, 2.8014e-05, 2.3627e-04],
        #                           ...,
        #                           [7.0410e-01, 1.7881e-07, 1.1921e-07,  ..., 1.5450e-04, 6.3360e-05, 7.5102e-06],
        #                           [2.8564e-02, 9.5367e-07, 9.5367e-07,  ..., 3.9339e-06, 3.4857e-04, 1.4901e-06],
        #                           [6.5796e-02, 1.0729e-06, 5.3644e-07,  ..., 2.5368e-04, 1.3912e-04, 1.7560e-04]])
        print('attn_masks: ', attn_masks)
        # attn_masks:  tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #                      1, 1, 1, 1, 1, 1, 1, 1, 1])
        print('img_mask: ', img_mask)
        # img_mask:  tensor([False, False,  True, False, False, False, False, False, False, False,
        #                    True, False, False, False, False,  True, False, False, False, False,
        #                    False, False,  True,  True,  True])
        print('img_mask_tgt: ', img_mask_tgt)
        # img_mask_tgt:  tensor([False, False, False, False, False, False, False, False, False, False,
        #                        False, False, False, False, False,  True, False, False, False, False,
        #                        False, False, False,  True, False, False, False, False,  True, False,
        #                        False, False, False, False, False,  True,  True,  True])
        exit(1)
        # -----------------------------------------------[调试代码 END]---------------------------------------------------
        '''

        return (input_ids, img_feat, img_pos_feat,
                img_soft_labels, attn_masks, img_mask, img_mask_tgt, word_region_map)


def mrc_collate(inputs):
    (input_ids, img_feats, img_pos_feats, img_soft_labels,
     attn_masks, img_masks, img_mask_tgts, word_region_maps) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]
    num_bbs = [f.size(0) for f in img_feats]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)
    img_soft_label = pad_tensors(img_soft_labels, num_bbs)
    img_masks = pad_sequence(img_masks, batch_first=True, padding_value=0)
    label_targets = _get_targets(img_masks, img_soft_label)

    img_feat = _mask_img_feat(img_feat, img_masks)
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
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'img_masks': img_masks,
             'img_mask_tgt': img_mask_tgt,
             'label_targets': label_targets,
             'word_region_maps': word_region_maps}
    return batch
