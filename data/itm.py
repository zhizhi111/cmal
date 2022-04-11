"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Itm dataset
"""
from collections import defaultdict
import copy
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from cytoolz import concat
import numpy as np

from .data import (DetectFeatTxtTokDataset, DetectFeatLmdb, TxtTokLmdb,
                   pad_tensors, get_gather_index, get_ids_and_lens)
from .sampler import TokenBucketSampler


class TokenBucketSamplerForItm(TokenBucketSampler):
    def __init__(self, dset, *args, **kwargs):
        super().__init__(dset.lens, *args, **kwargs)
        self.dset = dset

    def __iter__(self):
        it = super().__iter__()
        self.dset.new_epoch()
        self._lens = self.dset.lens
        return it


def _has_overlap(la, lb):
    if len(la) < len(lb):
        la, lb = lb, la
    s = set(la)
    return any(b in s for b in lb)


def sample_negative(sample_pool, ground_truths, num_sample):
    """ random and retry """
    outputs = ground_truths[:1]
    while _has_overlap(outputs, ground_truths):
        outputs = random.sample(sample_pool, num_sample)
    return outputs


class ItmDataset(DetectFeatTxtTokDataset):
    """ NOTE this Dataset handles distributed training itself
    (for more efficient negative sampling) """

    def __init__(self, txt_db, img_db, neg_sample_p=0.5):
        assert isinstance(txt_db, TxtTokLmdb)
        assert isinstance(img_db, DetectFeatLmdb)

        self.txt_db = txt_db
        self.img_db = img_db

        self.txt_lens, self.ids = get_ids_and_lens(txt_db)
        self.all_imgs = list(set(txt_db[id_]['img_fname'] for id_ in self.ids))

        self.neg_sample_p = neg_sample_p
        self.new_epoch()

    def new_epoch(self):
        """ should be called every epoch for more randomness"""
        self.labels = np.random.choice(
            [0, 1], size=len(self.ids),
            p=[self.neg_sample_p, 1 - self.neg_sample_p])

        self.lens = []
        self.train_imgs = []
        for i, (id_, tl) in enumerate(zip(self.ids, self.txt_lens)):
            img_fname = super().__getitem__(i)['img_fname']
            if self.labels[i] == 0:
                img_fname = sample_negative(self.all_imgs, [img_fname], 1)[0]
            self.train_imgs.append(img_fname)
            self.lens.append(tl + self.img_db.name2nbb[img_fname])

    def __getitem__(self, i):
        example = super().__getitem__(i)
        # labels and negative images should be sampled every epoch
        ground_truth_label = self.labels[i]
        img_fname = self.train_imgs[i]
        img_feat, img_pos_feat, num_bb = self._get_img_feat(img_fname)

        # text input
        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)
        target = torch.Tensor(1).long()
        target.data.fill_(ground_truth_label)

        '''
        # -------------------------------------------------[调试代码]-----------------------------------------------------
        print('input_ids: ', input_ids)
        # input_ids:  tensor([101, 138, 176, 5132, 15475, 2288, 2041, 1397,  1106, 170, 1353, 2780, 119, 102])
        from data import bert_base_cased_vocab
        tokens = [bert_base_cased_vocab.vocab[str(input_id.item())] for input_id in input_ids]
        print('tokens: ', tokens)
        # tokens:  ['[CLS]', 'A', 'g', '##ira', '##ffe', 'standing', 'alone', 'next', 'to', 'a', 'small', 'tree',
        #           '.', '[SEP]']
        print('img_feat: ', img_feat)
        # img_feat:  tensor([[1.4600e-01, 0.0000e+00, 1.6084e+00,  ..., 8.5156e+00, 8.9722e-02, 1.8613e+00],
        #                    [0.0000e+00, 0.0000e+00, 2.0762e+00,  ..., 1.0188e+01, 8.2275e-02, 2.5176e+00],
        #                    [6.9238e-01, 0.0000e+00, 5.3749e-03,  ..., 2.0508e-01, 8.1250e-01, 5.3406e-03],
        #                    ...,
        #                    [1.5781e+00, 0.0000e+00, 6.5137e-01,  ..., 5.0586e-01, 2.9883e+00, 5.5511e-02],
        #                    [0.0000e+00, 2.3108e-01, 4.6875e+00,  ..., 5.2223e-03, 9.1602e-01, 6.9275e-02],
        #                    [0.0000e+00, 3.6523e-01, 2.6309e+00,  ..., 0.0000e+00, 2.0154e-01, 2.9321e-01]])
        print('img_pos_feat: ', img_pos_feat)
        # img_pos_feat:  tensor([[0.2135, 0.2571, 0.6631, 0.7266, 0.4495, 0.4695, 0.2110],
        #                        [0.4939, 0.4700, 0.9985, 0.9985, 0.5049, 0.5283, 0.2667],
        #                        [0.2908, 0.3367, 0.3330, 0.4937, 0.0422, 0.1570, 0.0066],
        #                        [0.4382, 0.2959, 0.5776, 0.4497, 0.1395, 0.1538, 0.0215],
        #                        [0.5122, 0.6362, 0.5640, 0.6880, 0.0516, 0.0517, 0.0027],
        #                        [0.3550, 0.6016, 0.3804, 0.6440, 0.0253, 0.0426, 0.0011],
        #                        [0.8901, 0.4844, 0.9985, 0.5732, 0.1088, 0.0889, 0.0097],
        #                        [0.2377, 0.8599, 0.3142, 0.9683, 0.0767, 0.1084, 0.0083],
        #                        [0.5806, 0.5571, 0.7134, 0.7090, 0.1327, 0.1517, 0.0201],
        #                        [0.6890, 0.3921, 0.8418, 0.5293, 0.1527, 0.1375, 0.0210],
        #                        [0.2595, 0.3818, 0.2959, 0.4946, 0.0363, 0.1127, 0.0041],
        #                        [0.7100, 0.0000, 0.8076, 0.1004, 0.0977, 0.1004, 0.0098],
        #                        [0.6206, 0.5298, 0.6465, 0.5732, 0.0261, 0.0434, 0.0011],
        #                        [0.6494, 0.0021, 0.9116, 0.5103, 0.2622, 0.5083, 0.1333],
        #                        [0.1108, 0.3037, 0.1724, 0.3564, 0.0616, 0.0528, 0.0033],
        #                        [0.0000, 0.2474, 0.2756, 0.9985, 0.2756, 0.7510, 0.2070],
        #                        [0.0698, 0.0000, 0.7319, 0.4592, 0.6621, 0.4592, 0.3041],
        #                        [0.7300, 0.5605, 0.9766, 0.7002, 0.2462, 0.1398, 0.0344],
        #                        [0.4807, 0.7764, 0.5356, 0.9482, 0.0547, 0.1719, 0.0094],
        #                        [0.8105, 0.7500, 0.9570, 0.8311, 0.1461, 0.0811, 0.0118],
        #                        [0.1779, 0.6372, 0.2349, 0.7402, 0.0570, 0.1028, 0.0059],
        #                        [0.7583, 0.2155, 0.8135, 0.2634, 0.0548, 0.0481, 0.0026],
        #                        [0.2241, 0.5620, 0.8652, 0.9985, 0.6411, 0.4363, 0.2797],
        #                        [0.4941, 0.5054, 0.6216, 0.5952, 0.1276, 0.0900, 0.0115],
        #                        [0.0455, 0.6235, 0.8931, 0.9985, 0.8472, 0.3748, 0.3175],
        #                        [0.1823, 0.7021, 0.2260, 0.7979, 0.0437, 0.0956, 0.0042],
        #                        [0.5981, 0.2471, 0.6328, 0.3005, 0.0346, 0.0534, 0.0018],
        #                        [0.2607, 0.0667, 0.4033, 0.3262, 0.1428, 0.2595, 0.0371],
        #                        [0.0000, 0.0283, 0.2588, 0.4841, 0.2588, 0.4558, 0.1180],
        #                        [0.7261, 0.1017, 0.8379, 0.4106, 0.1117, 0.3088, 0.0345],
        #                        [0.6587, 0.8647, 0.7207, 0.9702, 0.0618, 0.1055, 0.0065],
        #                        [0.2671, 0.5991, 0.3174, 0.7778, 0.0501, 0.1787, 0.0090],
        #                        [0.4778, 0.4683, 0.5059, 0.5122, 0.0278, 0.0439, 0.0012],
        #                        [0.7417, 0.2505, 0.8159, 0.3394, 0.0739, 0.0887, 0.0066],
        #                        [0.1036, 0.2839, 0.1682, 0.3408, 0.0646, 0.0569, 0.0037],
        #                        [0.1018, 0.2651, 0.1698, 0.3188, 0.0681, 0.0537, 0.0037],
        #                        [0.4158, 0.4937, 0.6470, 0.6084, 0.2312, 0.1151, 0.0266],
        #                        [0.0000, 0.0000, 0.3862, 0.2698, 0.3862, 0.2698, 0.1042],
        #                        [0.1144, 0.1935, 0.9824, 0.8896, 0.8682, 0.6963, 0.6045],
        #                        [0.7554, 0.2184, 0.8145, 0.2874, 0.0591, 0.0689, 0.0041],
        #                        [0.7974, 0.7485, 0.9082, 0.8315, 0.1107, 0.0833, 0.0092],
        #                        [0.4258, 0.0000, 0.9985, 0.5708, 0.5728, 0.5708, 0.3269],
        #                        [0.6934, 0.0000, 0.8208, 0.1375, 0.1276, 0.1375, 0.0175],
        #                        [0.4758, 0.4441, 0.5396, 0.4863, 0.0636, 0.0421, 0.0027],
        #                        [0.1831, 0.6431, 0.2291, 0.7637, 0.0461, 0.1207, 0.0056],
        #                        [0.0000, 0.7095, 0.0351, 0.7720, 0.0351, 0.0626, 0.0022],
        #                        [0.0000, 0.3962, 0.1270, 0.4661, 0.1270, 0.0698, 0.0089],
        #                        [0.0000, 0.5239, 0.2593, 0.9985, 0.2593, 0.4744, 0.1230],
        #                        [0.0000, 0.0000, 0.4468, 0.1986, 0.4468, 0.1986, 0.0887],
        #                        [0.7529, 0.2781, 0.8154, 0.3374, 0.0625, 0.0594, 0.0037],
        #                        [0.7231, 0.3096, 0.8120, 0.3586, 0.0889, 0.0491, 0.0044]])
        print('attn_masks: ', attn_masks)
        # attn_masks:  tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #                      1, 1, 1, 1, 1, 1])
        print('target: ', target)
        # target:  tensor([0])
        exit(1)
        # -----------------------------------------------[调试代码 END]---------------------------------------------------
        '''

        return input_ids, img_feat, img_pos_feat, attn_masks, target


def itm_collate(inputs):
    (input_ids, img_feats, img_pos_feats, attn_masks, targets
     ) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    targets = torch.cat(targets, dim=0)
    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'targets': targets}
    return batch


def _compute_ot_scatter(txt_lens, max_txt_len, joint_len):
    """
    输入的 joint_len 为 max(txt_len + img_len)
    return: 返回值 size 为 (len(txt_lens), joint_len)
            其中每一行的前一部分（对应文本的那一部分）为从 0 到 txt_len，
            每一行的后一部分（对应图像的那一部分）为从 max_txt_len 到 max_txt_len + img_len
    """
    ot_scatter = torch.arange(0, joint_len, dtype=torch.long
                              ).unsqueeze(0).repeat(len(txt_lens), 1)
    for i, tl in enumerate(txt_lens):
        max_ind = max_txt_len + (joint_len - tl)
        ot_scatter.data[i, tl:] = torch.arange(max_txt_len, max_ind,
                                               dtype=torch.long).data
    return ot_scatter


def _compute_pad(lens, max_len):
    """
    return: 返回值 size 为 (len(lens), max_len)
            其中每一行对应有效内容的一部分填充 0，后面的填充 1
    """
    pad = torch.zeros(len(lens), max_len, dtype=torch.bool)
    for i, l in enumerate(lens):
        pad.data[i, l:].fill_(1)
    return pad


def itm_ot_collate(inputs):
    (input_ids, img_feats, img_pos_feats, attn_masks, targets
     ) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    targets = torch.cat(targets, dim=0)
    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    # OT inputs
    max_tl = max(txt_lens)
    max_nbb = max(num_bbs)
    ot_scatter = _compute_ot_scatter(txt_lens, max_tl, attn_masks.size(1))
    txt_pad = _compute_pad(txt_lens, max_tl)
    img_pad = _compute_pad(num_bbs, max_nbb)
    ot_inputs = {'ot_scatter': ot_scatter,
                 'scatter_max': ot_scatter.max().item(),
                 'txt_pad': txt_pad,
                 'img_pad': img_pad}

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'targets': targets,
             'ot_inputs': ot_inputs}
    return batch


class ItmRankDataset(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db, neg_sample_size=1):
        assert neg_sample_size > 0, \
            "ItmRankDataset need at least 1 negative sample"
        super().__init__(txt_db, img_db)

        txt2img = self.txt_db.txt2img
        self.txt2img = {id_: txt2img[id_] for id_ in self.ids}
        # images partitioned by rank
        self.img2txts = defaultdict(list)
        for id_, img in self.txt2img.items():
            self.img2txts[img].append(id_)
        self.img_name_list = list(self.img2txts.keys())

        assert neg_sample_size > 0
        self.neg_sample_size = neg_sample_size

    def __getitem__(self, i):
        gt_txt_id = self.ids[i]
        gt_img_fname = self.txt2img[gt_txt_id]

        id_pairs = [(gt_txt_id, gt_img_fname)]
        # sample negatives
        neg_sample_img_ids = sample_negative(
            self.img_name_list, [gt_img_fname], self.neg_sample_size)
        neg_sample_txt_ids = sample_negative(
            self.ids, self.img2txts[gt_img_fname], self.neg_sample_size)
        id_pairs.extend([(gt_txt_id, neg_img_id)
                         for neg_img_id in neg_sample_img_ids] +
                        [(neg_txt_id, gt_img_fname)
                         for neg_txt_id in neg_sample_txt_ids])
        inputs = self._collect_inputs(id_pairs)
        assert len(inputs) == (1 + 2 * self.neg_sample_size)
        return inputs

    def _collect_inputs(self, id_pairs):
        # create input features
        inputs = []
        for txt_id, img_id in id_pairs:
            example = self.txt_db[txt_id]
            # text input
            input_ids = example['input_ids']
            input_ids = self.txt_db.combine_inputs(input_ids)
            # img input
            img_feat, img_pos_feat, num_bb = self._get_img_feat(img_id)
            # mask
            attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

            inputs.append((input_ids, img_feat, img_pos_feat, attn_masks))

        return inputs


def itm_rank_collate(inputs):
    (input_ids, img_feats, img_pos_feats, attn_masks,
     ) = map(list, unzip(concat(i for i in inputs)))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    sample_size = len(inputs[0])
    assert all(sample_size == len(i) for i in inputs)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'sample_size': sample_size}
    return batch


class ItmRankDatasetHardNegFromText(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db, neg_sample_size=1):
        assert neg_sample_size > 0, "need at least 1 negative sample"
        super().__init__(txt_db, img_db)

        txt2img = self.txt_db.txt2img
        self.txt2img = {id_: txt2img[id_] for id_ in self.ids}
        self.img2txts = self.txt_db.img2txts
        self.img_name_list = list(self.img2txts.keys())
        self.neg_sample_size = neg_sample_size

    def __getitem__(self, i):
        gt_txt_id = self.ids[i]
        gt_img_fname = self.txt2img[gt_txt_id]

        input_ids = self.txt_db[gt_txt_id]['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)
        input_ids = input_ids.unsqueeze(0)
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                    ).unsqueeze(0)

        neg_img_ids = sample_negative(
            self.img_name_list, [gt_img_fname], self.neg_sample_size)
        img_ids = [gt_img_fname] + neg_img_ids
        # process image features (gt always first)
        img_feats, img_pos_feats, num_bbs = map(
            list, unzip(map(self._get_img_feat, img_ids)))
        img_feat = pad_tensors(img_feats, num_bbs)
        img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

        tl = input_ids.size(1)
        attn_masks = torch.zeros(len(img_ids), max(num_bbs) + tl).long()
        for i, nbb in enumerate(num_bbs):
            attn_masks.data[i, :tl + nbb].fill_(1)
        out_size = attn_masks.size(1)
        gather_index = get_gather_index([tl] * len(img_ids), num_bbs,
                                        len(img_ids), tl, out_size)

        batch = {'input_ids': input_ids,
                 'position_ids': position_ids,
                 'img_feat': img_feat,
                 'img_pos_feat': img_pos_feat,
                 'attn_masks': attn_masks,
                 'gather_index': gather_index}
        return batch


class ItmRankDatasetHardNegFromImage(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db, neg_sample_size=1):
        assert neg_sample_size > 0, "need at least 1 negative sample"
        super().__init__(txt_db, img_db)

        txt2img = self.txt_db.txt2img
        self.txt2img = {id_: txt2img[id_] for id_ in self.ids}
        self.img2txts = self.txt_db.img2txts
        self.txt_name_list = list(self.txt2img.keys())
        self.neg_sample_size = neg_sample_size

    def __getitem__(self, i):
        gt_txt_id = self.ids[i]
        gt_img_id = self.txt2img[gt_txt_id]
        gt_txt_ids = self.img2txts[gt_img_id]

        # process image features (gt always first)
        img_feat, img_pos_feat, nbb = self._get_img_feat(gt_img_id)
        img_feat = img_feat.unsqueeze(0)
        img_pos_feat = img_pos_feat.unsqueeze(0)

        # sample negative
        neg_txt_ids = sample_negative(
            self.txt_name_list, gt_txt_ids, self.neg_sample_size)
        txt_ids = [gt_txt_id] + neg_txt_ids

        # process text inputs
        all_inputs = []
        txt_lens = []
        for txt_id in txt_ids:
            input_ids = self.txt_db.combine_inputs(
                self.txt_db[txt_id]['input_ids'])
            all_inputs.append(input_ids)
            txt_lens.append(len(input_ids))
        input_ids = pad_sequence(all_inputs, batch_first=True, padding_value=0)
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                    ).unsqueeze(0)

        attn_masks = torch.zeros(len(txt_ids), max(txt_lens) + nbb).long()
        for i, tl in enumerate(txt_lens):
            attn_masks.data[i, :tl + nbb].fill_(1)
        out_size = attn_masks.size(1)
        gather_index = get_gather_index(txt_lens, [nbb] * len(txt_ids),
                                        len(txt_ids), tl, out_size)

        batch = {'input_ids': input_ids,
                 'position_ids': position_ids,
                 'img_feat': img_feat,
                 'img_pos_feat': img_pos_feat,
                 'attn_masks': attn_masks,
                 'gather_index': gather_index}
        return batch


def itm_rank_hn_collate(inputs):
    assert len(inputs) == 1
    return inputs[0]


class ItmValDataset(DetectFeatTxtTokDataset):
    """ For evaluating Image-Text-Retrieval task """

    def __init__(self, db_dir, img_dir, mini_batch_size=400):
        super().__init__(db_dir, img_dir)
        del self.lens
        self.txt2img = self.txt_db.txt2img
        self.img2txts = self.txt_db.img2txts
        self.all_img_ids = list(self.img2txts.keys())

        assert len(self.img2txts) >= mini_batch_size > 0
        self.bs = mini_batch_size

    def _get_batch_ids(self, i):
        gt_txt_id = self.ids[i]
        gt_img_id = self.txt2img[gt_txt_id]

        # sample fixed negatives for each gt image
        i = self.all_img_ids.index(gt_img_id)
        neg_st = i + 1
        neg_end = neg_st + self.bs - 1
        if neg_end > len(self.all_img_ids):
            # warp around
            neg_end -= len(self.all_img_ids)
            neg_img_ids = (self.all_img_ids[neg_st:]
                           + self.all_img_ids[:neg_end])
        else:
            neg_img_ids = self.all_img_ids[neg_st:neg_end]

        assert len(neg_img_ids) == (self.bs - 1), \
            "Did not sample enough neg samples"

        return gt_img_id, neg_img_ids

    def __getitem__(self, i):
        """ this returns list of mini-batches """
        gt_img_id, neg_img_ids = self._get_batch_ids(i)
        # NOTE 1st one is gt img
        batch = self.get_batch(i, [gt_img_id] + neg_img_ids)
        return batch

    def get_batch(self, i, img_ids):
        example = super().__getitem__(i)

        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)
        input_ids = input_ids.unsqueeze(0).expand(len(img_ids), -1).clone()
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                    ).unsqueeze(0)

        # process image features (gt always first)
        img_feats, img_pos_feats, num_bbs = map(
            list, unzip(map(self._get_img_feat, img_ids)))
        img_feat = pad_tensors(img_feats, num_bbs)
        img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

        tl = input_ids.size(1)
        attn_masks = torch.zeros(len(img_ids), max(num_bbs) + tl).long()
        for i, nbb in enumerate(num_bbs):
            attn_masks.data[i, :tl + nbb].fill_(1)
        out_size = attn_masks.size(1)
        gather_index = get_gather_index([tl] * len(img_ids), num_bbs,
                                        len(img_ids), tl, out_size)

        batch = {'input_ids': input_ids,
                 'position_ids': position_ids,
                 'img_feat': img_feat,
                 'img_pos_feat': img_pos_feat,
                 'attn_masks': attn_masks,
                 'gather_index': gather_index}
        return batch


def itm_val_collate(inputs):
    assert len(inputs) == 1, "input batch size > 1"
    return inputs[0]


class ItmEvalDataset(ItmValDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_img_ids = sorted(copy.deepcopy(self.all_img_ids),
                                  key=lambda i: self.img_db.name2nbb[i])

    def __getitem__(self, i):
        mini_batches = []
        for st in range(0, len(self.all_img_ids), self.bs):
            mini_batches.append(
                self.get_batch(i, self.all_img_ids[st:st + self.bs]))
        return mini_batches


itm_eval_collate = itm_val_collate
