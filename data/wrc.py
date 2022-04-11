"""
wrc dataset
"""

import numpy as np
import torch
from toolz.sandbox import unzip
from torch.nn.utils.rnn import pad_sequence

from .data import (DetectFeatTxtTokDataset, DetectFeatLmdb, TxtTokLmdb,
                   pad_tensors, get_gather_index)
from .word_region_util import obj2bert


# KevinHwang@220306
def _get_word_region_map(input_ids, img_soft_label):
    # 使用 argmax 取得每一个 region 的实体标签序号
    # background class should not be the target
    argmax_soft_labels = torch.argmax(img_soft_label[:, 1:-1], dim=1).tolist()
    # 将实体标签序号转化为 token（每一个实体标签序号可能对应多个 token）
    converted_argmax = [obj2bert[obj_label] if obj_label in obj2bert.keys() else [] for obj_label in
                        argmax_soft_labels]
    obj_tokens = list(set(tk for tokens in converted_argmax for tk in tokens))  # 获取所有的实体 token 并去重
    word_region_map = {}  # 获取 word 和 region 的对应关系，key是 word 的位置索引， region 是区域的索引列表
    if len(obj_tokens) > 0:
        # 遍历所有的实体 token
        for tar_token in obj_tokens:
            for i, tk in enumerate(input_ids):
                if tk == tar_token:
                    tar_region_idxs = []
                    for j, rg in enumerate(converted_argmax):
                        if tar_token in rg:
                            tar_region_idxs.append(j)
                    # 考虑到后续需要加 [CLS] ，这里是 i+1 而不是 i
                    word_region_map[i + 1] = tar_region_idxs

    return word_region_map


class WrcDataset(DetectFeatTxtTokDataset):
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

        word_region_map = _get_word_region_map(input_ids, img_soft_label)

        # transfer input_ids to tensor from list
        input_ids = torch.tensor([self.txt_db.cls_] + input_ids + [self.txt_db.sep])

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

        return input_ids, img_feat, img_pos_feat, word_region_map, attn_masks


def wrc_collate(inputs):
    (input_ids, img_feats, img_pos_feats, word_region_maps, attn_masks) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

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
             'word_region_maps': word_region_maps,
             'attn_masks': attn_masks,
             'gather_index': gather_index}
    return batch
