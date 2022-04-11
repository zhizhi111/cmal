import argparse
import lmdb
import msgpack
from lz4.frame import decompress
import os
from os.path import exists

import torch
from horovod import torch as hvd

from model.pretrain import UniterForPretraining
from utils.const import IMG_DIM, IMG_LABEL_DIM
from utils.logger import LOGGER
from utils.misc import parse_with_config, set_random_seed


def get_input_ids():
    txt_env = lmdb.open("/txt/pretrain_coco_train.db", readonly=True, create=False)
    txt_txn = txt_env.begin()
    i = 0
    # for key, value in txt_txn.cursor():
    key = b'166354'
    value = txt_txn.get(key)
    txt_dump = msgpack.loads(decompress(value), raw=False)
    print('key: ', key)
    print('txt_dump: ', txt_dump)

    txt_env.close()

    input_ids = torch.tensor(txt_dump['input_ids'])

    return input_ids


def main(opts):
    hvd.init()
    n_gpu = hvd.size()
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    opts.rank = rank
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
        device, n_gpu, hvd.rank(), opts.fp16))

    if opts.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, "
                         "should be >= 1".format(
            opts.gradient_accumulation_steps))

    set_random_seed(opts.seed)

    # Prepare model
    if opts.checkpoint:
        checkpoint = torch.load(opts.checkpoint)
    else:
        checkpoint = {}
    model = UniterForPretraining.from_pretrained(
        opts.model_config, checkpoint,
        img_dim=IMG_DIM, img_label_dim=IMG_LABEL_DIM)
    model.to(device)

    input_ids = get_input_ids()
    cls = torch.tensor([101])
    sep = torch.tensor([102])
    input_ids = torch.cat((cls, input_ids), 0)
    input_ids = torch.cat((input_ids, sep), 0)
    input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(device)
    print('input_ids: ', input_ids)

    position_ids = torch.arange(0, input_ids.size(0), dtype=torch.long)
    position_ids = position_ids.unsqueeze(0).to(device)
    print('position_ids: ', position_ids)

    attention_mask = torch.ones_like(input_ids)
    print('attention_mask: ', attention_mask)

    sequence_output = model.uniter(input_ids=input_ids, position_ids=position_ids, img_feat=None, img_pos_feat=None,
                                   attention_mask=attention_mask, output_all_encoded_layers=False)
    # print('sequence_output: ', sequence_output)
    print('size of sequence_output\'items: ', sequence_output.size())

    pooled_output = model.uniter.pooler(sequence_output)
    # print('pooled_output: ', pooled_output)
    print('size of pooled_output\'items: ', pooled_output.size())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    # NOTE: train tasks and val tasks cannot take command line arguments
    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')

    parser.add_argument("--model_config", type=str,
                        help="path to model structure config json")
    parser.add_argument("--checkpoint", default=None, type=str,
                        help="path to model checkpoint (*.pt)")

    parser.add_argument(
        "--output_dir", default=None, type=str,
        help="The output directory where the model checkpoints will be "
             "written.")

    parser.add_argument('--mrm_prob', default=0.15, type=float,
                        help='probability to mask in MRM training')
    parser.add_argument('--itm_neg_prob', default=0.5, type=float,
                        help='probability to make negative examples'
                             'in ITM training')
    parser.add_argument('--itm_ot_lambda', default=0.0, type=float,
                        help='weight of OT (optimal transport) loss (WRA)')

    # Prepro parameters
    parser.add_argument('--max_txt_len', type=int, default=60,
                        help='max number of tokens in text (BERT BPE)')
    parser.add_argument('--conf_th', type=float, default=0.2,
                        help='threshold for dynamic bounding boxes '
                             '(-1 for fixed)')
    parser.add_argument('--max_bb', type=int, default=100,
                        help='max number of bounding boxes')
    parser.add_argument('--min_bb', type=int, default=10,
                        help='min number of bounding boxes')
    parser.add_argument('--num_bb', type=int, default=36,
                        help='static number of bounding boxes')

    # training parameters
    parser.add_argument("--train_batch_size", default=4096, type=int,
                        help="Total batch size for training. "
                             "(batch by tokens)")
    parser.add_argument("--val_batch_size", default=4096, type=int,
                        help="Total batch size for validation. "
                             "(batch by tokens)")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                        help="Number of updates steps to accumualte before "
                             "performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--valid_steps", default=1000, type=int,
                        help="Run validation every X steps")
    parser.add_argument("--num_train_steps", default=100000, type=int,
                        help="Total number of training updates to perform.")
    parser.add_argument("--optim", default='adamw',
                        choices=['adam', 'adamax', 'adamw'],
                        help="optimizer")
    parser.add_argument("--betas", default=[0.9, 0.98], nargs='+',
                        help="beta for adam optimizer")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="tune dropout regularization")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="weight decay (L2) regularization")
    parser.add_argument("--grad_norm", default=2.0, type=float,
                        help="gradient clipping (-1 for no clipping)")
    parser.add_argument("--warmup_steps", default=10000, type=int,
                        help="Number of training steps to perform linear "
                             "learning rate warmup for.")

    # device parameters
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true', help="pin memory")

    # can use config files
    parser.add_argument('--config', required=True, help='JSON config files')

    args = parse_with_config(parser)

    if exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not "
                         "empty.".format(args.output_dir))

    # options safe guard
    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512

    main(args)
