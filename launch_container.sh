# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#TXT_DB=$1
#IMG_DIR=$2
#OUTPUT=$3
#PRETRAIN_DIR=$4

# KevinHwang:no input parameters
TXT_DB=/home/hky/processed_data_and_pretrained_models/txt_db
IMG_DIR=/home/hky/processed_data_and_pretrained_models/img_db
OUTPUT=/home/hky/processed_data_and_pretrained_models/finetune
PRETRAIN_DIR=/home/hky/processed_data_and_pretrained_models/pretrained
ANN_DIR=/home/hky/processed_data_and_pretrained_models/ann

if [ -z $CUDA_VISIBLE_DEVICES ]; then
  CUDA_VISIBLE_DEVICES='all'
fi

docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --rm -it \
  --mount src=$(pwd),dst=/src,type=bind \
  --mount src=$OUTPUT,dst=/storage,type=bind \
  --mount src=$PRETRAIN_DIR,dst=/pretrain,type=bind,readonly \
  --mount src=$TXT_DB,dst=/txt,type=bind,readonly \
  --mount src=$IMG_DIR,dst=/img,type=bind,readonly \
  --mount src=$ANN_DIR,dst=/ann,type=bind,readonly \
  -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
  -v /etc/localtime:/etc/localtime \
  -w /src hky/uniter:v1
