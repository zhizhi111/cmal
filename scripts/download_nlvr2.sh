# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

DOWNLOAD=$1

for FOLDER in 'ann' 'img_db' 'txt_db' 'pretrained' 'finetune'; do
    if [ ! -d $DOWNLOAD/$FOLDER ] ; then
        mkdir -o $DOWNLOAD/$FOLDER
    fi
done

BLOB='https://acvrpublicycchen.blob.core.windows.net/uniter'

# annotations
NLVR='https://raw.githubusercontent.com/lil-lab/nlvr/master/nlvr2/data'
axel -n 64 $NLVR/dev.json -o $DOWNLOAD/ann/
axel -n 64 $NLVR/test1.json -o $DOWNLOAD/ann/

# image dbs
for SPLIT in 'train' 'dev' 'test'; do
    axel -n 64 $BLOB/img_db/nlvr2_$SPLIT.tar -o $DOWNLOAD/img_db/
    tar -xvf $DOWNLOAD/img_db/nlvr2_$SPLIT.tar -C $DOWNLOAD/img_db
done

# text dbs
for SPLIT in 'train' 'dev' 'test1'; do
    axel -n 64 $BLOB/txt_db/nlvr2_$SPLIT.db.tar -o $DOWNLOAD/txt_db/
    tar -xvf $DOWNLOAD/txt_db/nlvr2_$SPLIT.db.tar -C $DOWNLOAD/txt_db
done

if [ ! -f $DOWNLOAD/pretrained/uniter-base.pt ] ; then
    axel -n 64 $BLOB/pretrained/uniter-base.pt -o $DOWNLOAD/pretrained/
fi

axel -n 64 $BLOB/finetune/nlvr-base.tar -o $DOWNLOAD/finetune/
tar -xvf $DOWNLOAD/finetune/nlvr-base.tar -C $DOWNLOAD/finetune
