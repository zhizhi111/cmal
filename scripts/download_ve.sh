# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

DOWNLOAD=$1

for FOLDER in 'img_db' 'txt_db' 'pretrained' 'finetune'; do
    if [ ! -d $DOWNLOAD/$FOLDER ] ; then
        mkdir -p $DOWNLOAD/$FOLDER
    fi
done

BLOB='https://acvrpublicycchen.blob.core.windows.net/uniter'

# image db
if [ ! -d $DOWNLOAD/img_db/flickr30k ] ; then
    axel -n 64 $BLOB/img_db/flickr30k.tar -o $DOWNLOAD/img_db/
    tar -xvf $DOWNLOAD/img_db/flickr30k.tar -C $DOWNLOAD/img_db
fi

# text dbs
for SPLIT in 'train' 'dev' 'test'; do
    axel -n 64 $BLOB/txt_db/ve_$SPLIT.db.tar -o $DOWNLOAD/txt_db/
    tar -xvf $DOWNLOAD/txt_db/ve_$SPLIT.db.tar -C $DOWNLOAD/txt_db
done

if [ ! -f $DOWNLOAD/pretrained/uniter-base.pt ] ; then
    axel -n 64 $BLOB/pretrained/uniter-base.pt -o $DOWNLOAD/pretrained/
fi

