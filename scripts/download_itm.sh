# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

DOWNLOAD=$1

for FOLDER in 'img_db' 'txt_db' 'pretrained' 'finetune'; do
    if [ ! -d $DOWNLOAD/$FOLDER ] ; then
        mkdir -p $DOWNLOAD/$FOLDER
    fi
done

BLOB='https://acvrpublicycchen.blob.core.windows.net/uniter'

# image dbs
for SPLIT in 'train2014' 'val2014'; do
    if [ ! -d $DOWNLOAD/img_db/coco_$SPLIT ] ; then
        axel -n 64 $BLOB/img_db/coco_$SPLIT.tar -o $DOWNLOAD/img_db/
        tar -xvf $DOWNLOAD/img_db/coco_$SPLIT.tar -C $DOWNLOAD/img_db
    fi
done
if [ ! -d $DOWNLOAD/img_db/flickr30k ] ; then
    axel -n 64 $BLOB/img_db/flickr30k.tar -o $DOWNLOAD/img_db/
    tar -xvf $DOWNLOAD/img_db/flickr30k.tar -C $DOWNLOAD/img_db
fi

# text dbs
for SPLIT in 'train' 'restval' 'val' 'test'; do
    axel -n 64 $BLOB/txt_db/itm_coco_$SPLIT.db.tar -o $DOWNLOAD/txt_db/
    tar -xvf $DOWNLOAD/txt_db/itm_coco_$SPLIT.db.tar -C $DOWNLOAD/txt_db
done
for SPLIT in 'train' 'val' 'test'; do
    axel -n 64 $BLOB/txt_db/itm_flickr30k_$SPLIT.db.tar -o $DOWNLOAD/txt_db/
    tar -xvf $DOWNLOAD/txt_db/itm_flickr30k_$SPLIT.db.tar -C $DOWNLOAD/txt_db
done

if [ ! -f $DOWNLOAD/pretrained/uniter-base.pt ] ; then
    axel -n 64 $BLOB/pretrained/uniter-base.pt -o $DOWNLOAD/pretrained/
fi

