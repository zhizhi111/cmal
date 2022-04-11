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
if [ ! -d $DOWNLOAD/img_db/re_coco_gt ] ; then
    axel -n 64 $BLOB/img_db/re_coco_gt.tar -o $DOWNLOAD/img_db/
    tar -xvf $DOWNLOAD/img_db/re_coco_gt.tar -C $DOWNLOAD/img_db
fi
if [ ! -d $DOWNLOAD/img_db/re_coco_det ] ; then
    axel -n 64 $BLOB/img_db/re_coco_det.tar -o $DOWNLOAD/img_db/
    tar -xvf $DOWNLOAD/img_db/re_coco_det.tar -C $DOWNLOAD/img_db
fi

# text dbs
axel -n 64 $BLOB/txt_db/re_txt_db.tar -o $DOWNLOAD/txt_db/
tar -xvf $DOWNLOAD/txt_db/re_txt_db.tar -C $DOWNLOAD/txt_db/

if [ ! -f $DOWNLOAD/pretrained/uniter-base.pt ] ; then
    axel -n 64 $BLOB/pretrained/uniter-base.pt -o $DOWNLOAD/pretrained/
fi

