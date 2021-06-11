#!/bin/sh

device="$1"
seed=0
data="$2"
net="$3"
loss="$4"

case "${data}" in
imagenet* ) datadir='/home/takumi/tmp/nvme/ilsvrc12/imagenet12/images/' ;;
"inat2018" ) datadir='/home/takumi/tmp/nvme/iNaturalist2018/' ;;
"inat2019" ) datadir='/home/takumi/tmp/nvme/iNaturalist2019/' ;;
esac

CUDA_VISIBLE_DEVICES=$device python main_train.py --dataset ${data}  --data ${datadir} --net-config ${net}  --loss-config ${loss} --out-dir ./results/${data}/${net}/${loss}/ --workers 12 --seed ${seed}

CUDA_VISIBLE_DEVICES=$device python main_finetune.py --dataset ${data}  --data ${datadir} --net-config ${net}_finetune  --loss-config CosLoss --model-file ./results/${data}/${net}/${loss}/model_best.pth.tar --out-dir ./results/${data}/${net}/${loss}/finetune/ --workers 12 --seed ${seed} 