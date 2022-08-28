

# Baseline
net='resnet18'
lr=0.001
epochs=100
batch_size=16
ratio=1.0
baug=1
times=1
expname='office_Product_baseline_v2_ours'
path='exp/dg/'
result=$path$expname
CUDA_VISIBLE_DEVICES='1' python main_dg_baseline.py -a $net --lr $lr --epochs $epochs \
--batch-size $batch_size  .  --source Real_World Clipart  Art --target Product \
--ratio $ratio  --result $result  --baug $baug --expname $expname \
--times $times


## EReplayB
net='resnet18'
lr=0.001
epochs=100
batch_size=32
ratio=1.0
baug=4
times=1
expname='office_Product_baseline_v2_ours_EReplayB32'
path='exp/dg/'
result=$path$expname
CUDA_VISIBLE_DEVICES='1' python main_dg_baseline.py -a $net --lr $lr --epochs $epochs \
--batch-size $batch_size  .  --source Real_World Clipart Art --target Product \
--ratio $ratio  --result $result  --baug $baug --expname $expname \
--times $times


## EReplayB+ESAUGF
net='resnet18'
lr=0.001
epochs=100
batch_size=32
ratio=1.0
baug=4
times=1
expname='office_Art_baseline_v2_ours_EReplayB32_ESAugF2'
path='exp/dg/'
result=$path$expname
CUDA_VISIBLE_DEVICES='6' python main_dg_baseline.py -a $net --lr $lr --epochs $epochs \
--batch-size $batch_size  .  --target Art \
--ratio $ratio  --result $result  --baug $baug --expname $expname \
--times $times --ESAugF  --classes 65


## EReplayB+ESAUGF --PACS
net='resnet18'
lr=0.001
epochs=100
batch_size=4
ratio=1.0
baug=4
times=1
expname='PACS_cartoon_EReplayB_ESAugF2'
path='exp/dg/'
result=$path$expname
CUDA_VISIBLE_DEVICES='4' python main_dg_baseline.py -a $net --lr $lr --epochs $epochs \
--batch-size $batch_size  .  --target cartoon \
--ratio $ratio  --result $result  --baug $baug --expname $expname \
--times $times  --classes 7  --ESAugF




## ALL
net='resnet50'
lr=0.001
epochs=100
batch_size=16
ratio=1.0
baug=4
times=1
expname='office_Real_Res50'
path='exp/dg/'
result=$path$expname
CUDA_VISIBLE_DEVICES='1' python main_dg_threemodels.py -a $net --lr $lr --epochs $epochs \
--batch-size $batch_size  .  --target Real_World \
--ratio $ratio  --result $result  --baug $baug --expname $expname \
--times $times  --ESAugF  --classes 65


### All on PACS
net='resnet18'
lr=0.001
epochs=100
batch_size=4
ratio=1.0
baug=4
times=1
expname='PACS_photo_TripleE_F4'
path='exp/dg/'
result=$path$expname
CUDA_VISIBLE_DEVICES='7' python main_dg_threemodels.py -a $net --lr $lr --epochs $epochs \
--batch-size $batch_size .  --target photo  \
--ratio $ratio  --result $result  --baug $baug --expname $expname \
--times $times  --ESAugF  --classes 7 \
 --evaluate --resume_1 exp/dg/PACS_art_TripleE_F/sub_model_1_best.pth.tar \
--resume_2 exp/dg/PACS_art_TripleE_F/sub_model_2_best.pth.tar \
--resume_3 exp/dg/PACS_art_TripleE_F/sub_model_3_best.pth.tar


## ALL on DomainBed
net='resnet50'
lr=0.001
epochs=100
batch_size=16
ratio=1.0
baug=4
times=1
expname='DomainBed_all_clipart_res50'
path='exp/dg/'
result=$path$expname
CUDA_VISIBLE_DEVICES='0' python main_dg_threemodels.py -a $net --lr $lr --epochs $epochs \
--batch-size $batch_size  .  --target clipart \
--ratio $ratio  --result $result  --baug $baug --expname $expname \
--times $times  --ESAugF  --classes 345

## ALL on poverty
net='resnet50'
lr=0.001
epochs=100
batch_size=8
ratio=1.0
baug=4
times=1
expname='poverty_all_res50'
path='exp/dg/'
result=$path$expname
CUDA_VISIBLE_DEVICES='3' python main_dg_wilds.py -a $net --lr $lr --epochs $epochs \
--batch-size $batch_size  .  --target clipart \
--ratio $ratio  --result $result  --baug $baug --expname $expname \
--times $times  --ESAugF  --classes 2