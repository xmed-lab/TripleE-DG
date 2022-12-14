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
CUDA_VISIBLE_DEVICES='0' python main_dg_threemodels.py -a $net --lr $lr --epochs $epochs --batch-size $batch_size .  --target photo  --ratio $ratio  --result $result  --baug $baug --expname $expname  --times $times  --ESAugF  --classes 7