## TripleE: Frustratingly Easy Domain Generalization

By [Xiaomeng Li](https://xmengli.github.io/), [Hongyu Ren](http://hyren.me/), Huifeng Yao and [Ziwei Liu](https://liuziwei7.github.io/).

This repository is for our paper [TripleE: Frustratingly Easy Domain Generalization]. 

<!-- <br/>
<p align="center">
  <img src="figure/framework.png">
</p>
 -->

## Installation

```
conda env create -f environment.yml
```
Note that the code is only tested in the above environment. 


## Data Preparation
* Download [PACS dataset](https://drive.google.com/drive/folders/1SKvzI8bCqW9bcoNLNCrTGbg7gBSw97qO), 
[Digits-DG dataset](https://drive.google.com/uc?id=15V7EsHfCcfbKgsDmzQKj_DfXt_XYp_P7), 
[Office-Home dataset](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw)
These three datasets are widely used in many DG papers as the benchmark.
  

* Put data under `./DATASET/`
* The correct path should be 
```
./DATASET/PACS/kfold/sketch/...
./DATASET/digits_dg/mnist/...
./DATASET/OfficeHome/Art/...  
```


## Train on PACS dataset
* Run the following command to get result on `art_painting` domain. 
* Change `times` to `1,2,3` to run the experiment for 3 times. 
```
python main_dg.py -a resnet18 --lr 0.001 --epochs 100 --batch-size 4  .  --source cartoon  sketch   photo --target art_painting   --ratio 1.0  --result exp/dg/dg_art  --baug 4 --gpu 0 --times 1  --ncesole
```
* Modify `source` and `target` to get results on other domains. 

* By averaging three results on each domain, you will get the following results: 

| Sketch    | Cartoon   | Art    |  Photo  | Average 
| ---------- | :-----------:  | :-----------: | :-----------: | :-----------:
| 83.65%    | 79.36%   | 85.72%     | 94.37% | 85.78% 

* Evaluate our models: download models from [cartoon](https://gohkust-my.sharepoint.com/:u:/g/personal/eexmli_ust_hk/EQUZU9JnCV5Hje30gTN29BkBNDiFYmBQXbDa1F2Gypn59g?e=j56u9G), [art_painting](https://gohkust-my.sharepoint.com/:u:/g/personal/eexmli_ust_hk/EXgBELtc0VFNpgDQnMeufxEB_EkAfTbF5-pvzQrSEfcNbA?e=CnRCKG), [photo](https://gohkust-my.sharepoint.com/:u:/g/personal/eexmli_ust_hk/Ef-B6X98bBNPtH3v2qKLVo0BKE7GlyNSkaWM0T91abOoLg?e=5o8fbG), [sketch](https://gohkust-my.sharepoint.com/:u:/g/personal/eexmli_ust_hk/ETlPaYXHTUdFkc82wkrL5y4BUs7Oom8exsrg-clk0zuyRg?e=ZvvDQU). 
Specify the path `--resume` for models and run 
```
python main_dg.py -a resnet18 --lr 0.001 --epochs 100 --batch-size 4  .  --source cartoon  sketch   photo --target art_painting   --ratio 1.0  --result exp/dg/dg_art  --baug 4 --gpu 0 --times 1  --ncesole  --evaluate --resume exp/model_best_art.pth.tar 
```


## Train on Digits-DG dataset
```
python main_dg_digits_IDCsub.py -a cnn --lr 0.01 --epochs 100 --batch-size 4  .  --source  svhn  syn  mnist --target mnist_m   --ratio 1.0  --result exp/dg/digits_mnistm_new  --baug 4 --gpu 3 --times 1  --ncesole

python main_dg_digits_IDCsub.py -a cnn --lr 0.01 --epochs 100 --batch-size 4  .  --source  mnist_m  syn  mnist --target svhn   --ratio 1.0  --result exp/dg/digits_svhn_test4  --baug 4 --gpu 4 --times 1  --ncesole

python main_dg_digits_IDCsub.py -a cnn --lr 0.01 --epochs 100 --batch-size 4  .  --source  svhn  mnist_m  mnist --target syn   --ratio 1.0  --result exp/dg/digits_syn_new  --baug 4 --gpu 7 --times 1  --ncesole

python main_dg_digits_IDCsub.py -a cnn --lr 0.01 --epochs 100 --batch-size 4  .  --source  svhn  syn  mnist_m --target mnist   --ratio 1.0  --result exp/dg/digits_mnist_new  --baug 4 --gpu 3 --times 1  --ncesole
```

## Evaluate on Digits-DG dataset
```
python main_dg_digits.py -a cnn --lr 0.01 --epochs 100 --batch-size 4  .  --source  svhn  syn  mnist --target mnist_m   --ratio 1.0  --result exp/dg/digits_mnistm_ensemble  --baug 4 --gpu 5 --times 1  --ncesole --evaluate --resume exp/dg/digits_mnist_ensemble/model_best.pth.tar

python main_dg_digits_base.py -a cnn --lr 0.01 --epochs 100 --batch-size 4  .  --source  mnist_m  syn  mnist --target svhn   --ratio 1.0  --result exp/dg/digits_svhn_cnsn_new  --baug 4 --gpu 5 --times 1  --ncesole --evaluate --resume exp/dg/digits_svhn_cnsn_new/model_best.pth.tar

python main_dg_digits.py -a cnn --lr 0.01 --epochs 100 --batch-size 4  .  --source  svhn  mnist_m  mnist --target syn   --ratio 1.0  --result exp/dg/digits_syn_ensemble  --baug 4 --gpu 5 --times 1  --ncesole --evaluate --resume exp/dg/digits_mnist_ensemble/model_best.pth.tar

python main_dg_digits.py -a cnn --lr 0.01 --epochs 100 --batch-size 4  .  --source  svhn  syn  mnist_m --target mnist   --ratio 1.0  --result exp/dg/digits_mnist_ensemble  --baug 4 --gpu 5 --times 1  --ncesole --evaluate --resume exp/dg/digits_mnist_ensemble/model_best.pth.tar
```

## Train on COVID-19 dataset 
```
python main_dg_covid.py -a resnet18 --lr 0.001 --epochs 100 --batch-size 4  .  --source Set1 Set2 Set3 --target Set4   --ratio 1.0  --result exp/dg/covid  --baug 4 --gpu 0 --times 1  --ncesole
``` 


## Note
* Contact: Xiaomeng Li (eexmli@ust.hk)

## Citation

If this code is useful for your research, please consider citing:

<!-- 
  ```shell
@article{li2020self,
  title={Self-supervised Feature Learning via Exploiting Multi-modal Data for Retinal Disease Diagnosis},
  author={Li, Xiaomeng and Jia, Mengyu and Islam, Md Tauhidul and Yu, Lequan and Xing, Lei},
  journal={IEEE Transactions on Medical Imaging},
  year={2020},
  publisher={IEEE}
}

  ``` -->