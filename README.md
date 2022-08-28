## TripleE: Easy Domain Generalization via Episodic Replay

By [Xiaomeng Li](https://xmengli.github.io/), [Hongyu Ren](http://hyren.me/), [Huifeng Yao](https://scholar.google.com/citations?hl=en&user=hzNA39EAAAAJ) and [Ziwei Liu](https://liuziwei7.github.io/).

This repository is for our paper [TripleE: Easy Domain Generalization via Episodic Replay]. 

<!-- <br/>
<p align="center">
  <img src="figure/framework.png">
</p>
 -->

## Installation

```
conda env create -f IDC_environment.yml
```


## Data Preparation
* Download [PACS dataset](https://drive.google.com/drive/folders/1SKvzI8bCqW9bcoNLNCrTGbg7gBSw97qO), 
[Digits-DG dataset](https://drive.google.com/uc?id=15V7EsHfCcfbKgsDmzQKj_DfXt_XYp_P7), 
[Office-Home dataset](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw)
[VCLS dataset]
[XXX dataset]
[XXX dataset]


* Put data under `./DATASET/`
* The correct path should be 
```
./DATASET/PACS/kfold/sketch/...
./DATASET/digits_dg/mnist/...
./DATASET/OfficeHome/Art/...  
```


## Train on PACS dataset
* Check training command in `./scripts/`. 
* Change `times` to `1,2,3` to run the experiment for 3 times. 
* Set `target` to get results on different target domains.


* By averaging three results on each domain, you will get the following results: 

| Sketch    | Cartoon   | Art    |  Photo  | Average 
| ---------- | :-----------:  | :-----------: | :-----------: | :-----------:
| 84.45%    | 80.84%   | 85.30%     | 96.27% | 86.72% 

<!-- * Evaluate our models: download models from [cartoon](https://gohkust-my.sharepoint.com/:u:/g/personal/eexmli_ust_hk/EQUZU9JnCV5Hje30gTN29BkBNDiFYmBQXbDa1F2Gypn59g?e=j56u9G), [art_painting](https://gohkust-my.sharepoint.com/:u:/g/personal/eexmli_ust_hk/EXgBELtc0VFNpgDQnMeufxEB_EkAfTbF5-pvzQrSEfcNbA?e=CnRCKG), [photo](https://gohkust-my.sharepoint.com/:u:/g/personal/eexmli_ust_hk/Ef-B6X98bBNPtH3v2qKLVo0BKE7GlyNSkaWM0T91abOoLg?e=5o8fbG), [sketch](https://gohkust-my.sharepoint.com/:u:/g/personal/eexmli_ust_hk/ETlPaYXHTUdFkc82wkrL5y4BUs7Oom8exsrg-clk0zuyRg?e=ZvvDQU). 
Specify the path `--resume` for models and run 
```
python main_dg.py -a resnet18 --lr 0.001 --epochs 100 --batch-size 4  .  --source cartoon  sketch   photo --target art_painting   --ratio 1.0  --result exp/dg/dg_art  --baug 4 --gpu 0 --times 1  --ncesole  --evaluate --resume exp/model_best_art.pth.tar 
```


 -->
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