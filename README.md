# DSACA
Code release for Dilated-Scale-Aware Category-Attention ConvNet for Multi-Class Object Counting (under review)

![Image text](https://github.com/PRIS-CV/DSACA/blob/main/images/intro.png)

## Changelog
- 2021/04/21 upload the code.

## Requirements
- python 3.6
- PyTorch 1.3.0
- torchvision

## Pre trained model
`VisDrone_class8.pth` and `RSOC_class2.pth` download.
- VisDrone model best (password:qsw6) [Link](https://pan.baidu.com/s/1nORmkUbV1c-5MLZvYKToiA).
- RSOC model best (password:t15a) [Link](https://pan.baidu.com/s/1GjwsWG6FXu6D0DRGumRQ5Q).

## Data
- Download datasets  
- Extract them to `dataset/RSOC/` and `dataset/VisDrone/`, respectively.
* e.g., RSOC (modified from the DOTA dataset) and VisDrone datasets
```
  -/DSACA-main
      -/DSACA-main/dataset
         └─── VisDrone
            └───train_data
            └───test_data
         └─── RSOC
            └───train
            └───val
      -/DSACA-main/pre_trained
         └─── VisDrone_class8.pth
         └─── RSOC_class2.pth
         └─── pre_trained.md
      -/DSACA-main/density_generate
         └─── VisDrone.py
         └─── RSOC.py
      -/DSACA-main/make_npydata
         └─── VisDrone_make_npydata.py
         └─── RSOC_make_npydata.py
      -/DSACA-main/Network
         └─── VisDrone_class8.py
         └─── baseline_DSAM_CAM.py
      -/DSACA-main/images
         └─── intro.png
      └─── config.py
      └─── dataset.py
      └─── image.py
      └─── utils.py
      └─── VisDrone_train_class8_CAM_DSAM.py
      └─── RSOC_train_class2_CAM_DSAM.py
      └─── README.md
```

# Train & Test
- Cd ` density_generate`  then run `VisDrone.py` and `RSOC.py` for dataset pre-processing.
- Cd `make_npydata`  then run `VisDrone.py` and `RSOC.py` for target path pre-saving.
- Edit `config.py` for training-parameters seting.
- Run `RSOC_train_class8_CAM_DSAM.py` or `VisDrone_train_class2_CAM_DSAM.py` for training & testing.

## Contact
Thanks for your attention!
If you have any suggestion or question, you can leave a message here or contact us directly:
- xuwei2020@bupt.edu.cn
- mazhanyu@bupt.edu.cn
