# DSACA
Code release for Dilated-Scale-Aware Category-Attention ConvNet for Multi-Class Object Counting (under review)

![Image text](https://github.com/PRIS-CV/DSACA/blob/main/images/intro.png)

## Changelog
- 2021/04/21 upload the code.

## Requirements
- python 3.6
- PyTorch 1.3.0
- torchvision

## Data
- Download datasets
- Extract them to `dataset/RSOC/` and `dataset/VisDrone/`, respectively.
* e.g., RSOC and VisDrone datasets
```
  -/DSACA-main
      -/DSACA-main/dataset
         └─── RSOC
         └─── VisDrone
      -/DSACA-main/pre_train
         └─── RSOC_class2.pth
         └─── VisDrone_class8.pth
      -/DSACA-main/density_generate
         └─── RSOC.py
         └─── VisDrone_class8.py
         └─── ...
      -/DSACA-main/make_npydata
         └─── RSOC_make_npydata.py
         └─── VisDrone_make_npydata.py
      -/DSACA-main/Network
         └─── baseline_DSAM_CAM.py
         └─── VisDrone_class8.py
      └─── config.py
      └─── dataset.py
      └─── image.py
      └─── utils.py
      └─── RSOC_train_class2_CAM_DSAM.py
      └─── VisDrone_train_class8_CAM_DSAM.py
      └─── README.md
 ```
 
# DSACA
- run `density_generate/RSOC.py` and `density_generate/VisDrone_class8.py` for dataset pre-processing.
- run `make_npydata/RSOC.py` and `make_npydata/VisDrone_class8.py` for target path pre-saving.
- Edit `config.py` for training-parameters seting.
- run `RSOC_train_class2_CAM_DSAM.py` or `VisDrone_train_class8_CAM_DSAM.py` for training & testing.

## Contact
Thanks for your attention!
If you have any suggestion or question, you can leave a message here or contact us directly:
- xuwei2020@bupt.edu.cn
- mazhanyu@bupt.edu.cn
