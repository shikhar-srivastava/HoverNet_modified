# 2nd Place Submission to ISBI 2020's MoNuSAC Post-Challenge
### Hover-Net adapted to the Multi-Task, Transfer-Learning setting

To obtain instance-masks from the trained model on the MoNuSAC test data in the required challenge format:

- Ensure library requirements mentioned in requirements.txt are installed
- Download `MoNuSAC test-data` & trained model
- Modify `data_dir`, `output-dir`, `model-path` to reflect MoNuSAC test data, desired output path, & downloaded trained model path (.index file) respectively. 

Run the below script as:
```
python test_script.py --data_dir --output_dir --model_path --img_ext --gpu

```


## Getting Started

Install the required libraries before using this code. Please refer to `requirements.txt`



## Dataset
Download the datasets & modify the paths in `config_multitask`:
- [CoNSeP](https://warwick.ac.uk/fac/sci/dcs/research/tia/data/hovernet/)
- [Kumar & CPM-17](https://drive.google.com/open?id=1l55cv3DuY-f7-JotDN7N5nbNnjbLWchK)

Ground truth files are in `.mat` format, refer to the README included with the datasets for further information.

### Repository 

Files needed to modify multi-task behavior:
- [test_script](./src/test_script.py) : Test & Eval script
- [config_multitask](./src/config_multitask.py) : Config
- [train_multitask](./src/train_multitask.py) : Training Script
- [hover_multitask](./src/opt/hover_multitask.py) : Network & Training protocol

### Global Repository Structure 

For details on the general repository structure, refer to `hover_net-README.md` 

### Citations

MoNuSAC 2020

```
@article{monusac2020,
author = {Verma, Ruchika; Kumar, Neeraj; Patil, Abhijeet; Kurian, Nikhil; Rane, Swapnil; and Sethi, Amit},
year = {2020},
month = {02},
pages = {},
language = {en},
title = {Multi-organ Nuclei Segmentation and Classification Challenge 2020},
publisher = {Unpublished},
doi = {10.13140/RG.2.2.12290.02244/1},
 url = {http://rgdoi.net/10.13140/RG.2.2.12290.02244/1}
}
```

HoVer-Net Paper [Linked [here](https://arxiv.org/abs/1812.06499)]
```
@article{graham2019hover,
  title={Hover-net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images},
  author={Graham, Simon and Vu, Quoc Dang and Raza, Shan E Ahmed and Azam, Ayesha and Tsang, Yee Wah and Kwak, Jin Tae and Rajpoot, Nasir},
  journal={Medical Image Analysis},
  pages={101563},
  year={2019},
  publisher={Elsevier}
}
```
