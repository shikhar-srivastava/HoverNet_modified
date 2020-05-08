# Submission to MoNuSAC Post-Challenge 2020

### Hover-Net adapted to the Multi-Task, Transfer-Learning setting

To obtain instance-masks from the trained model on the MoNuSAC test data in the required challenge format:

- Ensure library requirements mentioned in requirements.txt are installed
- Download `MoNuSAC test-data` & trained model
- Modify `data_dir`, `output-dir`, `model-path` to reflect MoNuSAC test data, desired output path, & downloaded trained model path (.index file) respectively. 

Run the below script as:
```
python test_script.py --data_dir=<Path to MoNuSAC Test Dir> --output_dir='' --model_path=< path to model file (.index)> --img_ext='.tif' --gpu=''

```


## Getting Started

Install the required libraries before using this code. Please refer to `requirements.txt`



## Dataset
Download the datasets & modify the paths in `config_multitask`:
- [CoNSeP](https://warwick.ac.uk/fac/sci/dcs/research/tia/data/hovernet/)
- [Kumar,CPM-17](https://drive.google.com/open?id=1l55cv3DuY-f7-JotDN7N5nbNnjbLWchK)

Ground truth files are in `.mat` format, refer to the README included with the datasets for further information.

### Repository 

Files needed to modify multi-task behavior:
- `test_script.py :` 
- `config_multitask :` 
- `train_multitask :` 
- `config_multitask :` 


### Global Repository Structure 
For details on the general repository structure, refer to `hover_net-README.md` 

HoVer-Net Paper linked [here](https://arxiv.org/abs/1812.06499)
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