# UADet

**<center><font size=6>UADet: A Remarkably Simple Yet Effective Uncertainty-Aware Open-Set Object Detection Framework</font></center>** 

Silin Cheng<sup>1</sup>, Yuanpei Liu<sup>2</sup>, Kai Han<sup>1</sup>  
<sup>1</sup> <sub>The University of Hong Kong</sub>

[![Paper](https://img.shields.io/badge/arXiv-2412.10028-brightgreen)](https://arxiv.org/abs/2412.09229)
[![Project](https://img.shields.io/badge/Project-red)](https://visual-ai.github.io/UADet/)
<a href="mailto: hnchengsilin@gmail.com">
        <img alt="emal" src="https://img.shields.io/badge/contact_me-email-yellow">
    </a>

### Setup

The code is based on [opendet2](https://github.com/csuhan/opendet2). 

* **Installation** 

Here is a from-scratch setup script.

```
conda create -n UADet python=3.9 -y
conda activate UADet

pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

git clone https://github.com/facebookresearch/detectron2.git
git checkout d1e0456
python -m pip install -e detectron2

git clone https://github.com/Visual-AI/UADet.git
cd UADet
pip install -v -e .
```

* **Prepare datasets** 

Please follow [datasets/README.md](datasets/README.md) for dataset preparation. Then we generate VOC-COCO datasets.

```
bash datasets/opendet2_utils/prepare_openset_voc_coco.sh
# using data splits provided by us.
cp datasets/voc_coco_ann datasets/voc_coco -rf
```

### Model Zoo

We report the results on VOC and VOC-COCO-20, and provide pretrained models. Please refer to the corresponding log file for full results.

* **Faster R-CNN**


| Method  | backbone | mAP<sub>&uarr;</sub>(VOC) | WI<sub>&darr;</sub> | AOSE<sub>&darr;</sub> | mAP<sub>&uarr;</sub> | U-AP<sub>&uarr;</sub> | U-Recall<sub>&uarr;</sub> | Download |
|---------|:--------:|:-------------------------:|:-------------------:|:---------------------:|:--------------------:|:---------------------:|:------------------------:|:--------:|
| FR-CNN  | R-50     | 80.06                     | 19.50               | 16518                 | 58.36                | 0                     | 0                        | [config](configs/faster_rcnn_R_50_FPN_3x_baseline.yaml) [model](https://drive.google.com/drive/folders/10uFOLLCK4N8te08-C-olRyDV-cJ-L6lU?usp=sharing) |
| PROSER  | R-50     | 79.42                     | 20.44               | 14266                 | 56.72                | 16.99                 | 37.34                    | [config](configs/faster_rcnn_R_50_FPN_3x_proser.yaml) [model](https://drive.google.com/drive/folders/1_L85gisyvDtBXPe2UbI49vrd5FoBIOI_?usp=sharing) |
| ORE     | R-50     | 79.80                     | 18.18               | 12811                 | 58.25                | 2.60                  | -                        | [config]() [model]() |
| DS      | R-50     | 79.70                     | 16.76               | 13062                 | 58.46                | 8.75                  | 19.80                    | [config](configs/faster_rcnn_R_50_FPN_3x_ds.yaml) [model](https://drive.google.com/drive/folders/1OWDjL29E2H-_lSApXqM2r8PS7ZvUNtiv?usp=sharing) |
| OpenDet | R-50     | 80.02                     | 12.50               | 10758                 | 58.64                | 14.38                 | 37.65                    | [config](configs/faster_rcnn_R_50_FPN_3x_opendet.yaml) [model](https://drive.google.com/drive/folders/1fzD0iJ6lJrPL4ffByeO9M-udckbYqIxY?usp=sharing) |
| UAdet   | R-50     | 80.13                     | 13.19               | 10186                 | 59.12                | 15.09                 | 59.03                    | [config](configs/faster_rcnn_R_50_FPN_3x_uadet.yaml) [model](https://drive.google.com/drive/folders/uadet_models) |

### Train and Test

* **Testing**

First, you need to download pretrained weights in the model zoo, e.g., [UADet](https://drive.google.com/file/d/1UKsSpo6gfM4NwnbGET60RkkjF_L5mwzN/view?usp=sharing).

Then, run the following command:
```
python tools/train_net.py --dist-url auto --num-gpus 4 --config-file configs/faster_rcnn_R_50_FPN_3x_opendet.yaml \
        --eval-only MODEL.WEIGHTS output/faster_rcnn_R_50_FPN_3x_opendet/model_final.pth
```

* **Training**

The training process is the same as detectron2.
```
python tools/train_net.py --dist-url auto --num-gpus 4 --config-file configs/faster_rcnn_R_50_FPN_3x_opendet.yaml
```


### Citation

If you find our work useful for your research, please consider citing:

```BibTeX
@article{cheng2024uadet,
  title={UADet: A Remarkably Simple Yet Effective Uncertainty-Aware Open-Set Object Detection Framework},
  author={Cheng, Silin and Liu, Yuanpei and Han, Kai},
  journal={arXiv preprint arXiv:2412.09229},
  year={2024}
}
```