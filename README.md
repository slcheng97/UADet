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


<table style="border-collapse: collapse; font-size: 0.78em; width: auto;">
  <thead>
    <tr>
      <th style="border: 1px solid #ddd; padding: 3px 5px;">Method</th>
      <th style="border: 1px solid #ddd; padding: 3px 5px;">backbone</th>
      <th style="border: 1px solid #ddd; padding: 3px 5px;">mAP<sub>&uarr;</sub>(VOC)</th>
      <th style="border: 1px solid #ddd; padding: 3px 5px;">WI<sub>&darr;</sub></th>
      <th style="border: 1px solid #ddd; padding: 3px 5px;">AOSE<sub>&darr;</sub></th>
      <th style="border: 1px solid #ddd; padding: 3px 5px;">mAP<sub>&uarr;</sub></th>
      <th style="border: 1px solid #ddd; padding: 3px 5px;">U-AP<sub>&uarr;</sub></th>
      <th style="border: 1px solid #ddd; padding: 3px 5px;">U-Recall<sub>&uarr;</sub></th>
      <th style="border: 1px solid #ddd; padding: 3px 5px; white-space: nowrap;">Download</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">FR-CNN</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">R-50</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">80.06</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">19.50</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">16518</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">58.36</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">0</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">0</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px; white-space: nowrap;"><a href="configs/faster_rcnn_R_50_FPN_3x_baseline.yaml">config</a> <a href="https://drive.google.com/drive/folders/10uFOLLCK4N8te08-C-olRyDV-cJ-L6lU?usp=sharing">model</a></td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">PROSER</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">R-50</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">79.42</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">20.44</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">14266</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">56.72</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">16.99</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">37.34</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px; white-space: nowrap;"><a href="configs/faster_rcnn_R_50_FPN_3x_proser.yaml">config</a> <a href="https://drive.google.com/drive/folders/1_L85gisyvDtBXPe2UbI49vrd5FoBIOI_?usp=sharing">model</a></td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">ORE</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">R-50</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">79.80</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">18.18</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">12811</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">58.25</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">2.60</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">-</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px; white-space: nowrap;"><a href="#">config</a> <a href="#">model</a></td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">DS</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">R-50</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">79.70</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">16.76</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">13062</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">58.46</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">8.75</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">19.80</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px; white-space: nowrap;"><a href="configs/faster_rcnn_R_50_FPN_3x_ds.yaml">config</a> <a href="https://drive.google.com/drive/folders/1OWDjL29E2H-_lSApXqM2r8PS7ZvUNtiv?usp=sharing">model</a></td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">OpenDet</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">R-50</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">80.02</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">12.50</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">10758</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">58.64</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">14.38</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">37.65</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px; white-space: nowrap;"><a href="configs/faster_rcnn_R_50_FPN_3x_opendet.yaml">config</a> <a href="https://drive.google.com/drive/folders/1fzD0iJ6lJrPL4ffByeO9M-udckbYqIxY?usp=sharing">model</a></td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">UAdet</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">R-50</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">80.13</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">13.19</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">10186</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">59.12</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">15.09</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px;">59.03</td>
      <td style="border: 1px solid #ddd; padding: 3px 5px; white-space: nowrap;"><a href="configs/faster_rcnn_R_50_FPN_3x_uadet.yaml">config</a> <a href="https://drive.google.com/drive/folders/uadet_models">model</a></td>
    </tr>
  </tbody>
</table>

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