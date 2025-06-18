## <div align="center">DWD-YOLO for dense small worker detection in large construction scenes</div>
See below for a quickstart install and usage examples, and see our Docs for full documentation on training, validation, prediction and deployment.

Pip install the Ultralytics package including all requirements in a [**Python>=3.8**](https://www.python.org/) environment with [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).

[![PyPI - Version](https://img.shields.io/pypi/v/ultralytics?logo=pypi&logoColor=white)](https://pypi.org/project/ultralytics/) [![Ultralytics Downloads](https://static.pepy.tech/badge/ultralytics)](https://www.pepy.tech/projects/ultralytics) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ultralytics?logo=python&logoColor=gold)](https://pypi.org/project/ultralytics/)

### Training
```bash
python yolo_train.py [model_cgf] [data] [save_name]
```

### Test
```bash
python yolo_test.py [model_weights] [data]
```
### IoU simulation experiment
```bash
# saving gt_pred_data.json file
python get_gt_pre_box.py

# simulation
python IoU_simulation.py
```

### Datasets and Pre-trained models
We constructed a Dense Small Worker Detection (DSWD) dataset on large construction sites to address the shortcomings of existing data, which contains 1705 low-resolution images and 48553 small worker instances from 25 construction sites acquired at different times and different illuminations. DSWD has 73.63\% of workers with areas less than $32\times32$ pixels, contains an average of 28 small workers per image, and has significant overlap and occlusion, which represents the challenge of dense small worker detection at large sites.
