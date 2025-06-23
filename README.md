## <div align="center">STW-YOLO for Small Target Worker Detection in Large Construction Scenes</div>
See below for a quickstart install and usage examples, and see our Docs for full documentation on training, validation, prediction and deployment.

Pip install the STW-YOLO package including all requirements in a [**Python>=3.10**](https://www.python.org/) environment with [**PyTorch>=2.1.2**](https://pytorch.org/get-started/locally/) and CUDA>=11.8.

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

### Datasets

### Evaluation
Because we are using the [HR-FPN](https://github.com/laochen330/HR-FPN) evaluation metrics, you will need to replace the cocoeval.py file in the “coco” folder with the cocoeval.py file in the “pycocotools” folder in your environment to ensure that the program runs correctly.
