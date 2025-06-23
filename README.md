## <div align="center">STW-YOLO for Small Target Worker Detection in Large Construction Scenes</div>
See below for a quickstart install and usage examples, and see our Docs for full documentation on training, validation, prediction and deployment.

Pip install the Ultralytics package including all requirements in a [**Python>=3.8**](https://www.python.org/) environment with [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).

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
