from ultralytics import YOLO
import sys

if __name__ == '__main__':
    # 加载模型
    model = YOLO('checkpoints/yolo12s.pt')  # build a new model from scratch
    # model_cgf = sys.argv[1]
    # data = sys.argv[2]
    # name = sys.argv[3]
    print("---------------------p2_rp5_sf_yolo12s--------------------")
    model.train(data=r"./Lib/contr_worker.yaml",
                epochs=200,
                model= "./Lib/p2_rp5_sf_yolo12s.yaml",
                name = "sas_yolo11s",
                imgsz=640,
                batch=32)