from ultralytics import YOLO
import sys

if __name__ == '__main__':
    # 消融模型
    ablition_model = ["yolov8s","yolo11s","YOLOv11s+P2","YOLOv11s+P2+rP5","p2_rp5_new_yolo11s","p2_rp5_yolo11s"][-1]
    # ablition_model = sys.argv[1]
    # 加载模型
    print("model:",ablition_model)
    model = YOLO('./runs/{}/weights/best.pt'.format(ablition_model))  # build a new model from scratch
    print('========================================================================')
    res = model.val(data=r"./Lib/contr_worker.yaml", epochs=200, imgsz=640, batch=32, save_json=True)
    print(res)
    model.val(data=r"./Lib/mocs_test.yaml", epochs=200, imgsz=640, batch=32, save_json=True,name = ablition_model+"/"+"mocs_test" )
    model.val(data=r"./Lib/soda_test.yaml", epochs=200, imgsz=640, batch=32, save_json=True)
    data_list = ['DSWD','MOCS','SODA','CHVG']
    for data_i in data_list:
        for type in ['small','medium','large']:
            data_yaml = data_i+'_'+type+'.yaml'
            print(ablition_model,data_i,type)
            model.val(data=r"./Lib/{}".format(data_yaml),epochs=200,imgsz=640,batch=32, save_json=True)
    print('========================================================================')



