# Performance-evaluation-of-vehicle-detection-and-tracking-using-Yolo-s
from google.colab import drive
drive.mount('/content/drive')
pip install ultralytics -q
#Auxiliary functions
def risize_frame(frame, scale_percent):
    """Function to resize an image in a percent scale"""
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return resized
# Object Detecion
import cv2
from ultralytics import YOLO
#plots
import matplotlib.pyplot as plt
import seaborn as sns

#basics
import pandas as pd
import numpy as np
import os
import subprocess

from tqdm.notebook import tqdm

# Display image and videos
import IPython
from IPython.display import Video, display
from matplotlib import pyplot as plt
%matplotlib inline

Defdetect_vehicles(input_video_name,output_video_name,objects_to_detect,yolo_model_version="yolov8x.pt",accuracy_confidence_filter=None):
    ### Configurations
    # Input video path
    path = f'/content/drive/MyDrive/yolo_project/dataset/{input_video_name}'

    #loading a YOLO model
    model = YOLO(yolo_model_version)

    #geting names from classes
    dict_classes = model.model.names

   #Verbose during prediction
    verbose = False
    # Scaling percentage of original frame
    scale_percent = 50

    #-------------------------------------------------------
    # Reading video with cv2
    video = cv2.VideoCapture(path)

    # Objects to detect Yolo
    class_IDS = objects_to_detect
    # Auxiliary variables
    centers_old = {}
    centers_new = {}
    obj_id = 0
    veiculos_contador_in = dict.fromkeys(class_IDS, 0)
    veiculos_contador_out = dict.fromkeys(class_IDS, 0)
    end = []
    frames_list = []
    cy_linha = int(1500 * scale_percent/100 )
    cx_sentido = int(2000 * scale_percent/100)
    offset = int(8 * scale_percent/100 )
    contador_in = 0
    contador_out = 0
    print(f'[INFO] - Verbose during Prediction: {verbose}')

    # Original informations of video
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = video.get(cv2.CAP_PROP_FPS)
    print('[INFO] - Original Dim: ', (width, height))

    # Scaling Video for better performance
    if scale_percent != 100:
        print('[INFO] - Scaling change may cause errors in pixels lines ')
        width = int(width * scale_percent / 100)
        height = int(height * scale_percent / 100)
        print('[INFO] - Dim Scaled: ', (width, height))

    #-------------------------------------------------------
    ### Video output ####
    video_name = output_video_name
    output_path = "rep_" + video_name
    tmp_output_path = "/content/drive/MyDrive/yolo_project/" + output_path
    VIDEO_CODEC = "MP4V"

    output_video = cv2.VideoWriter(tmp_output_path,
                                  cv2.VideoWriter_fourcc(*VIDEO_CODEC),
                                  fps, (width, height))

    #-------------------------------------------------------
    # Executing Recognition
    frames = []
    for i in tqdm(range(int(video.get(cv2.CAP_PROP_FRAME_COUNT)))):

        # reading frame from video
        _, frame = video.read()

        #Applying resizing of read frame
        frame  = risize_frame(frame, scale_percent)

        if verbose:
            print('Dimension Scaled(frame): ', (frame.shape[1], frame.shape[0]))

        # Getting predictions
        y_hat = model.predict(frame, conf = 0.7, classes = class_IDS, device = 0, verbose = False)

        # Getting the bounding boxes, confidence and classes of the recognize objects in the current frame.
        boxes   = y_hat[0].boxes.xyxy.cpu().numpy()
        conf    = y_hat[0].boxes.conf.cpu().numpy()
        classes = y_hat[0].boxes.cls.cpu().numpy()

        # Storing the above information in a dataframe
        positions_frame = pd.DataFrame(y_hat[0].cpu().numpy().boxes.data, columns = ['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class'])

        frames.extend(y_hat[0].cpu().numpy().boxes.data)
        #Translating the numeric class labels to text
        labels = [dict_classes[i] for i in classes]

        # Drawing transition line for in\out vehicles counting
        cv2.line(frame, (0, cy_linha), (int(4500 * scale_percent/100 ), cy_linha), (255,255,0),8)

        # For each vehicles, draw the bounding-box and counting each one the pass thought the transition line (in\out)
        for ix, row in enumerate(positions_frame.iterrows()):
            # Getting the coordinates of each vehicle (row)
            xmin, ymin, xmax, ymax, confidence, category,  = row[1].astype('int')

            # Calculating the center of the bounding-box
            center_x, center_y = int(((xmax+xmin))/2), int((ymax+ ymin)/2)

            # drawing center and bounding-box of vehicle in the given frame
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 5) # box
            cv2.circle(frame, (center_x,center_y), 5,(255,0,0),-1) # center of box

            #Drawing above the bounding-box the name of class recognized.
            cv2.putText(img=frame, text=labels[ix]+' - '+str(np.round(conf[ix],2)),
                        org= (xmin,ymin-10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 0, 0),thickness=2)

            # Checking if the center of recognized vehicle is in the area given by the transition line + offset and transition line - offset
            if (center_y < (cy_linha + offset)) and (center_y > (cy_linha - offset)):
                if  (center_x >= 0) and (center_x <=cx_sentido):
                    contador_in +=1
                    veiculos_contador_in[category] += 1
                else:
                    contador_out += 1
                    veiculos_contador_out[category] += 1

        #updating the counting type of vehicle
        contador_in_plt = [f'{dict_classes[k]}: {i}' for k, i in veiculos_contador_in.items()]
        contador_out_plt = [f'{dict_classes[k]}: {i}' for k, i in veiculos_contador_out.items()]

        #drawing the number of vehicles in\out
        cv2.putText(img=frame, text='N. vehicles In',
                    org= (30,30), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=1, color=(255, 255, 0),thickness=1)

        cv2.putText(img=frame, text='N. vehicles Out',
                    org= (int(2800 * scale_percent/100 ),30),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 0),thickness=1)

        #drawing the counting of type of vehicles in the corners of frame
        xt = 40
        for txt in range(len(contador_in_plt)):
            xt +=30
            cv2.putText(img=frame, text=contador_in_plt[txt],
                        org= (30,xt), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=1, color=(255, 255, 0),thickness=1)

            cv2.putText(img=frame, text=contador_out_plt[txt],
                        org= (int(2800 * scale_percent/100 ),xt), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=1, color=(255, 255, 0),thickness=1)

        #drawing the number of vehicles in\out
        cv2.putText(img=frame, text=f'In:{contador_in}',
                    org= (int(1820 * scale_percent/100 ),cy_linha+60),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 0),thickness=2)

        cv2.putText(img=frame, text=f'Out:{contador_out}',
                    org= (int(1800 * scale_percent/100 ),cy_linha-40),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 0),thickness=2)

        if verbose:
            print(contador_in, contador_out)
        #Saving frames in a list
        frames_list.append(frame)
        #saving transformed frames in a output video formaat
        output_video.write(frame)

    #Releasing the video
    output_video.release()

    ####  pos processing
    # Fixing video output codec to run in the notebook\browser

    subprocess.run(
        ["ffmpeg",  "-i", tmp_output_path,"-crf","18","-preset","veryfast","-hide_banner","-loglevel","error","-vcodec","libx264",output_path])
    accuracy_detections_df = pd.DataFrame(frames, columns = ['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class'])
    conf_df = accuracy_detections_df[["conf"]]
    if accuracy_confidence_filter is not None:
      # filter frames those are greater than or equal to given accuracy_confidence_filter
      conf_df = conf_df[conf_df["conf"]>=accuracy_confidence_filter]
    conf_df['conf'].plot(kind='line', figsize=(8, 4), title=f'Car Detection Accuracy with {yolo_model_version}')
    plt.gca().spines[['top', 'right']].set_visible(False)

input_video_name="vehicle-counting.mp4"
output_video_name="vehicle-counting-yolov3.mp4"
objects_to_detect=[2,3,5,7]
yolo_model_version="yolov3u.pt"
accuracy_confidence_filter = 0.9
detect_vehicles(input_video_name,output_video_name,objects_to_detect,yolo_model_version=yolo_model_version, accuracy_confidence_filter=accuracy_confidence_filter)


[INFO] - Verbose during Prediction: False
[INFO] - Original Dim:  (3840, 2160)
[INFO] - Scaling change may cause errors in pixels lines 
[INFO] - Dim Scaled:  (1920, 1080)
100%
538/538 [01:03<00:00, 12.51it/s]

input_video_name="vehicle-counting.mp4"
output_video_name="vehicle-counting-yolov3.mp4"
objects_to_detect=[2,3,5,7]
yolo_model_version="yolov5x.pt"
accuracy_confidence_filter = 0.9
detect_vehicles(input_video_name,output_video_name,objects_to_detect,yolo_model_version=yolo_model_version, accuracy_confidence_filter=accuracy_confidence_filter)
PRO TIP 💡 Replace 'model=yolov5x.pt' with new 'model=yolov5xu.pt'.
YOLOv5 'u' models are trained with https://github.com/ultralytics/ultralytics and feature improved performance vs standard YOLOv5 models trained with https://github.com/ultralytics/yolov5.

[INFO] - Verbose during Prediction: False
[INFO] - Original Dim:  (3840, 2160)
[INFO] - Scaling change may cause errors in pixels lines 
[INFO] - Dim Scaled:  (1920, 1080)
100%
538/538 [01:04<00:00, 12.18it/s]

input_video_name="vehicle-counting.mp4"
output_video_name="vehicle-counting-yolov3.mp4"
objects_to_detect=[2,3,5,7]
yolo_model_version="yolov8x.pt"
accuracy_confidence_filter = 0.9
detect_vehicles(input_video_name,output_video_name,objects_to_detect,yolo_model_version=yolo_model_version, accuracy_confidence_filter=accuracy_confidence_filter)

[INFO] - Verbose during Prediction: False
[INFO] - Original Dim:  (3840, 2160)
[INFO] - Scaling change may cause errors in pixels lines 
[INFO] - Dim Scaled:  (1920, 1080)
100%
538/538 [00:50<00:00, 12.36it/s]

model = YOLO('yolov8x.pt')
dict_classes = model.model.names
dict_classes


