import cv2 #opencv
import urllib.request #to open and read URL
import numpy as np
from ultralytics import YOLO


# OBJECT CLASSIFICATION PROGRAM FOR VIDEO IN IP ADDRESS
model_path = r'models\car-sp.pt'
url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTpsMrhRprh8i-D6tmcbvmZGLcuWc8lx4eG_A&s'
thickness = 2
color = (184, 122, 61) # BGR --> blue, gree, red
classNames = ['license-plate', 'spppp']
winName = 'ESP32 CAMERA'

model = YOLO(model_path)

while(1):
    imgResponse = urllib.request.urlopen (url) # here open the URL
    imgNp = np.array(bytearray(imgResponse.read()),dtype=np.uint8) # np.array必須傳入可迭代的物件，所以須將 bytes轉乘bytearray
    img = cv2.imdecode (imgNp,-1) #decodificamos，將二進制格式的圖片轉換成 OpenCV 可用的物件

    ## YOLOv8 模型預測
    results = model(img, conf=0.3)
    res = results[0]
    object_cout = len(res)
    if len(res) != 0:
        for i in range(object_cout):
            x1, y1, x2, y2 = map(int, res.boxes.xyxy[i])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(img, classNames[int(res.boxes.cls[i])], (x1+10, y1+30), cv2.FONT_HERSHEY_COMPLEX, 1, color, thickness)
    else:
        print("No object detect")


     
    #wait for ESC to be pressed to end the program
    cv2.imshow(winName,img)
    kb = cv2.waitKey(1) 
    if kb == 27: break # "Esc" 鍵的 ASCII，ord("\x1b") = 27
    
cv2.destroyAllWindows()
