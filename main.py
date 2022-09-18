import cv2
import numpy as np
import os

# 開啟影片區
cap = cv2.VideoCapture(0) # 0~1

# 設定畫面尺寸
width = 100
height = 100
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, 25)

# 計算畫面積
area = width * height

ret, frame = cap.read()
avg = cv2.blur(frame, (4, 4))
avg_float = np.float32(avg)

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break

    # 模糊處理
    blur = cv2.blur(frame, (4, 4))

    # 計算目前影格與平均影像的差異值
    diff = cv2.absdiff(avg, blur)

    # 將圖片轉換為灰階
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # 篩選變動大於門檻的區域
    ret, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

    # 使用型態轉函數去除雜訊
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hasMotion = False

    for c in cnts:
        # 忽略太小的區域
        if cv2.contourArea(c) < 5000:
            continue

        hasMotion = True

        (x, y, w, h) = cv2.boundingRect(c)
        # 畫出外誆
        cv2.rectangle(frame, (x, y), (x + y, y + h), (0, 255, 0), 2)

    # 更新平均影像
    cv2.accumulateWeighted(blur, avg_float, 0.01)
    avg = cv2.convertScaleAbs(avg_float)
    cv2.imshow("test", frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()