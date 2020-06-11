import cv2
import  numpy as np

img=cv2.imread('static/moulds/male/medicine/bachelor/1321.png')
img_back=cv2.imread('static/background/bg2.jpg')

#日常缩放
height, width, channels = img.shape
img_back=cv2.resize(img_back, (width + 400, height), interpolation=cv2.INTER_CUBIC)

#转换hsv
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#获取mask
lower_blue=np.array([78,43,46])
upper_blue=np.array([100,255,255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)

#腐蚀膨胀
erode=cv2.erode(mask,None,iterations=1)
dilate=cv2.dilate(erode,None,iterations=1)

#遍历替换
center=[0, 200]#在新背景图片中的位置
for i in range(height):
    for j in range(width):
        if dilate[i,j]==0:#0代表黑色的点
            img_back[center[0]+i,center[1]+j]=img[i,j]#此处替换颜色，为BGR通道

cv2.imshow('res',img_back)

cv2.waitKey(0)
