import cv2
from PIL import Image
from io import BytesIO
from faceswap import swap
import numpy
import os
import random


def detect(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图片转化成灰度

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")  # 加载级联分类器模型
    face_cascade.load('static/haarcascade_frontalface_alt2.xml')  # 一定要告诉编译器文件所在的具体位置
    '''此文件是opencv的haar人脸特征分类器'''
    locs = face_cascade.detectMultiScale(gray, 1.3, 5)

    faces = []

    for (x, y, w, h) in locs:
        a, b, c, d = x, y, x + w, y + h
        face = img[b:d, a:c]
        faces.append(face)
    return faces


def save_temp_imgs(list):
    temp_imgs = []
    for l in list:
        pic = Image.open(l)
        img = cv2.cvtColor(numpy.asarray(pic), cv2.COLOR_RGB2BGR)
        temp_imgs.append(img)

    return temp_imgs


def join(seamless_ims):
    ims = []

    for seamless_im in seamless_ims:
        image = Image.fromarray(cv2.cvtColor(seamless_im, cv2.COLOR_BGR2RGB))
        ims.append(image)

    width = 0
    for im in ims:
        w, height = im.size
        width += w

    # 创建空白长图
    result = Image.new(ims[0].mode, (width, height))


    # 拼接图片
    temp = 0
    for im in ims:
        w, height = im.size
        result.paste(im, box=(temp, 0))
        temp += w

    img = cv2.cvtColor(numpy.asarray(result), cv2.COLOR_RGB2BGR)

    return img


def get_moulds_path(sex, major, dgree):
    path = 'static/moulds/%s/%s/%s' % (sex, major, dgree)
    return path


def process(faces, moulds_path):
    seamless_ims = []

    moulds = os.listdir(moulds_path)

    n = len(faces)
    for i in range(n):
        face = faces[i]
        x = random.randint(0, len(moulds)-1)
        mould = moulds_path + '/' + moulds[x]
        seamless_im = swap(face, mould)
        seamless_ims.append(seamless_im)

    return seamless_ims


def load_background(img):
    img_back_path = 'static/background/'
    list = os.listdir(img_back_path)
    x = random.randint(0, len(list) - 1)

    img_back = cv2.imread('static/background/' + list[x])

    # 日常缩放
    height, width, channels = img.shape
    img_back = cv2.resize(img_back, (width + 400, height), interpolation=cv2.INTER_CUBIC)

    # 转换hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 获取mask
    lower_blue = numpy.array([100, 75, 75])
    upper_blue = numpy.array([101, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 腐蚀膨胀
    erode = cv2.erode(mask, None, iterations=1)
    dilate = cv2.dilate(erode, None, iterations=1)

    # 遍历替换
    center = [0, 200]  # 在新背景图片中的位置
    for i in range(height):
        for j in range(width):
            if dilate[i, j] == 0:
                try:
                    img_back[center[0] + i, center[1] + j] = img[i , j]  # 此处替换颜色，为BGR通道
                except:
                    pass

    img_back = img_back[:, :, ::-1]
    img_with_background = Image.fromarray(img_back)

    return img_with_background


def excute(list, sex, major, degree):
    paths = save_temp_imgs(list)
    faces_ = []
    for path in paths:
        faces = detect(path)
        faces_ += faces
    moulds_path = get_moulds_path(sex, major, degree)
    seamless_ims = process(faces_, moulds_path)
    img = join(seamless_ims)
    img_with_background = load_background(img)
    f = BytesIO()
    img_with_background.save(f, "png")

    return f
