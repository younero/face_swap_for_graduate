import cv2
from PIL import Image
from io import BytesIO
from faceswap import swap
import numpy as np
import os
import random


def cv2_to_PIL_Image(seamless_img):
    image = Image.fromarray(cv2.cvtColor(seamless_img, cv2.COLOR_BGR2RGB))

    f = BytesIO()
    image.save(f, "png")

    return f


def process(face, sex, major):
    n = 1
    mould_path = 'moulds/%s/%d/%s' % (sex, n, major)
    x = len(os.listdir(mould_path))
    mould = mould_path + '/%d.jpg' % int(x * random.random())
    seamless_im = swap(face, mould)

    return seamless_im


def excute(img, sex, major):
    img = Image.open(img)
    face = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    seamless_img = process(face, sex, major)
    f = cv2_to_PIL_Image(seamless_img)

    return f
