import numpy as np
import sys
import string
import os
import shutil
import cv2
import glob
import random
from skimage import color, exposure


def ComputeHist(img):
    h, w = img.shape
    hist, bin_edge = np.histogram(img.reshape(1, w*h), bins=list(range(257)))
    return hist


def ComputeMinLevel(hist, rate, pnum):
    sum = 0
    for i in range(256):
        sum += hist[i]
        if (sum >= (pnum * rate * 0.01)):
            return i


def ComputeMaxLevel(hist, rate, pnum):
    sum = 0
    for i in range(256):
        sum += hist[255-i]
        if (sum >= (pnum * rate * 0.01)):
            return 255-i


def LinearMap(minlevel, maxlevel):
    if (minlevel >= maxlevel):
        return []
    else:
        newmap = np.zeros(256)
        for i in range(256):  # 获取阈值外的像素值 i< minlevel，i> maxlevel
            if (i < minlevel):
                newmap[i] = 0
            elif (i > maxlevel):
                newmap[i] = 255
            else:
                newmap[i] = (i-minlevel)/(maxlevel-minlevel) * 255
        return newmap


def CreateNewImg(img):
    h, w, d = img.shape
    newimg = np.zeros([h, w, d])
    for i in range(d):
        imgmin = np.min(img[:, :, i])
        imgmax = np.max(img[:, :, i])
        imghist = ComputeHist(img[:, :, i])
        minlevel = ComputeMinLevel(imghist, 8.3, h*w)
        maxlevel = ComputeMaxLevel(imghist, 2.2, h*w)
        newmap = LinearMap(minlevel, maxlevel)
        if (newmap.size == 0):
            continue
        for j in range(h):
            newimg[j, :, i] = newmap[img[j, :, i]]
    return newimg


def ContrastImg(img1, c, b):  # 亮度就是每个像素所有通道都加上b
    rows, cols, chunnel = img1.shape
    # np.zeros(img1.shape, dtype=uint8)
    blank = np.zeros([rows, cols, chunnel], img1.dtype)
    dst = cv2.addWeighted(img1, c, blank, 1-c, b)
    #cv2.imshow("con_bri_demo", dst)
    return dst

# 输入目录名和前缀名，重命名后的名称结构类似prefix_0001


def RenameFiles(srcdir, prefix):
    srcfiles = os.listdir(srcdir)
    index = 1
    for srcfile in srcfiles:
        srcfilename = os.path.splitext(srcfile)[0][1:]
        sufix = os.path.splitext(srcfile)[1]
        # 根据目录下具体的文件数修改%号后的值，"%04d"最多支持9999
        destfile = srcdir + prefix + "_%03d" % (index) + ".png"
        srcfile = os.path.join(srcdir, srcfile)
        # 仅更换名字
        # os.rename(srcfile, destfile)
        # 复制并更换名字
        shutil.copyfile(srcfile, destfile)
        index += 1


if __name__ == '__main__':
    #img = cv2.imread('./20181106194742.png')
    img_path = "J:/Imageset/keras_seg_imgset/seg/JPEG/"
    images = sorted(glob.glob(img_path + "*.jpg"))
    i = 1

    prefix = "img"
    # 批量换名字或者复制换名字
    RenameFiles(img_path, prefix)

    #cv2.imshow('original_img', img)
    #cv2.imshow('new_img', newimg/255)
    cv2.waitKey()
    cv2.destroyAllWindows()
