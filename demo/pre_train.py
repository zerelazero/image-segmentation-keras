import keras_segmentation
import cv2

# load the pretrained model trained on ADE20k dataset
model1 = keras_segmentation.pretrained.pspnet_50_ADE_20K()

# load the pretrained model trained on Cityscapes dataset
model2 = keras_segmentation.pretrained.pspnet_101_cityscapes()

# load the pretrained model trained on Pascal VOC 2012 dataset
model3 = keras_segmentation.pretrained.pspnet_101_voc12()

# load any of the 3 pretrained models

out = model1.predict_segmentation(
    inp="img001.jpg",
    out_fname="out001.png"
)

cv2.imread(out)
