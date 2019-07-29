import matplotlib.pyplot as plt
import keras_segmentation

model = keras_segmentation.models.unet.vgg_unet(
    n_classes=3,  input_height=256, input_width=256)

model.train(
    train_images="J:/Imageset/keras_seg_imgset/train/JPEG/",
    train_annotations="J:/Imageset/keras_seg_imgset/seg/JPEG/",
    checkpoints_path="/tmp/vgg_unet_1", epochs=5
)

out = model.predict_segmentation(
    inp="J:/Imageset/keras_seg_imgset/train/JPEG/img_008.jpg",
    out_fname="out.png"
)

plt.imshow(out)
