import numpy as np
import PIL.Image

#this file is to test the conversion from the label file genrated by labelme to a np array

label_png = "Image_Segmentation\\labels\\0001_2\\label.png"

lbl = np.asarray(PIL.Image.open(label_png))
print(lbl.dtype)
print(lbl.shape)
print(np.unique(lbl))