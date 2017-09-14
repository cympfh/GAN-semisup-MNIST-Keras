import numpy
from PIL import Image


def image_save(x, name, n=10):
    for i in range(n):
        arr = numpy.clip(x[i], 0, 1)
        arr = (arr * 255).reshape((28, 28)).astype(numpy.uint8)
        img = Image.fromarray(arr)
        img.save(name.format(i=i))
