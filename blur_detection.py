import os
import numpy
# import BlurDetection
from imutils import paths
import argparse
import cv2
import sys
import scripts
import FocusMask


def evaluate(img_col, args):
    numpy.seterr(all='ignore')
    assert isinstance(img_col, numpy.ndarray), 'img_col must be a numpy array'
    assert img_col.ndim == 3, 'img_col must be a color image ({0} dimensions currently)'.format(img_col.ndim)
    assert isinstance(args, argparse.Namespace), 'args must be of type argparse.Namespace not {0}'.format(type(args))
    img_gry = cv2.cvtColor(img_col, cv2.COLOR_RGB2GRAY)
    rows, cols = img_gry.shape
    crow, ccol = rows//2, cols//2
    f = numpy.fft.fft2(img_gry)
    fshift = numpy.fft.fftshift(f)
    fshift[crow-75:crow+75, ccol-75:ccol+75] = 0
    f_ishift = numpy.fft.ifftshift(fshift)
    img_fft = numpy.fft.ifft2(f_ishift)
    img_fft = 20*numpy.log(numpy.abs(img_fft))
    if args.display and not args.testing:
        cv2.destroyAllWindows()
        scripts.display('img_fft', img_fft)
        scripts.display('img_col', img_col)
        cv2.waitKey(0)
    result = numpy.mean(img_fft)
    return img_fft, result, result < args.thresh

def blur_detector(img_col, thresh=10, mask=False):
    assert isinstance(img_col, numpy.ndarray), 'img_col must be a numpy array'
    assert img_col.ndim == 3, 'img_col must be a color image ({0} dimensions currently)'.format(img_col.ndim)
    args = scripts.gen_args()
    args.thresh = thresh
    if mask:
        return FocusMask.blur_mask(img)
    else:
        return evaluate(img_col=img_col, args=args)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to input directory of images")
ap.add_argument("-t", "--threshold", type=float, default=100.0,
	help="focus measures that fall below this value will be considered 'blurry'")
args = vars(ap.parse_args())


# loop over the input images
for imagePath in paths.list_images(args["images"]):
    image = cv2.imread(imagePath)
    img_fft, val, blurry = blur_detector(image)
    print(imagePath + "- {0} blurry = {1}".format(["isn't", "is"][blurry], val))