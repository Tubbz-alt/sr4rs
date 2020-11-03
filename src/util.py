from torch.utils.data import random_split

def random_split_ratio(ds, validation_ratio, test_ratio):
    """
    splits dataset by ratio (0..1) of validation and test in validation, test and train (remainder)
    """
    l = len(ds)
    val, test = int(l * validation_ratio), int(l * test_ratio)
    train = l - (val + test)
    return random_split(ds, [train, val, test])

from PIL import ImageFilter
class GaussianBlur:
    """
    Blur with Gaussian, can be used in transforms.Compose
    :param radius
    """

    def __init__(self, radius):
        self.radius = radius

    def __call__(self, image):
        return image.filter(ImageFilter.GaussianBlur(self.radius))

import math    
def psnr(gt, img, max_val=1.):
    """
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition
    """
    rmse = math.sqrt( ((img-gt) ** 2).mean().cpu().detach() )
    if rmse == 0:
        return 100
    else:
        PSNR = 20 * math.log10(max_val / rmse)
        return PSNR

