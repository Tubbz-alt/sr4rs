from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from .util import GaussianBlur

class UCMerced(Dataset):
    """
    UCMerced Dataset
    :param path: path where dataset is extracted
    """

    def __init__(self, path=None):
        self.path = Path(path)
        self.imgs = list(self.path.rglob("*.tif"))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        image = Image.open(img_path).convert("RGB")
        return image



class HQLQ(Dataset):
    """
    Downscales the Dataset and gives the downscaled images and a ground truth
    """

    def __init__(self, Dataset, augtrans=[], hqcropsize=96, scalingfactor=4):
        """
        :param Dataset
        :param augtrans list of augmentations transforms, eg. random crops etc
        :param hqcropsize cropsize of ground truth
        :param scalingfactor factor for downscaling
        """
        self.ds = Dataset
        crop = [] if hqcropsize==-1 else [transforms.CenterCrop((hqcropsize, hqcropsize))]
        self.transform_prehq = transforms.Compose(augtrans + crop )
        self.transform_prelq = transforms.Compose(
            [GaussianBlur(scalingfactor / 2), transforms.Resize(hqcropsize // scalingfactor, 0)]
        )
        self.transform_postlq = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
        self.transform_posthq = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: 2 * x - 1)])

    def __getitem__(self, idx):
        hq = self.transform_prehq(self.ds[idx])
        lq = self.transform_prelq(hq)
        return (self.transform_postlq(lq), self.transform_posthq(hq))

    def __len__(self):
        return len(self.ds)
