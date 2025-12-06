from    typing      import  *
from    pathlib     import  Path
import  os

import  numpy       as      np
import  torch

import  cv2


__all__: list[str] = [
    'MAX_COLORSCALE',
    'load_images',
    'ConvFeatureMap',
]


##################################################
##################################################
MAX_COLORSCALE = 255.0


##################################################
##################################################
def load_images(
        folder_path:    Union[Path, str],
        transform:      Optional[Callable[[np.ndarray], torch.Tensor]] = None,
        device:         Optional[torch.device] = None,
    ) -> torch.Tensor:
    """Read the image files in `folder_path` after transformation defined by `transform`.
    
    Arguments:
        `folder_path` (`Path` or `str`):
            The path of the directory in which image files are loaded.
        `transform` (`Callable[[numpy.ndarray], torch.Tensor]` or `None`, default: `None`):
            The transformation to be applied to the loaded images.
            If `None`, then `transform` is initialized as `Compose([ToTensor(), Resize((128, 128)), Normalize(mean, std)])`, where `mean` and `std` are the mean and the standard deviation of the ImageNet dataset.
        `device` (`torch.device` or `None`, default: None`):
            The device in which the output tensor locates.
            If `None`, then `device` is set to `torch.get_default_device()`.
    """
    from    tqdm.notebook       import  tqdm
    if transform is None:
        from    torchvision.transforms  import  Compose, ToTensor, Resize, Normalize
        transform = Compose(
            [
                ToTensor(),
                Resize((128, 128)),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    if not isinstance(folder_path, Path):
        folder_path = Path(folder_path)
    if device is None:
        device = torch.get_default_device()
    
    images = []
    for file in tqdm(list(folder_path.glob('*')), desc=folder_path.name):
        file = str(file)
        if file.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder_path, file)
            img: np.ndarray
            img = cv2.imread(img_path)   # (H, W, 3)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
            img = transform(img)
            images.append(img)
    images = torch.stack(images, dim=0)

    return images.to(device)


##################################################
##################################################
class ConvFeatureMap(torch.nn.Module):
    """A convolutional feature extractor using ResNet50."""
    def __init__(self, device: Optional[torch.device]=None) -> Self:
        from    torchvision.models      import  resnet50, ResNet50_Weights
        super().__init__()
        if device is None:  device = torch.get_default_device()
        
        _resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.extractor = torch.nn.Sequential(*list(_resnet.children())[:-2])
        self.to(device)
        del(_resnet)
        
        return
    
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.extractor.forward(X)
    def compute_feature(self, X: torch.Tensor) -> torch.Tensor:
        return self.extractor.forward(X)


##################################################
##################################################
# End of file