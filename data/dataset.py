from typing import Optional, Callable, Any

from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision import transforms as T


class AnimeFaces128Dataset(ImageFolder):
    """
    数据集
    """

    def __init__(self, image_size: int, root: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None,
                 loader: Callable[[str], Any] = default_loader, is_valid_file: Optional[Callable[[str], bool]] = None):
        if not transform:
            transform = T.Compose([
                T.Resize(image_size),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        super().__init__(root, transform, target_transform, loader, is_valid_file)
