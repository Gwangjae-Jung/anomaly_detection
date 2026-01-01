from    typing              import  Callable, Sequence, Self
from    pathlib             import  Path
import  torch
from    torch.utils.data    import  Dataset


__all__: list[str] = ['Dataset_MNIST', 'Dataset_MVTEC_AD']


# ALL_CLASSES__MNIST:       Sequence[int] = tuple(range(10))
# ALL_CLASSES__MVTEC_AD:    Sequence[str] = tuple(['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper',])
MAX_INTEGER_INTENSITY = 255.0


class Dataset_MNIST(Dataset):
    def __init__(
            self,
            path:           str | Path,
            loaded_classes: Sequence[int] = tuple(range(10)),
            transformer:    Callable[[torch.Tensor, object], torch.Tensor] = lambda x: x,
        ) -> Self:
        import  numpy   as  np
        _loaded = np.load(Path(path))
        self.__data     = torch.from_numpy(_loaded['data']).type(torch.float)
        self.__targets  = torch.from_numpy(_loaded['targets']).type(torch.long)
        arg = []
        for i in loaded_classes:
            arg += torch.argwhere(self.__targets==i).flatten().tolist()
        self.__data     = (self.__data[arg]/MAX_INTEGER_INTENSITY).unsqueeze(1) # Shape: (N, 1, 28, 28)
        self.__data     = transformer(self.__data)
        self.__targets  = self.__targets[arg]
        return
    
    
    def __len__(self) -> int:
        return len(self.__targets)
    
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.__data[index], self.__targets[index].item()


class Dataset_MVTEC_AD(Dataset):
    def __init__(
            self,
            root:           str | Path,
            loaded_classes: Sequence[str] = tuple(['good']),
            loaded_index:   Callable[[str, int], bool] = lambda cls, idx: True,
            transformer:    Callable[[torch.Tensor, object], torch.Tensor] = lambda x: x,
        ) -> Self:
        import  numpy   as  np
        images = []
        _loaded = np.load(Path(path))
        self.__data     = torch.from_numpy(_loaded['data']).type(torch.float)
        self.__targets  = torch.from_numpy(_loaded['targets']).type(torch.long)
        arg = []
        for i in loaded_classes:
            arg += torch.argwhere(self.__targets==i).flatten().tolist()
        self.__data     = self.__data[arg].unsqueeze(1)
        self.__targets  = self.__targets[arg]
        return
    
    
    def __len__(self) -> int:
        return len(self.__targets)
    
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.__data[index], self.__targets[index].item()
    