from    typing              import  Sequence, Self
from    pathlib             import  Path
import  torch
from    torch.utils.data    import  Dataset


__all__: list[str] = ['Dataset_MNIST']


class Dataset_MNIST(Dataset):
    def __init__(
            self,
            path:           str | Path,
            normal_classes: Sequence[int] = tuple(range(10)),
        ) -> Self:
        import  numpy   as  np
        _loaded = np.load(Path(path))
        self.__data     = torch.from_numpy(_loaded['data']).type(torch.float)
        self.__targets  = torch.from_numpy(_loaded['targets']).type(torch.long)
        arg = []
        for i in normal_classes:
            arg += torch.argwhere(self.__targets == i).flatten().tolist()
        self.__data     = self.__data[arg].unsqueeze(1) / 255.0 # Shape: (N, 1, 28, 28)
        self.__targets  = self.__targets[arg]
        return
    
    
    def __len__(self) -> int:
        return len(self.__targets)
    
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.__data[index], self.__targets[index].item()
    