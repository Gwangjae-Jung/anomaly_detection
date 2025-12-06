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
        self.__data   = torch.from_numpy(np.load(Path(path))).type(torch.float)
        self.__label  = torch.from_numpy(np.load(Path(path))).type(torch.long)
        arg = []
        for i in normal_classes:
            arg += torch.argwhere(self.__label == i).flatten().tolist()
        self.__data   = self.__data[arg]
        self.__label  = self.__label[arg]
        return
    
    
    def __len__(self) -> int:
        return len(self.__label)
    
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.__data[index], self.__label[index].item()
    