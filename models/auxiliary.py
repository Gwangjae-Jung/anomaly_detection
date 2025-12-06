from    typing  import  Self
import  torch
from    torch.nn    import  functional   as  F


class Autoencoder_MNIST(torch.nn.Module):
    def __init__(self) -> Self:
        super().__init__()
        self.__rep_dim = 32

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 5, bias=False, padding=2),
            torch.nn.BatchNorm2d(8, eps=1e-04, affine=False),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(2, 2),
            
            torch.nn.Conv2d(8, 4, 5, bias=False, padding=2),
            torch.nn.BatchNorm2d(4, eps=1e-04, affine=False),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(2, 2),
            
            torch.nn.Flatten(),
            torch.nn.Linear(4*7*7, self.__rep_dim, bias=False),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.ConvTranspose2d(self.__rep_dim//16, 4, 5, bias=False, padding=2),
            
            torch.nn.BatchNorm2d(4, eps=1e-04, affine=False),
            torch.nn.LeakyReLU(),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.ConvTranspose2d(4, 8, 5, bias=False, padding=3),
            
            torch.nn.BatchNorm2d(8, eps=1e-04, affine=False),
            torch.nn.LeakyReLU(),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.ConvTranspose2d(8, 1, 5, bias=False, padding=2),
            
            torch.nn.Sigmoid(),
        )
        
        return


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.encoder.forward(x)
        x = x.view(x.size(0), self.__rep_dim//16, 4, 4)
        return self.decoder.forward(x)