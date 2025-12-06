from    typing      import  Callable
import  torch


__all__: list[str] = ['count_parameters', 'initialize_weights']


def count_parameters(model: torch.nn.Module, complex_as_two: bool=True) -> int:
    num_params: list[int] = []
    model: torch.nn.Module
    cnt = 0
    for p in model.parameters():
        cnt += p.numel() * (1 + (complex_as_two and p.is_complex()))
    num_params.append(cnt)
    return num_params


def initialize_weights(
        model:          torch.nn.Module,
        init_name:      str = "xavier normal",
        init_kwargs:    dict[str, object] = {},
    ) -> None:
    """Initialize the weights in `models`.
    
    -----
    Arguments:
        `model` (`torch.nn.Module`):
            A model or a sequence of models to be initialize.
        `init_name` (`str`):
            The method to be used to initialize the model(s). The possible names are: `"constant"`, `"dirac"`, `"eye"`, `"kaiming normal"`, `"kaiming uniform"`, `"normal"`, `"ones"`, `"orthogonal"`, `"sparse"`, `"trunc_normal"`, `"uniform"`, `"xavier_normal"`, `"xavier uniform"`, `"zeros"`.
        `init_kwargs` (`dict[str, object]`, default: `{}`):
            Any further arguments for weight initialization.
    """
    try:
        initializer: Callable[[torch.Tensor, object], torch.Tensor] = getattr(torch.nn.init, init_name)
    except:
        raise KeyError(
            f"The passed value {init_name} of 'init_name'is not in the list of supported initalization."
        )
    for p in model.parameters():
        try:
            initializer(p, **init_kwargs)
        except:
            continue
    return
