from    typing      import  Self, Optional
import  torch


_DEBUG: bool = True


def _construct_base_encoder() -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(28 * 28, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
    )


class DeepSVDD(torch.nn.Module):
    """An implementation of the Deep Support Vector Data Description (Deep SVDD) model."""
    def __init__(
            self,
            encoder:            torch.nn.Module,
            nu:                 float = 0.05,
            C:                  float = 1.0,
            is_soft_boundary:   bool = True,
            device:             Optional[torch.device] = None,
        ) -> Self:
        """The initializer for `DeepSVDD`.
        
        Arguments:
            `encoder` (`torch.nn.Module`):
                The encoder network to be fine-tuned in the Deep SVDD model. See **Remark** for the constraints of the encoder.
            `nu` (`float`, default=`0.05`):
                The hyperparameter nu in the Deep SVDD model, which is an upper bound on the fraction of outliers and a lower bound on the fraction of samples being outside or on the boundary of the hypersphere.
            `C` (`float`, default=`1.0`):
                The penalty term for the soft-boundary Deep SVDD loss.
            `device` (`Optional[torch.device]`, default=`None`):
                The device to which the model should be moved. If `None`, the default device is used.
        
        ### Remark
        1. Constraints of the encoder
            * It should not contain any bias terms in its layers.
            * The activation functions should not be bounded both above and below. Hence, ReLU cannot be used.
        """
        super().__init__()
        self.encoder: torch.nn.Module
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = _construct_base_encoder()
        for md_enc in self.encoder.modules():
            for p, _ in md_enc.named_parameters():
                if p.endswith("bias"):
                    raise AttributeError("Deep SVDD encoder should not contain bias terms.")
        
        if not (isinstance(nu, float) and nu>=0.0 and nu<1.0):
            raise ValueError("The hyperparameter nu should be in the range [0, 1).")
        if (not isinstance(C, float)) or C<0.0:
            raise ValueError("The penalty term should be a nonnegative real number.")

        self.__nu:      float   = nu
        self.__C:       float   = C
        self.__center:  Optional[torch.Tensor]  = None
        self.__is_soft: bool    = is_soft_boundary
        
        self.r_soft         = torch.nn.Parameter(torch.randn((1,), device=device, requires_grad=True))
        self.__r_one_class    = torch.tensor([0.0], device=device)
        
        if device is None: device = torch.get_default_device()
        self.__device = device
        self.to(device)
        return
    
    
    def initialize_center(self, x: Optional[torch.Tensor]=None, dataloader: Optional[torch.utils.data.DataLoader]=None, eps: float=1e-2) -> None:
        """Initializes the center of the hypersphere.
        
        Arguments:
            `x` (`Optional[torch.Tensor]`, default=`None`):
                The input tensor of shape `(N, ...)` to compute the initial center. If `None`, the `dataloader` argument must be provided.
            `dataloader` (`Optional[torch.utils.data.DataLoader]`, default=`None`):
                The dataloader to compute the initial center. If `None`, the `x` argument must be provided.
            `eps` (`float`, default=`1e-2`):
                A small value to avoid having zero-valued components in the center.
        """
        if x is None and dataloader is None:
            raise ValueError("Either 'x' or 'dataloader' must be provided to initialize the center.")
        flag = self.training
        self.eval()
        with torch.inference_mode():
            if x is not None:
                c = self.encoder.forward(x.to(self.__device)).flatten(start_dim=1)
            else:
                x = []
                for data, _ in dataloader:
                    x.append(self.encoder.forward(data.to(self.__device)).flatten(start_dim=1))
                c = torch.cat(x, dim=0)
        c = c.mean(dim=0)
        c[(torch.abs(c) < eps) & (c < 0)] = -eps
        c[(torch.abs(c) < eps) & (c > 0)] = eps
        self.__center = c
        self.train(flag)
        return
    
    
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the anomaly scores for the input samples. The score of an input sample `p` is defined as the norm of `encoder(p)-center`.
        
        Arguments:
            `x` (`torch.Tensor`): The input tensor of shape `(batch_size, ...)`.
        Returns:
            `torch.Tensor`: The computed anomaly scores of shape `(batch_size, 1)`.
        """
        z: torch.Tensor = self.encoder.forward(x).flatten(start_dim=1)
        base_score = (z-self.__center).norm(p=2, dim=1, keepdim=True)
        if self.__is_soft:
            return base_score - self.r_soft.pow(2)
        else:
            return base_score
    
    
    def _compute_penalty(self, unit: bool=False) -> torch.Tensor:
        """Computes the penalty term of shape `(1,)`, which is defined as the sum of the Frobenius norms of the weights of all layers in the encoder, multiplied by `C/2`.
        """
        penalty = torch.zeros((1,), device=self.__device)
        for p in self.encoder.parameters():
            penalty = penalty + torch.norm(p, p='fro').pow(2)
        penalty = penalty/2
        if unit:    return penalty
        else:       return self.__C*penalty
    
    
    def compute_loss__soft_boundary(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the soft-boundary Deep SVDD loss.
        
        Arguments:
            `x` (`torch.Tensor`): The input tensor of shape `(batch_size, ...)`.
        
        Returns:
            `torch.Tensor`: The computed loss of shape `(1,)`.
        """
        anomaly_score = self.anomaly_score(x).pow(2).mean()/self.__nu
        penalty = self._compute_penalty()
        if _DEBUG and anomaly_score.numel()>1:
            raise RuntimeError(f"The average anomaly score should be a single scalar value, but is of shape {list(anomaly_score.shape)}.")
        if _DEBUG and penalty.numel()>1:
            raise RuntimeError(f"The penalty term should be a single scalar value, but is of shape {list(penalty.shape)}.")
        return torch.clamp_min(anomaly_score, self.r_soft.pow(2)) + penalty
    
    
    def compute_loss__one_class(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the one-class Deep SVDD loss.
        
        Arguments:
            `x` (`torch.Tensor`): The input tensor of shape `(batch_size, ...)`.
        
        Returns:
            `torch.Tensor`: The computed loss of shape `(1,)`.
        """
        anomaly_scores = self.anomaly_score(x).pow(2).mean()
        penalty = self._compute_penalty()
        if _DEBUG and anomaly_scores.numel()>1:
            raise RuntimeError(f"The average anomaly score should be a single scalar value, but is of shape {list(anomaly_scores.shape)}.")
        if _DEBUG and penalty.numel()>1:
            raise RuntimeError(f"The penalty term should be a single scalar value, but is of shape {list(penalty.shape)}.")
        return anomaly_scores + penalty
    
    
    def set_one_class_radius(self, r: float) -> None:
        """Sets the radius for the one-class Deep SVDD.
        
        Arguments:
            `r` (`float`): The radius to be set.
        """
        self.__r_one_class = torch.tensor([r], device=self.__device)
        return
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the anomaly scores for the input samples. This is equivalent to calling `anomaly_score(x)`.
        
        Arguments:
            `x` (`torch.Tensor`): The input tensor of shape `(batch_size, ...)`.
        Returns:
            `torch.Tensor`: The computed anomaly scores of shape `(batch_size, 1)`.
        """
        return self.anomaly_score(x)
    
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict whether the input samples are anomalies. `x[i]` is `0` if it is normal, `1` otherwise."""
        _cfg = {'size': [1], 'device': x.device}
        zeros   = torch.zeros(**_cfg)
        ones    = torch.ones( **_cfg)
        score   = self.anomaly_score(x)
        if self.__is_soft:
            return torch.where(score<=0.0, zeros, ones)
        else:
            return torch.where(score<=self.__r_one_class.pow(2), zeros, ones)
    
    
    @property
    def nu(self) -> float:  return self.__nu
    @property
    def C(self) -> float:   return self.__C
    @property
    def is_soft_boundary(self) -> bool: return self.__is_soft
    @property
    def soft_radius(self) -> float:         return self.r_soft.item()
    @property
    def one_class_radius(self) -> float:    return self.__r_one_class.item()
    @property
    def device(self) -> torch.device:   return self.__device