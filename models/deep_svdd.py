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
            encoder:    Optional[torch.nn.Module],
            nu:         float = 0.05,
            C:          float = 1.0,
            device:     Optional[torch.device] = None,
        ) -> Self:
        """The initializer for `DeepSVDD`.
        
        Arguments:
            `encoder` (`Optional[torch.nn.Module]`):
                The encoder network to be used in the Deep SVDD model. See **Remark** for the constraints of the encoder.
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
            for p in md_enc.named_parameters():
                if p.endswith("bias"):
                    raise AttributeError("Deep SVDD encoder should not contain bias terms.")
        
        if not (isinstance(nu, float) and nu>=0.0 and nu<1.0):
            raise ValueError("The hyperparameter nu should be in the range [0, 1).")
        if (not isinstance(C, float)) or C<0.0:
            raise ValueError("The penalty term should be a nonnegative real number.")

        self.__nu       = float(nu)
        self.__C        = float(C)
        self.__center   = ... # NOTE: The center of the hypersphere should not be trained
        self.radius     = torch.nn.Parameter(torch.randn((1,)))
        
        if device is None: device = torch.get_default_device()
        self.__device = device
        self.to(device)
        return
    
    
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the anomaly scores for the input samples. The score of an input sample `p` is defined as the norm of `encoder(p)-center`.
        
        Arguments:
            `x` (`torch.Tensor`): The input tensor.
        Returns:
            `torch.Tensor`: The computed anomaly scores.
        """
        dev: torch.Tensor = self.encoder.forward(x) - self.__center
        return dev.flatten(start_dim=1).norm(p=2, keepdim=True)
    
    
    def _compute_penalty(self, unit: bool=False) -> torch.Tensor:
        """Computes the penalty term, which is defined as the sum of the Frobenius norms of the weights of all layers in the encoder, multiplied by `C/2`.
        """
        penalty = torch.zeros((1,), device=self.__device)
        for p in self.encoder.parameters():
            penalty = penalty + torch.norm(p, p='fro').pow(2)
        penalty = penalty / 2
        if unit:    return penalty
        else:       return self.__C * penalty
    
    
    def compute_loss__soft_boundary(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the soft-boundary Deep SVDD loss.
        
        Arguments:
            `x` (`torch.Tensor`): The input tensor.
        
        Returns:
            `torch.Tensor`: The computed loss.
        """
        anomaly_score = self.anomaly_score(x).pow(2).mean()/self.__nu
        penalty = self._compute_penalty()
        if _DEBUG and anomaly_score.numel()>1:
            raise RuntimeError(f"The average anomaly score should be a single scalar value, but is of shape {list(anomaly_score.shape)}.")
        if _DEBUG and penalty.numel()>1:
            raise RuntimeError(f"The penalty term should be a single scalar value, but is of shape {list(penalty.shape)}.")
        return torch.clamp_min(anomaly_score, self.radius.pow(2)) + penalty
    
    
    def compute_loss__one_class(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the one-class Deep SVDD loss.
        
        Arguments:
            `x` (`torch.Tensor`): The input tensor.
        Returns:
            `torch.Tensor`: The computed loss.
        """
        anomaly_scores = self.anomaly_score(x).pow(2).mean()
        penalty = self._compute_penalty()
        if _DEBUG and anomaly_scores.numel()>1:
            raise RuntimeError(f"The average anomaly score should be a single scalar value, but is of shape {list(anomaly_scores.shape)}.")
        if _DEBUG and penalty.numel()>1:
            raise RuntimeError(f"The penalty term should be a single scalar value, but is of shape {list(penalty.shape)}.")
        return anomaly_scores + penalty
    
    @property
    def nu(self) -> float:      return self.__nu
    @property
    def C(self) -> float:       return self.__C
    @property
    def device(self) -> torch.device: return self.__device