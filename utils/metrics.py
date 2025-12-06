import  torch


__all__: list[str] = ['roc_auc_score']


def roc_auc_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    from    sklearn.metrics    import  roc_auc_score    as  sk_roc_auc_score
    return sk_roc_auc_score(target.cpu().numpy(), pred.cpu().numpy())