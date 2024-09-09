from torchmetrics import Accuracy, MeanSquaredError, Precision, Recall


class AccuracyMetric(Accuracy):
    def __init__(self, num_classes: int, threshold: float = 0.5, average: str = "micro") -> None:
        super().__init__(num_classes=num_classes, threshold=threshold, average=average)

    def __call__(self, y_pred, y_true):
        """Compute the accuracy classification score.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values.
        y_true : torch.Tensor
            True values.

        Returns
        -------
        accuracy : torch.Tensor
            Accuracy.
        """
        y_pred = y_pred.float().argmax(dim=-1)
        y_true = y_true.float()
        return super().__call__(y_pred, y_true)


class PrecisionMetric(Precision):
    def __init__(self, num_classes: int, threshold: float = 0.5, average: str = "micro") -> None:
        super().__init__(num_classes=num_classes, threshold=threshold, average=average)

    def __call__(self, y_pred, y_true):
        """Compute the precision classification score.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values.
        y_true : torch.Tensor
            True values.

        Returns
        -------
        precision : torch.Tensor
            Precision.
        """
        y_pred = y_pred.float().argmax(dim=-1)
        y_true = y_true.float()
        return super().__call__(y_pred, y_true)


class RecallMetric(Recall):
    def __init__(self, num_classes: int, threshold: float = 0.5, average: str = "micro") -> None:
        super().__init__(num_classes=num_classes, threshold=threshold, average=average)

    def __call__(self, y_pred, y_true):
        """Compute the recall classification score.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted values.
        y_true : torch.Tensor
            True values.

        Returns
        -------
        recall : torch.Tensor
            Recall.
        """
        y_pred = y_pred.float().argmax(dim=-1)
        y_true = y_true.float()
        return super().__call__(y_pred, y_true)


class MeanSquaredErrorMetric(MeanSquaredError):
    def __init__(self, squared: bool = True) -> None:
        super().__init__(squared=squared)

    def __call__(self, y_pred, y_true):
        """
        Compute the mean squared error regression score

        Args:
            y_pred (torch.Tensor): Predicted values
            y_true (torch.Tensor): True values
        Returns:
            torch.Tensor: Mean squared error
        """
        y_pred = y_pred.float()
        y_true = y_true.float().unsqueeze(-1)

        return super().__call__(y_pred, y_true)
