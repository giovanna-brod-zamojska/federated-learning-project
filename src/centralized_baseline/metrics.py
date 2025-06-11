from torchmetrics.classification import (
    MulticlassPrecision,
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassRecall,
)


class Metrics:
    """
    A class to compute accuracy, precision, recall and f1 score given the logits and the targets.
    """

    def __init__(self, device, num_classes: int):
        self.device = device

        # Macro metrics

        self.precision_macro = MulticlassPrecision(
            num_classes=num_classes, average="macro"
        ).to(self.device)

        self.recall_macro = MulticlassRecall(
            num_classes=num_classes, average="macro"
        ).to(self.device)

        self.f1_macro = MulticlassF1Score(num_classes=num_classes, average="macro").to(
            self.device
        )

        # Micro metrics

        self.precision_micro = MulticlassPrecision(
            num_classes=num_classes, average="micro"
        ).to(self.device)

        self.recall_micro = MulticlassRecall(
            num_classes=num_classes, average="micro"
        ).to(self.device)

        self.f1_micro = MulticlassF1Score(num_classes=num_classes, average="micro").to(
            self.device
        )

        # Accuracy

        self.accuracy = MulticlassAccuracy(num_classes=num_classes, average="micro").to(
            self.device
        )

    def update(self, preds, target) -> dict:
        """
        Update metrics with new predictions and targets.

        Args:
            preds (torch.Tensor): Model predictions (class indices)
            target (torch.Tensor): Ground truth labels
        """

        # Ensure predictions and targets are on the correct device
        preds = preds.to(self.device)
        target = target.to(self.device)

        # Update all metrics
        self.precision_macro.update(preds, target)
        self.recall_macro.update(preds, target)
        self.f1_macro.update(preds, target)

        self.precision_micro.update(preds, target)
        self.recall_micro.update(preds, target)
        self.f1_micro.update(preds, target)

        self.accuracy.update(preds, target)

        return self.get_metrics()

    def get_metrics(self) -> dict:
        """Get the current metrics."""

        return {
            "precision_macro": self.precision_macro.compute(),
            "recall_macro": self.recall_macro.compute(),
            "f1_macro": self.f1_macro.compute(),
            "precision_micro": self.precision_micro.compute(),
            "recall_micro": self.recall_micro.compute(),
            "f1_micro": self.f1_micro.compute(),
            "accuracy": self.accuracy.compute(),
        }
