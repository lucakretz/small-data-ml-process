"""Implementation of the model using PyTorch Lightning"""

from typing import Dict
from typing import List
from typing import NoReturn
from typing import Tuple

from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
from pl_bolts.models import LogisticRegression

from sklearn.metrics import silhouette_score

import torch
from torch import nn
from torch.nn.modules.loss import TripletMarginLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import SGD
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from torchvision.models import resnet50


class EncoderModule(LightningModule):  # pylint: disable=too-many-ancestors
    """Model for feature encoding as a basis for specific
    downstream tasks.
    """

    def __init__(self, *args, **kwargs) -> NoReturn:
        super().__init__()
        self.save_hyperparameters()

        self.triplet_loss = TripletMarginLoss(float(self.hparams.margin))

        if self.hparams.pretrained:
            model = resnet50(pretrained=True)
            self.hparams.head_input_size = model.fc.in_features
            self.backbone = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        else:
            self.backbone = self._get_vanilla_cnn()

        self.head = self._get_head()

    def _get_vanilla_cnn(self) -> nn.Sequential:
        """Construction of the CNN used for feature extraction.

        Returns:
            nn.Sequential: CNN build from scratch.
        """

        vanilla_cnn = nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.Flatten(),
        )

        return vanilla_cnn

    def _get_head(self) -> nn.Sequential:
        """Construction of the model head.
        It translates the extracted features into a vector.

        Returns:
            nn.Sequential: Fully connected head.
        """
        if self.hparams.pretrained:
            cnn_features = self.hparams.head_input_size
        else:
            cnn_features = self._calculate_features_cnn(self.backbone)

        head = nn.Sequential(
            nn.Linear(cnn_features, 256),
            nn.ReLU(),
            nn.Linear(256, self.hparams.out_dim),
        )
        return head

    def _calculate_features_cnn(self, network: nn.Module) -> int:
        """Calculates output features for custom CNN.
        Supported layers:
        - Conv2d
        - MaxPool2d

        Args:
            network (nn.Module): Built CNN network.

        Returns:
            int: Numbers of features to expect from CNN.
        """
        width = self.hparams.image_dim[0]
        height = self.hparams.image_dim[1]
        out_channels = 0
        for layer in network:
            if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
                if isinstance(layer, nn.Conv2d):
                    kernel_size_width = layer.kernel_size[0]
                    kernel_size_height = layer.kernel_size[1]

                    stride_width = layer.stride[0]
                    stride_height = layer.stride[1]

                    padding_width = layer.padding[0]
                    padding_height = layer.padding[1]

                    out_channels = layer.out_channels

                if isinstance(layer, nn.MaxPool2d):
                    kernel_size_width = kernel_size_height = layer.kernel_size
                    stride_width = stride_height = layer.stride
                    padding_width = padding_height = layer.padding

                width = int(
                    (width + 2 * padding_width - kernel_size_width) / stride_width + 1
                )
                height = int(
                    (height + 2 * padding_height - kernel_size_height) / stride_height
                    + 1
                )

        return out_channels * width * height

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward step of the encoding model.

        Args:
            x (torch.Tensor): Input batch of samples.

        Returns:
            torch.Tensor: Output logits of the model.
        """
        x = self.backbone(x)
        x = self.head(x)
        return x

    def training_step(
        self, train_batch: Tuple[List[torch.Tensor], torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Performs a forward and model optimization step on training batch.

        Args:
            train_batch (Tuple[List[torch.Tensor], torch.Tensor]): Input training batch.
            batch_idx (int): Default parameter.

        Returns:
            torch.Tensor: Loss value used for model optimization.
        """

        triple, _ = train_batch
        encodings = list(map(self, triple))
        loss = self.triplet_loss(*encodings)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(
        self, val_batch: Tuple[List[torch.Tensor], torch.Tensor], batch_idx: int
    ) -> None:
        """Checks on model performance using unseen data batch.

        Args:
            val_batch (Tuple[List[torch.Tensor], torch.Tensor]):
            Unseen data for calculating validation metrics.
            batch_idx (int): Default parameter."""

        triple, _ = val_batch
        encodings = list(map(self, triple))
        loss = self.triplet_loss(*encodings)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        """Calculates silhouette score as an indicator for the quality of the
        learned clusters on the whole validation data."""

        val_samples, val_labels = zip(
            *[(triple[0], label) for triple, label in self.val_dataloader()]
        )
        encodings = self(torch.cat(val_samples).to(self.device))
        with torch.no_grad():
            val_score = silhouette_score(
                encodings.cpu().numpy(), torch.cat(val_labels).cpu().numpy()
            )

        self.log("val_score", val_score, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> Dict[str, object]:
        """Set configuration for the optimization strategy.

        Returns:
            Dict[str, object]: Provides optimizer and
            learning rate scheduler for training.
        """
        optimizer = SGD(
            self.head.parameters() if self.hparams.pretrained else self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        lr_scheduler = StepLR(
            optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


class TaskModule(LightningModule):  # pylint: disable=too-many-ancestors
    """Model trained on top of an encoder output.
    The architecture of the model is task dependant.

    Args:
        encoder_path (str): Path to the .ckpt file
        for the previously trained encoder model.
    """

    def __init__(self, encoder_path: str, *args, **kwargs):

        super().__init__()
        self.save_hyperparameters()
        # load encoder model
        # and make it non-trainable
        self.encoder = EncoderModule.load_from_checkpoint(encoder_path)
        self.encoder.freeze()

        self.accuracy = Accuracy()
        self.criterion = CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)

        # select model by task
        if self.hparams.task == "classification":
            self.head = self._get_classifier()
        elif self.hparams.task == "fraud_analysis":
            self.head = self._get_classifier()
        elif self.hparams.task == "anomaly_detection":
            self.head = LogisticRegression(self.hparams.head_input_size, num_classes=2)

    def _get_classifier(self) -> nn.Sequential:
        """Construction of the classifier head.

        Returns:
            nn.Sequential: Fully connected network for
            multi class classification.
        """

        head = nn.Sequential(
            nn.Linear(self.hparams.head_input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.hparams.out_dim),
        )
        return head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward step through the encoder and the task head.

        Args:
            x (torch.Tensor): Batch of samples.

        Returns:
            torch.Tensor: Logits for the forward pass.
        """
        x = self.encoder(x)
        x = self.head(x)
        return x

    def training_step(
        self, train_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Performs a forward and model optimization step on training batch.

        Args:
            train_batch (Tuple[torch.Tensor, torch.Tensor]): Input training batch.
            batch_idx (int): Default parameter.

        Returns:
            torch.Tensor: Loss value used for model optimization.
        """

        data, labels = train_batch
        logits = self(data)
        loss = self.criterion(logits, labels)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(
        self, val_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Checks on model performance using unseen data batch.

        Args:
            val_batch (Tuple[torch.Tensor, torch.Tensor]):
            Unseen data for calculating validation metrics.
            batch_idx (int): Default parameter.
        """
        data, labels = val_batch
        logits = self(data)
        loss = self.criterion(logits, labels)
        prediction = torch.argmax(self.softmax(logits), axis=-1)
        val_score = self.accuracy(prediction, labels)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_score", val_score, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(
        self, test_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Checks on model performance using unseen data batch.

        Args:
            val_batch (Tuple[torch.Tensor, torch.Tensor]):
            Unseen data for calculating validation metrics.
            batch_idx (int): Default parameter.
        """
        data, labels = test_batch
        logits = self(data)
        loss = self.criterion(logits, labels)
        prediction = torch.argmax(self.softmax(logits), axis=-1)
        test_accuracy = self.accuracy(prediction, labels)

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "test_accuracy", test_accuracy, prog_bar=True, on_step=False, on_epoch=True
        )

    def configure_optimizers(self) -> Dict[str, object]:
        """Set configuration for the optimization strategy.

        Returns:
            Dict[str, object]: Provides optimizer and
            learning rate scheduler for training.
        """
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        lr_scheduler = StepLR(
            optimizer, step_size=self.hparams.step_size, gamma=self.hparams.gamma
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
