import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
from federated_learning.schedulers import MinCapableStepLR
import os
import numpy as np
import copy


class Client:
    def __init__(self, args, client_idx: int, train_data_loader: torch.utils.data.DataLoader, test_data_loader: torch.utils.data.DataLoader):
        """
        :param args: Experiment arguments
        :param client_idx: Client index
        :param train_data_loader: Training data loader
        :param test_data_loader: Test data loader
        """
        self.args = args
        self.client_idx = client_idx
        self.device = self.initialize_device()
        self.set_net(self.load_default_model())
        self.loss_function = self.args.get_loss_function()()

        # Optimizer and Scheduler
        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=self.args.get_learning_rate(),
            betas=(self.args.get_beta1(), self.args.get_beta2()),
            eps=self.args.get_eps()
        )
        self.scheduler = MinCapableStepLR(
            self.args.get_logger(),
            self.optimizer,
            self.args.get_scheduler_step_size(),
            self.args.get_scheduler_gamma(),
            self.args.get_min_lr()
        )

        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader

    def initialize_device(self) -> torch.device:
        """Creates appropriate torch device for client operation."""
        return torch.device("cuda:0" if torch.cuda.is_available() and self.args.get_cuda() else "cpu")

    def set_net(self, net: torch.nn.Module):
        """Set the client's neural network."""
        self.net = net.to(self.device)

    def load_default_model(self) -> torch.nn.Module:
        """Load a default model from a file."""
        model_class = self.args.get_net()
        default_model_path = os.path.join(self.args.get_default_model_folder_path(), model_class.__name__ + ".model")
        return self.load_model_from_file(default_model_path)

    def load_model_from_file(self, model_file_path: str) -> torch.nn.Module:
        """Load a model from the specified file."""
        model_class = self.args.get_net()
        model = model_class()
        if os.path.exists(model_file_path):
            try:
                model.load_state_dict(torch.load(model_file_path))
            except Exception as e:
                self.args.get_logger().warning(
                    f"Couldn't load model. Error: {e}. Mapping CUDA tensors to CPU."
                )
                model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
        else:
            self.args.get_logger().warning(f"Model file not found: {model_file_path}")
        return model

    def get_client_index(self) -> int:
        """Return the client index."""
        return self.client_idx

    def get_nn_parameters(self) -> dict:
        """Return the NN's parameters."""
        return self.net.state_dict()

    def update_nn_parameters(self, new_params: dict):
        """Update the NN's parameters."""
        self.net.load_state_dict(copy.deepcopy(new_params), strict=True)

    def train(self, epoch: int) -> float:
        """Train the client for one epoch."""
        self.net.train()
        if self.args.should_save_model(epoch):
            self.save_model(epoch, self.args.get_epoch_save_start_suffix())

        running_loss = 0.0
        for i, (inputs, labels) in enumerate(self.train_data_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if i % self.args.get_log_interval() == 0:
                avg_loss = running_loss / self.args.get_log_interval()
                self.args.get_logger().info(f"[Epoch {epoch}, Step {i}] Loss: {avg_loss:.3f}")
                running_loss = 0.0

        self.scheduler.step()
        if self.args.should_save_model(epoch):
            self.save_model(epoch, self.args.get_epoch_save_end_suffix())

        return running_loss

    def save_model(self, epoch: int, suffix: str):
        """Save the model to a file."""
        save_path = self.args.get_save_model_folder_path()
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, f"model_{self.client_idx}_{epoch}_{suffix}.model")
        torch.save(self.get_nn_parameters(), file_path)

    def calculate_class_precision(self, confusion_mat: np.ndarray) -> np.ndarray:
        """Calculate precision for each class from the confusion matrix."""
        return np.divide(
            np.diagonal(confusion_mat),
            np.sum(confusion_mat, axis=0),
            out=np.zeros_like(np.diagonal(confusion_mat), dtype=float),
            where=np.sum(confusion_mat, axis=0) != 0
        )

    def calculate_class_recall(self, confusion_mat: np.ndarray) -> np.ndarray:
        """Calculate recall for each class from the confusion matrix."""
        return np.divide(
            np.diagonal(confusion_mat),
            np.sum(confusion_mat, axis=1),
            out=np.zeros_like(np.diagonal(confusion_mat), dtype=float),
            where=np.sum(confusion_mat, axis=1) != 0
        )

    def test(self):
        """Test the model and log the results."""
        self.net.eval()
        correct, total, loss = 0, 0, 0.0
        targets_, pred_ = [], []

        with torch.no_grad():
            for images, labels in self.test_data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.net(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                targets_.extend(labels.cpu().numpy())
                pred_.extend(predicted.cpu().numpy())
                loss += self.loss_function(outputs, labels).item()

        accuracy = 100 * correct / total
        confusion_mat = confusion_matrix(targets_, pred_)
        class_precision = self.calculate_class_precision(confusion_mat)
        class_recall = self.calculate_class_recall(confusion_mat)

        self.args.get_logger().info(f"Test Accuracy: {accuracy:.2f}%")
        self.args.get_logger().info(f"Test Loss: {loss:.4f}")
        self.args.get_logger().info(f"Classification Report:\n{classification_report(targets_, pred_)}")
        self.args.get_logger().info(f"Confusion Matrix:\n{confusion_mat}")
        self.args.get_logger().info(f"Class Precision: {class_precision}")
        self.args.get_logger().info(f"Class Recall: {class_recall}")

        return accuracy, loss, class_precision, class_recall
