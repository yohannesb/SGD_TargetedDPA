import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
from federated_learning.schedulers import MinCapableStepLR
import os
import numpy as np
import copy


class Client:

    def __init__(self, args, client_idx, train_data_loader, test_data_loader):
        """
        Initialize the client with its data, model, and training parameters.
        """
        self.args = args
        self.client_idx = client_idx

        self.device = self.initialize_device()
        self.set_net(self.load_default_model())

        self.loss_function = self.args.get_loss_function()()

        # Initialize Adam optimizer
        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=self.args.get_learning_rate(),
            betas=(self.args.get_beta1() if hasattr(self.args, 'get_beta1') else 0.9,
                   self.args.get_beta2() if hasattr(self.args, 'get_beta2') else 0.999),
            eps=self.args.get_eps() if hasattr(self.args, 'get_eps') else 1e-8
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

    def initialize_device(self):
        """
        Creates appropriate torch device for client operation.
        """
        return torch.device("cuda:0" if torch.cuda.is_available() and self.args.get_cuda() else "cpu")

    def set_net(self, net):
        """
        Set the client's neural network model.
        """
        self.net = net.to(self.device)

    def load_default_model(self):
        """
        Load the default model from file.
        """
        model_class = self.args.get_net()
        default_model_path = os.path.join(self.args.get_default_model_folder_path(), f"{model_class.__name__}.model")
        return self.load_model_from_file(default_model_path)

    def load_model_from_file(self, model_file_path):
        """
        Load a model from a file.
        """
        model_class = self.args.get_net()
        model = model_class()
        if os.path.exists(model_file_path):
            try:
                model.load_state_dict(torch.load(model_file_path))
            except RuntimeError:
                self.args.get_logger().warning("Mapping CUDA tensors to CPU to resolve error.")
                model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
        else:
            self.args.get_logger().warning(f"Could not find model: {model_file_path}")
        return model
    def get_client_index(self):
        """
        Returns the client index.
        """
        return self.client_idx

    def get_nn_parameters(self):
        """
        Return the NN's parameters.
        """
        return self.net.state_dict()

    def update_nn_parameters(self, new_params):
        """
        Update the NN's parameters.

        :param new_params: New weights for the neural network
        :type new_params: dict
        """
        self.net.load_state_dict(copy.deepcopy(new_params), strict=True)


    def train(self, epoch):
        """
        Train the model for one epoch.
        """
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
            if i % max(1, self.args.get_log_interval()) == 0:
                avg_loss = running_loss / self.args.get_log_interval()
                self.args.get_logger().info(f"Epoch [{epoch}], Step [{i}], Loss: {avg_loss:.3f}")
                running_loss = 0.0

        self.scheduler.step()
        if self.args.should_save_model(epoch):
            self.save_model(epoch, self.args.get_epoch_save_end_suffix())

        return running_loss

    def save_model(self, epoch, suffix):
        """
        Save the model to file.
        """
        self.args.get_logger().debug(f"Saving model: Epoch {epoch}")
        os.makedirs(self.args.get_save_model_folder_path(), exist_ok=True)
        save_path = os.path.join(
            self.args.get_save_model_folder_path(),
            f"model_{self.client_idx}_{epoch}_{suffix}.model"
        )
        torch.save(self.get_nn_parameters(), save_path)

    def calculate_class_precision(self, confusion_mat):
        """
        Calculate class-wise precision with division safety.
        """
        return np.diagonal(confusion_mat) / np.maximum(1, np.sum(confusion_mat, axis=0))

    def calculate_class_recall(self, confusion_mat):
        """
        Calculate class-wise recall with division safety.
        """
        return np.diagonal(confusion_mat) / np.maximum(1, np.sum(confusion_mat, axis=1))

    def test(self):
        """
        Evaluate the model on the test set.
        """
        self.net.eval()
        correct, total = 0, 0
        targets_, pred_ = [], []
        loss = 0.0

        with torch.no_grad():
            for images, labels in self.test_data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                targets_.extend(labels.cpu().numpy())
                pred_.extend(predicted.cpu().numpy())
                loss += self.loss_function(outputs, labels).item()

        accuracy = 100 * correct / total
        confusion_mat = confusion_matrix(targets_, pred_)

        class_precision = self.calculate_class_precision(confusion_mat)
        class_recall = self.calculate_class_recall(confusion_mat)

        self.args.get_logger().debug(f"Accuracy: {accuracy:.2f}%")
        self.args.get_logger().debug(f"Loss: {loss:.3f}")
        self.args.get_logger().debug("Classification Report:\n" + classification_report(targets_, pred_, zero_division=0))
        self.args.get_logger().debug(f"Confusion Matrix:\n{confusion_mat}")

        return accuracy, loss, class_precision, class_recall
