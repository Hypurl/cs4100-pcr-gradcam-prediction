import torch
import torch.nn as nn


def return_model_outputsclass(model_name):
    if model_name == 'PcrCNN':
        return ModelOutputsPcrCNN
    else:
        assert False, 'Invalid model_name'

class ModelOutputsPcrCNN():
    """Class for running a PcrCNN <model> and:
       (1) Extracting activations from intermediate target layers
       (2) Extracting gradients from intermediate target layers
       (3) Returning the final model output"""

    def __init__(self, model, target_layer_name:str):
        self.model = model
        self.target_layer_name = target_layer_name
        #Dict where the key is the name and the value is the gradient (hook)
        self.gradients = []
        self.gradient_names = []
        self.verbose = False

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def get_gradients(self):
        gradients_dict = {}
        for idx in range(len(self.gradient_names)):
            name = self.gradient_names[idx]
            grad = self.gradients[idx]
            gradients_dict[name] = grad
        return gradients_dict

    def run_model(self, x):
       """TODO: Implement run model for PcrCNN. This should run the model self.model on the input <x>, returning activations and output."""