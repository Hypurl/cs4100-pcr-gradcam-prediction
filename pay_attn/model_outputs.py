from train import PcrCNN


def return_model_outputs_class(model_name):
    if model_name == 'PcrCNN':
        return ModelOutputsPcrCNN
    else:
        assert False, 'Invalid model_name'


class ModelOutputsPcrCNN():
    """Class for running a PcrCNN <model> and:
       (1) Extracting activations from intermediate target layers
       (2) Extracting gradients from intermediate target layers
       (3) Returning the final model output"""

    def __init__(self, model: PcrCNN, target_layer_name: str):
        self.model = model
        self.target_layer_name = target_layer_name
        # Dict where the key is the name and the value is the gradient (hook)
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
        """
        <p>
        Runs the model on input x, while saving activations and gradients from the target layer.
        </p>
        :param x: Input x is expected for (batch, 3, 32, 256, 256). The caller is responsible for ensuring this is the case.
        :return:
        """

        assert list(x.shape)[1:] == [3, 32, 256, 256]
        activations = {}

        encoder = self.model.encoder._modules.items()

        for name, module in encoder:
            x = module(x)
            if name == self.target_layer_name:
                activations[name] = x.cpu().data
                x.register_hook(self.save_gradient)
                self.gradient_names.append(name)

        x = self.model.classifier(x)

        return activations, x
