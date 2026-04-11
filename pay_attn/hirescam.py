import numpy as np
import torch

from pay_attn import model_outputs


class HiResCam():
    """
    Inference layer adapted from https://github.com/rachellea/hirescam, Draelos & Carin, 2021 (arXiv:2011.08891)
    Context: CNNs compute gradients :math:`K \\in \\mathbb{R}^{m \\times m}`

    Methodology: HiResCAM uses gradients computed by the CNN to produce a heatmap :math:`L^c_{HiResCAM}
    \\in \\mathbb{R}^{d \\times h \\times w}` that highlights important regions in the input volume for a given class :math:`c`.

    See test.py for example usage.
    """
    def __init__(self, model, device, model_name, target_layer_name):
        """
        :param model: Trained PcrCNN model.
        :param device: The device to run inference on (e.g. 'cuda', 'mps', or 'cpu').
        :param model_name: name of the model architecture, used to select the correct ModelOuputs class.
        :param target_layer_name: Index of the encoder layer to extract activations and gradients from.
                For PcrCNN, this should be '3' to select the last ConvBlock in the encoder.
        """
        self.model = model
        self.model.eval()
        self.modeloutputsclass = model_outputs.return_model_outputs_class(model_name)
        self.device = device
        self.model_name = model_name
        self.target_layer_name = target_layer_name

    def return_explanation(self, ctvol, chosen_label_index):
        """
        Compute the HiResCam heatmap for a single input volume.

        :param ctvol:  Input volume of the shape e.g (1, 3, 32, 256, 256). The caller is responsible for ensuring this is the case.
        :param chosen_label_index: Index of the class for which to compute the heatmap.
            - 1 for regions that increase logit. (evidence FOR pCR)
            - 0 for regions that decrease logit. (evidence AGAINST pCR)
        :return: Raw CAM volume of shape (1,D,H,W), where D, H, W are the spatial dimensions of the target layer's output..
        The caller is responsible for upsampling and visualizing this heatmap as desired.
        """
        # obtain gradients and activations:
        extractor = self.modeloutputsclass(self.model, self.target_layer_name)
        self.all_target_activs_dict, output = extractor.run_model(ctvol)

        scalar = output.squeeze()
        if chosen_label_index == 0:
            scalar = -scalar

        self.model.zero_grad()
        scalar.backward(retain_graph=True)

        # grads_list is a list of gradients, for each of the target layers.
        # Hooks are registered when we do the backward pass, which is why
        # we needed to wait until after calling backward() to get the
        # gradients.
        self.all_grads_dict = extractor.get_gradients()

        # Select gradients and activations for the target layer:
        target_grads = self.all_grads_dict[self.target_layer_name].cpu().data.numpy()
        target_activs = self.all_target_activs_dict[
            self.target_layer_name].cpu().data.numpy()


        return self.hirescam(target_grads, target_activs)

    def hirescam(self, target_grads: np.ndarray, target_activs: np.ndarray) -> np.ndarray:
        """
        Compute the HiResCAM heatmap using the formula:
            :math:`L^c_{\\text{HiResCAM}} = \\sum_k \\frac{\\partial Y^c}{\\partial A^k} \\odot A^k`


        :param target_grads: Gradients at the target layer, shape (batch, num_features, D', H', W')
        :param target_activs: Activations at the target layer, shape (batch, num_features, D', H', W')
        :return: np.ndarray: Heatmap of shape (batch, D, H, W). NOTE that heatmap renderer is left to the caller.
        """
        raw_cam_volume = np.multiply(target_grads, target_activs)
        raw_cam_volume = np.sum(raw_cam_volume, axis=1)
        return raw_cam_volume

