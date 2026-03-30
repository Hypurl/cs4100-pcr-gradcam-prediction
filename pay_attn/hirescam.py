import numpy as np
import torch, torch.nn as nn
import model_outputs
"""
Inference layer using https://github.com/rachellea/hirescam

Context: CNNs compute gradients :math:`K \in \mathbb{R}^{m \\times m}`

Methodology: HiResCAM uses weighted gradients computed by the CNN


"""

class HiResCam():
    def __init__(self, model, device, label_meanings, model_name, target_layer_name):
        self.model = model
        self.model.eval()
        self.modeloutsputsclass = model_outputs.return_model_outputsclass(model_name)
        self.device = device
        self.label_meanings = label_meanings  # all the abnormalities IN ORDER
        self.model_name = model_name
        self.target_layer_name = target_layer_name

    def return_explanation(self, ctvol, gr_truth, volume_acc, chosen_label_index):

        # obtain gradients and activations:
        extractor = self.modeloutputsclass(self.model, self.target_layer_name)
        self.all_target_activs_dict, output = extractor.run_model(ctvol)


        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][chosen_label_index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True).to(self.device)
        one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

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

    @staticmethod
    def hirescam(self, target_grads, target_activs):
        raw_cam_volume = np.multiply(target_grads, target_activs)
        raw_cam_volume = np.sum(raw_cam_volume, axis=1) #sum over feature dimension; out shape [134, 6, 6]
        return raw_cam_volume

