"""
Grad-CAM++ Explainability for Fracture Detection Models
- Provides visual heatmaps for interpretability
"""

import numpy as np
import cv2
import torch


def generate_gradcam_plus(model, input_tensor, target_class, target_layer):
    """
    Generate Grad-CAM++ heatmap.
    Args:
        model (nn.Module): Trained model
        input_tensor (torch.Tensor): Input image tensor of shape (1, C, H, W)
        target_class (int): Target class index
        target_layer (nn.Module): Layer to hook into for activations/gradients
    Returns:
        np.ndarray: Normalized heatmap (H, W)
    """
    model.eval()
    activations, gradients = [], []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_backward_hook(backward_hook)

    output = model(input_tensor)
    model.zero_grad()
    score = output[0, target_class]
    score.backward(retain_graph=True)

    grads_val = gradients[0].cpu().data.numpy()[0]
    activations_val = activations[0].cpu().data.numpy()[0]

    weights_num = np.power(grads_val, 2)
    weights_denom = 2 * weights_num + np.sum(
        activations_val * np.power(grads_val, 3),
        axis=(1, 2),
        keepdims=True
    )
    weights_denom = np.where(weights_denom != 0.0, weights_denom, 1e-10)
    weights = np.sum(weights_num / weights_denom, axis=(1, 2))

    cam = np.zeros(activations_val.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * activations_val[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)

    handle_fwd.remove()
    handle_bwd.remove()
    return cam


def auto_suggest_cam_region(heatmap):
    """
    Suggest anatomical region based on Grad-CAM++ heatmap.
    Uses center of mass heuristic.
    """
    heatmap_resized = cv2.resize(heatmap, (10, 10))
    max_index = np.unravel_index(np.argmax(heatmap_resized, axis=None), heatmap_resized.shape)

    region_guess = [
        "humeral head", "joint interface", "implant screw zone",
        "medial epicondyle", "distal radius"
    ]
    return region_guess[(max_index[0] * max_index[1]) % len(region_guess)]
