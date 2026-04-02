import torch
import cv2
import numpy as np

def generate_gradcam(model, img, filepath):
    img = np.array(img)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
    img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)  # Shape: [1,3,224,224]

    img_tensor.requires_grad_(True)

    output = model(img_tensor)
    class_idx = output.argmax(dim=1).item()  # Get predicted class index

    model.zero_grad()
    output[0, class_idx].backward()  # Backprop on predicted class score

    gradients = img_tensor.grad[0].data.numpy()  # Gradients [3,H,W]
    weights = np.mean(gradients, axis=(1, 2))  # Global average pooling over H,W -> [3]
    
    # Use first conv layer activations (assuming model has features)
    activations = img_tensor[0].data.numpy()  # Use input as proxy activation [3,H,W]
    heatmap = np.zeros((224, 224), dtype=np.float32)
    for c in range(3):
        heatmap += weights[c] * activations[c]
    
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Convert img to BGR for overlay consistency
    img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    superimposed = heatmap * 0.4 + img_bgr * 0.6  # Proper weighted overlay

    output_path = filepath.replace(".jpg", "_cam.jpg").replace(".png", "_cam.png")
    cv2.imwrite(output_path, superimposed)

    return output_path

