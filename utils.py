import cv2
import numpy as np

def format_confidence(conf):
    return round(conf.item() * 100, 2)

def freshness_score(conf, label):
    if label == "Fresh":
        return int(conf)
    elif label == "Medium":
        return int(conf * 0.7)
    else:
        return int(conf * 0.4)

def overlay_heatmap(image, cam):
    image = image.squeeze().permute(1, 2, 0).numpy()
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = heatmap / 255 + image
    overlay = overlay / overlay.max()
    return np.uint8(255 * overlay)

def save_image(img, path):
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return path