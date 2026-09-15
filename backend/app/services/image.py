import cv2
import numpy as np
from PIL import Image

def apply_optimal_preprocessing(pil_img: Image.Image) -> Image.Image:
    """
    Applies the optimal preprocessing pipeline used in the benchmark notebook:
    1. Bilateral Filter for texture denoising while preserving edges.
    2. CLAHE (Contrast Limited Adaptive Histogram Equalization) on LAB color space for lighting balance.
    """
    # Convert PIL Image (RGB) to OpenCV format (BGR)
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # 1. Bilateral Filter
    denoised_bgr = cv2.bilateralFilter(img_bgr, d=7, sigmaColor=50, sigmaSpace=50)
    
    # 2. CLAHE
    lab = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    clahe_bgr = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    
    # Convert back to PIL Image (RGB)
    return Image.fromarray(cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2RGB))
