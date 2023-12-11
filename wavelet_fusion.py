import cv2
import numpy as np
import pywt

def wavelet_fusion(rgb_image_path, infrared_image_path, output_path):
    # Read the input images
    rgb_image = cv2.imread(rgb_image_path)
    infrared_image = cv2.imread(infrared_image_path, cv2.IMREAD_GRAYSCALE)

    # Resize the infrared image to match the size of the RGB image
    infrared_image_resized = cv2.resize(infrared_image, (rgb_image.shape[1], rgb_image.shape[0]))

    # Convert images to floating point for wavelet transform
    rgb_image_float = rgb_image.astype(np.float32) / 255.0
    infrared_image_float = infrared_image_resized.astype(np.float32) / 255.0

    # Apply wavelet transform to each image
    coeffs_rgb = pywt.dwt2(rgb_image_float, 'haar')
    coeffs_infrared = pywt.dwt2(infrared_image_float, 'haar')

    # Combine high-frequency components and low-frequency components
    fused_coeffs = (
        (coeffs_rgb[0] + coeffs_infrared[0]) / 2,
        (coeffs_rgb[1] + coeffs_infrared[1]) / 2,
        (coeffs_rgb[2] + coeffs_infrared[2]) / 2
    )

    # Inverse wavelet transform to obtain the fused image
    fused_image = pywt.idwt2(fused_coeffs, 'haar')

    # Clip values to the valid range [0, 1]
    fused_image = np.clip(fused_image, 0, 1)

    # Convert back to uint8 for saving
    fused_image_uint8 = (fused_image * 255).astype(np.uint8)

    # Save the fused image
    cv2.imwrite(output_path, fused_image_uint8)
    print(f"Fused image saved to {output_path}")

if __name__ == "__main__":
    # Replace 'path/to/rgb/image.jpg' and 'path/to/infrared/image.jpg' with your file paths
    rgb_image_path = 'path/to/rgb/image.jpg'
    infrared_image_path = 'path/to/infrared/image.jpg'
    
    # Replace 'path/to/output/fused_image.jpg' with your desired output path
    output_path = 'path/to/output/fused_image.jpg'

    wavelet_fusion(rgb_image_path, infrared_image_path, output_path)
