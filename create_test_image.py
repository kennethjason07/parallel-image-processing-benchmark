import os
import cv2
import numpy as np

def generate_test_image():
    os.makedirs("data", exist_ok=True)
    height, width = 2160, 3840  # 4K UHD
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # 1. Background gradient
    for y in range(height):
        r_val = int(255 * y / height)
        g_val = int(255 * (height - y) / height)
        b_val = 128
        img[y, :] = [b_val, g_val, r_val]

    # 2. Geometric shapes for sharp edge detection
    cv2.circle(img, (1920, 1080), 750, (255, 255, 255), 25)
    cv2.circle(img, (1920, 1080), 500, (0, 0, 255), 20)
    cv2.circle(img, (1920, 1080), 250, (0, 255, 0), 15)
    
    cv2.rectangle(img, (400, 300), (1400, 1100), (255, 255, 0), 18)
    cv2.rectangle(img, (2400, 300), (3400, 1100), (0, 255, 255), 18)

    # 3. High-contrast grid lines
    for x in range(0, width, 200):
        cv2.line(img, (x, 0), (x, height), (200, 200, 200), 2)
    for y in range(0, height, 200):
        cv2.line(img, (0, y), (width, y), (200, 200, 200), 2)

    # 4. Text overlays
    cv2.putText(img, "PARALLEL COMPUTING CASE STUDY", (500, 1600),
                cv2.FONT_HERSHEY_SIMPLEX, 3.5, (255, 255, 255), 8)
    cv2.putText(img, "Sequential vs OpenMP vs CUDA Benchmark", (600, 1750),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 5)
    cv2.putText(img, "Resolution: 3840 x 2160 (8,294,400 Pixels)", (750, 1880),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)

    output_path = "data/high_resolution_image.jpg"
    cv2.imwrite(output_path, img)
    print(f"Test image successfully generated at {output_path}")
    print(f"Dimensions: {width} x {height} ({width * height:,} pixels)")
    print(f"File size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    generate_test_image()
