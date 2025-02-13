import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt

# Set up Tesseract path (update this to your Tesseract installation path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image_path):
    """Preprocess the image: grayscale, thresholding, and edge detection."""
    # Load image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur and thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Detect edges
    edges = cv2.Canny(binary, 50, 150)
    return image, edges

def detect_shapes(edges):
    """Detect shapes using contour detection."""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Classify the shape based on the number of vertices
        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            # Check if it's a rectangle or square
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            shape = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
        elif len(approx) == 5:
            shape = "Pentagon"
        else:
            shape = "Circle"
        
        shapes.append((shape, approx))
    return shapes

def detect_text(image):
    """Detect text using OCR."""
    text = pytesseract.image_to_string(image)
    return text.strip()

def reconstruct_diagram(image, shapes, text):
    """Reconstruct the diagram by drawing shapes and text."""
    output_image = image.copy()
    for shape, approx in shapes:
        cv2.drawContours(output_image, [approx], -1, (0, 255, 0), 2)
        # Label the shape
        M = cv2.moments(approx)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(output_image, shape, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Add detected text
    cv2.putText(output_image, f"Detected Text: {text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return output_image

def main(image_path):
    # Step 1: Preprocess the image
    image, edges = preprocess_image(image_path)
    
    # Step 2: Detect shapes
    shapes = detect_shapes(edges)
    
    # Step 3: Detect text
    text = detect_text(image)
    
    # Step 4: Reconstruct the diagram
    output_image = reconstruct_diagram(image, shapes, text)
    
    # Display the result
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title("Reconstructed Diagram")
    plt.axis("off")
    plt.show()

# Run the prototype
if __name__ == "__main__":
    image_path = "sketch.jpg"  # Replace with your hand-drawn sketch file
    main(image_path)