from PIL import Image
import numpy as np
import os

## Convert sky to white pixels

def preprocess_image(input_path, output_path):
    # Define the color map
    color_map = {
        "background": (184, 61, 245),
        "road": (61, 61, 245),
        "marking": (221, 255, 51),
        "vehicle": (255, 204, 51),
        "road_sign": (255, 53, 94)
    }

    # Open the image
    img = Image.open(input_path).convert('RGB')  # Convert image to RGB
    
    # Convert image to numpy array
    img_array = np.array(img)

    # Create a mask for pixels that match any of the specified colors
    mask = np.zeros(img_array.shape[:2], dtype=bool)
    for color in color_map.values():
        mask |= np.all(img_array == np.array(color), axis=2)

    # Invert the mask to get pixels that don't match any specified color
    inv_mask = ~mask

    # Set non-matching pixels to white
    img_array[inv_mask] = [255, 255, 255]

    # Create a new image from the processed array
    processed_img = Image.fromarray(img_array)

    # Save the processed image
    processed_img.save(output_path)

    print(f"Processed image saved to {output_path}")

# Function to process all images in the given directory
def process_directory(input_dir, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{filename}")
            
            try:
                preprocess_image(input_path, output_path)
                print(f"Successfully processed {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")


input_directory = "/Users/casseyyimei/Documents/GitHub/RoadSegmentaion/Dataset/test_labels1"
output_directory = "/Users/casseyyimei/Documents/GitHub/RoadSegmentation/Dataset/test_labels"

process_directory(input_directory, output_directory)

