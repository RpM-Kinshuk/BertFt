from PIL import Image
import os

import cv2

# Define the common dimensions to which you want to resize your images
common_width = 1920
common_height = 1080

# Iterate through the input directory
def resize(input_directory, output_directory):
        if not os.path.exists(output_directory):
                os.makedirs(output_directory)
        for filename in os.listdir(input_directory):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                        # Read the image
                        input_path = os.path.join(input_directory, filename)
                        img = cv2.imread(input_path) #type: ignore

                        # Resize the image to the common dimensions
                        img = cv2.resize(img, (common_width, common_height)) #type: ignore

                        # Perform image processing here (e.g., apply filters, manipulate colors)

                        # Save the processed image to the output directory
                        output_path = os.path.join(output_directory, filename)
                        cv2.imwrite(output_path, img) #type: ignore

# Print a message indicating the processing is complete
print("Image processing and resizing completed.")

tasks = ['stsb']

def combine(task):
        input_dir = f"{task}/"
        output_dir = f"{task}_mod/"
        resize(input_dir, output_dir)
        ep1 = Image.open(f"{task}_mod/ep1.png")
        big_alpha = Image.open(f"{task}_mod/large_alph.png")
        ep2 = Image.open(f"{task}_mod/ep2.png")
        smol_alpha = Image.open(f"{task}_mod/small_alph.png")
        width, height = big_alpha.size  # Assuming all images have the same dimensions
        collage_width = width * 2
        collage_height = height * 2
        collage = Image.new("RGB", (collage_width, collage_height))
        collage.paste(ep1, (0, 0))
        collage.paste(big_alpha, (width, 0))
        collage.paste(ep2, (0, height))
        collage.paste(smol_alpha, (width, height))
        collage.save(f"{task}.png")
        
for task in tasks:
        combine(task)