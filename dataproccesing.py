import os
from PIL import Image

def convert_images_in_folder(input_folder, output_folder, new_extension, target_size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    to_convert = []

    # Step 1: Collect file paths to convert
    for root, _, files in os.walk(input_folder):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpeg', '.gif', '.bmp', '.jpg')):
                img_path = os.path.join(root, filename)
                to_convert.append(img_path)

    for img_path in to_convert:
        try:
            img = Image.open(img_path)

            # Convert images to RGB mode before saving
            if img.mode != 'RGB':
                img = img.convert('RGB')

            filename = os.path.basename(img_path)
            new_filename = os.path.splitext(filename)[0] + new_extension
            output_directory = os.path.join(output_folder, os.path.relpath(os.path.dirname(img_path), input_folder))
            
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            img_format = new_extension[1:].upper()
            img.save(os.path.join(output_directory, new_filename), img_format, quality=100)

            # Close the image
            img.close()

        except Exception as e:
            print(f"Error processing file {img_path}. Error message: {e}")

    # Note: If you are sure about the conversion, and you want to remove the original files, uncomment the following block:
    # for img_path in to_convert:
    #    os.remove(img_path)

    print("Conversion completed.")

input_folder = "data"
output_folder = "dataconv"
new_extension = ".png"
target_size = (224, 224)  # Note: you are not using target_size in this function

convert_images_in_folder(input_folder, output_folder, new_extension, target_size)
