import os
import shutil
import random
def split_data(dataset_dir, train_dir, val_dir, test_dir, split_ratio=(0.8, 0.1, 0.1)):
    # If the directory already exists, delete it
    for directory in [train_dir, val_dir, test_dir]:
        if os.path.exists(directory):
            shutil.rmtree(directory)

    # Create the directory if it does not exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Collect the filenames
    image_files = os.listdir(dataset_dir)
    random.shuffle(image_files)

    # Split the data
    total_images = len(image_files)
    num_train = int(total_images * split_ratio[0])
    num_val = int(total_images * split_ratio[1])
    num_test = total_images - num_train - num_val

    # Copy the images to the correct directory
    for i, image_file in enumerate(image_files):
        src_path = os.path.join(dataset_dir, image_file)
        if i < num_train:
            dst_path = os.path.join(train_dir, image_file)
        elif i < num_train + num_val:
            dst_path = os.path.join(val_dir, image_file)
        else:
            dst_path = os.path.join(test_dir, image_file)
        shutil.copy(src_path, dst_path)

    print("Splitting data completed.")