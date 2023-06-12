import os
import cv2
import numpy as np

# def add_noise(image_path, output_dir):
#     image = cv2.imread(image_path)

def augment_image(image_path, output_dir):
    # Load the image
    image = cv2.imread(image_path)

    # Apply small rotation
    angle = np.random.randint(-5, 5)
    rotation_matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    blur_radius = np.random.uniform(0.01, 0.6)
    blurred_image = cv2.GaussianBlur(rotated_image, (3, 3), blur_radius)

    # Add noise
    noise = np.random.normal(1, 0.3, blurred_image.shape).astype(np.uint8)
    noisy_image = cv2.add(blurred_image, noise)

    # Generate output filename
    image_name = os.path.basename(image_path)
    rand = round(abs(np.random.random()*10000))
    output_filename = os.path.join(output_dir, str(rand) + image_name)

    # Save augmented image
    cv2.imwrite(output_filename, noisy_image)
    print(f"Augmented image saved: {output_filename}")


def augment_images_in_directory(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get a list of image files in the input directory
    image_files = [file for file in os.listdir(input_dir) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Augment each image
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        augment_image(image_path, output_dir)


# Example usage
input_directory = 'M'
output_directory = 'M_aug'

augment_images_in_directory(input_directory, output_directory)
