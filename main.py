import numpy as np
import os
import pandas as pd
from count import ObjectCounter
from text_transformer import SceneDescriptionGenerator  # Assuming you have this module
from datasets import GroceryStoreDataset  # Assuming you have this module
import torchvision.transforms as transforms

# Define the transformation for the images
TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.CenterCrop((320, 320)),
    transforms.ToTensor(),
])

def main():
    # Path to the trained model and model type
    trained_model_path = 'checkpoints/retinanet/checkpoint.pth'
    model_type = 'retinanet'  # Can be 'ssd', 'frcnn', or 'retinanet'

    # Initialize the object counter with the specified model type
    object_counter = ObjectCounter(trained_model_path, model_type)

    # Path to the image to be processed
    image_path = "/work/cvcs_2023_group23/SKU110K_fixed/images/test_78.jpg"

    # Process the image using the object counter
    num_objects, relationships, colors = object_counter.count_objects_and_relations(image_path)

    # Display the number of objects detected
    print(f"Number of objects detected: {num_objects}")

    # Uncomment to display spatial relationships
    # for rel in relationships:
    #     print(f"Spatial Relationship: {rel}")

    # Initialize the scene description generator
    description_generator = SceneDescriptionGenerator()  # Assuming this module is defined

    # Generate and display the scene description
    description = description_generator.generate_description(num_objects, relationships, colors)
    print(f"Scene description: {description}")

if __name__ == '__main__':
    main()

