from shelf_classifier import ShelfClassifier

# Define the path to your annotations file and images folder
annotations_file = '/work/cvcs_2023_group23/SKU110K_fixed/annotations/annotations_val.csv'
images_folder = '/work/cvcs_2023_group23/SKU110K_fixed/images/'

# Instantiate the ShelfClassifier with the paths to the annotations file and images folder
shelf_classifier = ShelfClassifier(annotations_file, images_folder)

# Select a random image from the annotations to process
# Note: You can replace this with a specific image name if you want to process a known image
random_image_name = shelf_classifier.select_random_image()

# Use the optimize_shelf_placement method to get the optimal shelf and the final array for the selected image
optimal_shelf, final_array = shelf_classifier.optimize_shelf_placement(random_image_name)

# Print the results
print(f"Optimal Shelf: {optimal_shelf}")
print("Final Array:")
for item in final_array:
    print(item)