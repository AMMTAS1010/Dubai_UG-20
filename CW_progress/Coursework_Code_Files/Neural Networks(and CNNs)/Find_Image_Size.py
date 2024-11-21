import os
from PIL import Image
from collections import defaultdict
import matplotlib.pyplot as plt

def analyze_image_sizes(root_dir, extensions={'jpg', 'jpeg', 'png', 'bmp', 'gif'}):
    size_counts = defaultdict(int)
    aspect_ratios = defaultdict(int)
    total_images = 0
    processed_subdirs = 0

    print(f"Starting image size analysis in directory: {root_dir}\n")

    for subdir, dirs, files in os.walk(root_dir):
        processed_subdirs += 1
        print(f"Processing subdirectory: {subdir} with {len(files)} files")
        for file in files:
            if file.split('.')[-1].lower() in extensions:
                total_images += 1
                try:
                    img_path = os.path.join(subdir, file)
                    with Image.open(img_path) as img:
                        width, height = img.size
                        size_counts[(width, height)] += 1

                        # Calculate aspect ratio rounded to two decimals
                        aspect_ratio = round(width / height, 2)
                        aspect_ratios[aspect_ratio] += 1
                except Exception as e:
                    print(f"Error processing {file}: {e}")

    print("\nAnalysis Complete!\n")
    print(f"Total Images Processed: {total_images}")
    print(f"Unique Image Sizes: {len(size_counts)}")

    if size_counts:
        # Find the most common image size
        most_common_size = max(size_counts, key=size_counts.get)
        print(f"Most Common Image Size: {most_common_size} with {size_counts[most_common_size]} occurrences")
    else:
        print("No images found with the specified extensions.")

    # Plotting image size distribution
    if size_counts:
        sizes, counts = zip(*size_counts.items())
        plt.figure(figsize=(10, 6))
        plt.scatter([size[0] for size in sizes], [size[1] for size in sizes], s=[count for count in counts], alpha=0.6)
        plt.title('Image Size Distribution')
        plt.xlabel('Width (pixels)')
        plt.ylabel('Height (pixels)')
        plt.grid(True)
        plt.show()
    else:
        print("Skipping image size distribution plot due to no images found.")

    # Plotting aspect ratio distribution
    if aspect_ratios:
        ratios, ratio_counts = zip(*aspect_ratios.items())
        plt.figure(figsize=(10, 6))
        plt.bar(ratios, ratio_counts, width=0.05, alpha=0.7)
        plt.title('Aspect Ratio Distribution')
        plt.xlabel('Aspect Ratio (Width/Height)')
        plt.ylabel('Number of Images')
        plt.grid(True)
        plt.show()
    else:
        print("Skipping aspect ratio distribution plot due to no images found.")

if __name__ == "__main__":
    # Update this path to your dataset's root directory
    dataset_root = "../Datasets/archive/images"  # <-- UPDATE THIS PATH

    analyze_image_sizes(dataset_root)