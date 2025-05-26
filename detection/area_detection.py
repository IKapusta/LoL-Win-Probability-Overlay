import cv2
import os
import detection.areas
#from upscaling import upscale_image

def calculate_coordinates(image_width, image_height, base_coordinates):
    """
    Scale base coordinates from a reference resolution (1920x1080) to the actual image size.
    """    
    base_image_width = 1920
    base_image_height = 1080

    calculated_coordinates = []
    for box in base_coordinates:
        box_coordinates = []
        for point in box:
            x_percentage = point[0] / base_image_width
            y_percentage = point[1] / base_image_height
            x = int(x_percentage * image_width)
            y = int(y_percentage * image_height)
            box_coordinates.append((x, y))

        calculated_coordinates.append(box_coordinates)

    return calculated_coordinates


def extract_areas_of_interest(image_path, calculated_coordinates):
    """
    Extract rectangular areas from the image based on calculated coordinates.
    """    
    img = cv2.imread(image_path)
    areas_of_interest = []
    
    for i, box in enumerate(calculated_coordinates):
        x_min = min(point[0] for point in box)
        y_min = min(point[1] for point in box)
        x_max = max(point[0] for point in box)
        y_max = max(point[1] for point in box)
        
        area = img[y_min:y_max, x_min:x_max]

        areas_of_interest.append(area)
    
    return areas_of_interest


def process_screenshots(screenshot_paths, base_coordinates, output_folder, upscaling = True):
    """
    Process screenshots by extracting areas of interest and saving them to the specified output folder.

    Parameters:
    - screenshot_paths: List of paths to the screenshots.
    - base_coordinates: List of base coordinates for areas of interest.
    - output_folder: Path to the folder where extracted areas will be saved.
    Returns:
    -list: Extracted image regions.   
    """
    all_areas_of_interest = []
    
    os.makedirs(output_folder, exist_ok=True)
    
    for path in screenshot_paths:
        img = cv2.imread(path)
        
        image_height, image_width, _ = img.shape 
        calculated_coordinates = calculate_coordinates(image_width, image_height, base_coordinates)
        areas_of_interest = extract_areas_of_interest(path, calculated_coordinates)
        
        for i, area in enumerate(areas_of_interest):
            save_path = os.path.join(output_folder, f"upscaled_area_{i}.png")
            cv2.imwrite(save_path, area)
            #if upscaling: 
            #    upscaled_path = os.path.join(output_folder, f"upscaled_area_{i}.png")
            #    upscale_image(save_path, upscaled_path)
   
        
        all_areas_of_interest.extend(areas_of_interest)
    
    return all_areas_of_interest


if __name__ == "__main__":
    screenshot_paths = ['./screenshots/screenshot_3.png']  
    output_folder = "./scoreboards"
    areas_of_interest = process_screenshots(screenshot_paths, detection.areas.blue_levels, output_folder)
