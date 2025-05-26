import os

base_dir = './data/training_images'  
for digit_folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, digit_folder)
    if os.path.isdir(folder_path) and digit_folder.isdigit():
        for idx, filename in enumerate(os.listdir(folder_path)):
            old_path = os.path.join(folder_path, filename)
            extension = os.path.splitext(filename)[1]
            new_name = f"{digit_folder}_{idx}{extension}"
            new_path = os.path.join(folder_path, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed {old_path} -> {new_path}")

print("Done renaming!")
