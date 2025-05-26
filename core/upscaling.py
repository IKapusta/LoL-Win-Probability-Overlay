import subprocess
import os
from paths import CORE_DIR

def upscale_image(input_path, output_path):
    """
    Upscale an image using Real-ESRGAN.

    Parameters:
    - input_path: Path to the input image.
    - output_path: Path to save the upscaled image.
    """
    executable_path = CORE_DIR / "realesrgan/realesrgan-ncnn-vulkan.exe"
    if not os.path.exists(executable_path):
        raise FileNotFoundError(f"Executable not found at: {executable_path}")
    
    command = [
        executable_path,
        "-i", input_path,
        "-o", output_path
    ]
    
    try:
        subprocess.run(
            command,
            check=True,
            creationflags=subprocess.CREATE_NO_WINDOW  
        )
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")

# Example usage
if __name__ == "__main__":
    input_path = "./realesrgan/area_0.png"
    output_path = "./scoreboards/output.png"
    upscale_image(input_path, output_path)
