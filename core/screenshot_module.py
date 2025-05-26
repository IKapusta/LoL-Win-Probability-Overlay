import os
import pyautogui
import cv2
import numpy as np

from core.sample_builder import construct_sample
from core.upscaling import upscale_image
import core.models_gametime


class RealTimeMatchPredictor:

    def __init__(self, screenshot_folder="screenshots"):
        self.model = core.models_gametime.LOLWinProbabilityModel("../data/gametime_data.csv")
        self.model.load_models()

        self.screenshot_folder = screenshot_folder
        os.makedirs(self.screenshot_folder, exist_ok=True)

        self.viewers = []
        self.done = False


    def add_viewer(self,viewer):
        self.viewers.append(viewer)
    

    def end(self):
        self.done = True

    
    def update_viewers(self, gametime, prob):
        for viewer in self.viewers:
            viewer.update(gametime,prob)


    def take_screenshot(self):
        """Captures a screenshot using pyautogui and saves it in OpenCV format."""
        screenshot = pyautogui.screenshot()  
        screenshot = np.array(screenshot)  #
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)  

        screenshot_path = os.path.join(self.screenshot_folder, "screenshot.png")
        cv2.imwrite(screenshot_path, screenshot)  
        return screenshot_path

    def run_prediction_loop(self):
        """Continuously takes screenshots, processes them, and predicts the game state."""
        try:
            frame_count= 0
            while not self.done:
                frame_count += 1
                screenshot_path = self.take_screenshot()
                
                if screenshot_path:
                    try:
                        upscaled_path = os.path.join(self.screenshot_folder, "screenshot_upscaled.png")
                        upscale_image(screenshot_path, upscaled_path)  
                        sample, minute, gametime = construct_sample([upscaled_path])  
                        
                        prob = self.model.predict_win_probability(sample, minute)
                        prob = prob[0]
                        print("BLUEWINPROB:", prob)
                        self.update_viewers(frame_count, prob)

                    except Exception as e:
                        print(f"Error constructing sample from {screenshot_path}: {e}")

        except KeyboardInterrupt:
            print("Stopping prediction loop.")


if __name__ == "__main__":
    predictor = RealTimeMatchPredictor() 
    predictor.run_prediction_loop()
