from core.screenshot_module import RealTimeMatchPredictor
from overlay.overlay_combined import WinProbabilityOverlay
import threading

def main():
    # Create objects
    predictor = RealTimeMatchPredictor()
    ov = WinProbabilityOverlay(predictor)

    # Start prediction loop in a separate thread
    prediction_thread = threading.Thread(target=predictor.run_prediction_loop, daemon=True)
    prediction_thread.start()
    ov.run()

if __name__ == "__main__":
    main()