# LoL Win Probability Overlay

A Python-based project that analyzes League of Legends matches and displays a live win probability overlay using logistic regression models trained on real match data.

This project extracts in-game statistics from recorded or live videos, predicts win probabilities using trained logistic regression models at 10, 15, 20, and 25-minute marks, and displays these probabilities via a GUI overlay.

It includes:
- Frame upscaling and text extraction using OCR and Real-ESRGAN
- Object detection to track dragons, towers, kills, and other stats
- Model training pipeline for predicting match outcomes
- GUI overlay that updates probabilities in near real-time

## Usage:
git clone https://github.com/IKapusta/LoL-Win-Probability-Overlay.git

cd LoL-Win-Probability-Overlay

pip install -r requirements.txt

Required Real-ESRGAN is already provided in the core/realesrgan

if you want to retrain the models with different data, create a "data" folder in the project root directory and look into core/models_gametime.py for the training function 

## Acknowledgments
Real-ESRGAN

EasyOCR

Riot Games 
