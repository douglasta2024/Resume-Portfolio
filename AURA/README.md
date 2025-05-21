# Project 3

By: Douglas Ta

Google Drive Link (Stores my model weights and video demo): https://drive.google.com/drive/folders/1L8LcHDI2UC7Q0BqdCVZpKhtKokRdlACv?usp=sharing

How to Replicate:
1. Create virtual environment using requirements.txt (pip install -r requirements.txt)
2. Open up terminal and type in this specific command (python -m streamlit run app.py). Make sure you are in correct working dir.
3. Make sure to change model path to local path
4. Record audio of command into streamlit interface.

Command Instructions:

This AI system takes speech commands and classifies them as such utilizing a combination of a CNN and a transformer to do so. The following commands can be used: 'up', 'down', 'left', 'right', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'on', 'off', 'go', 'yes', 'no'. To record your commands make sure to start speaking at the 1st second mark and continue every command every other second. The system pairs inputs together so a directional command should always be followed up with a number. For example 'up' 'two' indicates that the drone should fly straight for 2 seconds.
