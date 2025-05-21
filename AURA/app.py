import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification
from transformers import AutoFeatureExtractor
import librosa
from board_commands import load_commands

def main():
    
    def activate():
        load_commands(predicted_classes)
        
    st.title("Language Drone")
    st.write("This AI system takes speech commands and classifies them as such utilizing a combination of a CNN and a transformer to do so. The following commands can be used: 'up', 'down', 'left', 'right', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'on', 'off', 'go', 'yes', 'no'")
    st.write("To record your commands make sure to start speaking at the 1st second mark and continue every command every other second.")

    # collects command audio
    input = st.audio_input("Record Command Here")

    target_labels = ['up', 'down', 'left', 'right', 'zero', 'one', 'two', 
                'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
                'on', 'off', 'go', 'yes', 'no']

    label2id = {label: i for i, label in enumerate(target_labels)}
    id2label = {i: label for i, label in enumerate(target_labels)}

    model_id = "superb/wav2vec2-base-superb-ks"

    # loading in the model weights
    loaded_model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_id,
        num_labels=len(target_labels),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True
    )

    # Loading in feature extractor for audio preprocessing
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

    # will need to change model weight file path for replication
    saved_model_path = "E:/temp/wav2vec2_model_weights.pth"
    loaded_model.load_state_dict(torch.load(saved_model_path, weights_only=True, map_location=torch.device('cpu')))
    loaded_model.eval()


    if input is not None:

        sr = 16000
        audio, _ = librosa.load(input, sr=sr)
        
        chunk_duration = 2
        chunk_samples = int(chunk_duration * sr)
        chunks = [audio[i:i + chunk_samples] for i in range(0, len(audio), chunk_samples)]

        predicted_classes = []

        for chunk in chunks:

            inputs = feature_extractor(
                chunk,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
                max_length=16000  # 1 second of audio
            )

            # Make prediction
            with torch.no_grad():
                output = loaded_model(inputs.input_values)
                _, predicted = torch.max(output.logits, 1)
                predicted_class = id2label[predicted.item()]
                predicted_classes.append(predicted_class)
            
        st.success(f"Predicted Command: {predicted_classes}")

        if len(predicted_classes) != 0:
            st.write("Confirm these are the correct navigation commands")
            st.button(label="Yes", on_click=activate)
            

if __name__ == "__main__":
    main()