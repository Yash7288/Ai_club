import sys
import numpy as np
import librosa
import tensorflow as tf

MODEL= "final_model.keras"
model = tf.keras.models.load_model(MODEL)

EMOTIONS = [
    "Neutral",
    "Calm",
    "Happy",
    "Sad",
    "Angry",
    "Fearful",
    "Disgust",
    "Surprise"
]


# PREPROCESSING the audio clip according to data
def preprocess_audio(wav_path):
    # Load audio (same as training)
    audio, sr = librosa.load(wav_path, sr=22050, duration=3)

    # Force exactly 3 seconds
    target_len = sr * 3
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    # Mel Spectrogram 
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128,
        n_fft=2048,
        hop_length=512
    )

  
    mel_db = librosa.power_to_db(mel, ref=np.max)

 
    if mel_db.shape[1] < 103:
        mel_db = np.pad(mel_db, ((0, 0), (0, 103 - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :103]

    
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())

    
    mel_db = mel_db[..., np.newaxis]
    mel_db = np.expand_dims(mel_db, axis=0)

    return mel_db

def predict_emotion(wav_path):
    x = preprocess_audio(wav_path)
    preds = model.predict(x, verbose=0)[0]

    idx = np.argmax(preds)
    emotion = EMOTIONS[idx]
    confidence = preds[idx] * 100

    return emotion, confidence

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <audio_file.wav>")
        sys.exit(1)

    wav_file = sys.argv[1]

    emotion, confidence = predict_emotion(wav_file)

    print("\nAudio File:", wav_file)
    print(" Predicted Emotion:", emotion)
    print(f" Confidence: {confidence:.2f}%\n")
