import streamlit as st
import librosa
import numpy as np
import soundfile as sf
from io import BytesIO
import tempfile
import os
from pydub import AudioSegment
import matplotlib.pyplot as plt

st.set_page_config(page_title="Voice Modifier", page_icon="🎙️", layout="wide")

st.title("🎙️ Voice Modifier")
st.caption("Upload MP3/WAV → apply deep / gender-swap / robot effects. All processing is local.")

# Sidebar for optional API keys (not required for core function)
with st.sidebar:
    st.header("Optional API Keys")
    openai_key = st.text_input("OpenAI API Key", type="password", help="Optional – not used by core effects")
    serp_key = st.text_input("SerpAPI Key", type="password", help="Optional – not used by core effects")
    st.info("Core effects run entirely locally for security and offline use.")

# File uploader with strict types
uploaded = st.file_uploader(
    "Upload voice recording (MP3 or WAV)",
    type=["mp3", "wav"],
    help="Max recommended size ~50 MB. Only audio files are accepted."
)

EFFECTS = {
    "Deep & Rich (Morgan Freeman / Don LaFontaine style)": "deep",
    "Male → Female": "male_to_female",
    "Female → Male": "female_to_male",
    "Robot Voice": "robot",
}

effect = st.selectbox("Choose effect", list(EFFECTS.keys()))

def apply_effect(y, sr, effect_name):
    """Apply the selected voice transformation."""
    if effect_name == "deep":
        y_out = librosa.effects.pitch_shift(y, sr=sr, n_steps=-6)
        D = librosa.stft(y_out)
        freqs = librosa.fft_frequencies(sr=sr)
        low_mask = freqs < 300
        D[low_mask] *= 1.4
        y_out = librosa.istft(D)
    elif effect_name == "male_to_female":
        y_out = librosa.effects.pitch_shift(y, sr=sr, n_steps=7)
        D = librosa.stft(y_out)
        freqs = librosa.fft_frequencies(sr=sr)
        high_mask = freqs > 2500
        D[high_mask] *= 1.25
        y_out = librosa.istft(D)
    elif effect_name == "female_to_male":
        y_out = librosa.effects.pitch_shift(y, sr=sr, n_steps=-8)
        D = librosa.stft(y_out)
        freqs = librosa.fft_frequencies(sr=sr)
        low_mask = freqs < 400
        D[low_mask] *= 1.3
        y_out = librosa.istft(D)
    elif effect_name == "robot":
        y_out = librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)
        t = np.arange(len(y_out)) / sr
        modulator = 0.4 * np.sin(2 * np.pi * 30 * t)
        y_out = y_out * (1 + modulator)
        y_out = np.round(y_out * 32) / 32
    else:
        y_out = y

    peak = np.max(np.abs(y_out))
    if peak > 0:
        y_out = y_out / peak * 0.95
    return y_out

if uploaded is not None:
    if uploaded.size > 80 * 1024 * 1024:
        st.error("File too large. Please use a file under ~80 MB.")
        st.stop()

    with st.spinner("Loading and processing audio…"):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            y, sr = librosa.load(tmp_path, sr=None, mono=True)
            os.unlink(tmp_path)

            y_mod = apply_effect(y, sr, EFFECTS[effect])

            # Always create a WAV buffer first (used for playback and as source for MP3)
            wav_buf = BytesIO()
            sf.write(wav_buf, y_mod, sr, format="WAV")
            wav_buf.seek(0)
            wav_bytes = wav_buf.getvalue()

            st.success("Processing complete")
            st.audio(wav_bytes, format="audio/wav")

            # Download format selector
            download_format = st.radio(
                "Download format",
                options=["WAV", "MP3"],
                horizontal=True,
                index=0
            )

            base_name = uploaded.name.rsplit(".", 1)[0]
            effect_key = EFFECTS[effect]

            if download_format == "WAV":
                st.download_button(
                    label="Download modified audio (WAV)",
                    data=wav_bytes,
                    file_name=f"modified_{effect_key}_{base_name}.wav",
                    mime="audio/wav"
                )
            else:
                # Convert WAV bytes to MP3 via pydub
                audio_segment = AudioSegment.from_wav(BytesIO(wav_bytes))
                mp3_buf = BytesIO()
                audio_segment.export(mp3_buf, format="mp3", bitrate="192k")
                mp3_bytes = mp3_buf.getvalue()

                st.download_button(
                    label="Download modified audio (MP3)",
                    data=mp3_bytes,
                    file_name=f"modified_{effect_key}_{base_name}.mp3",
                    mime="audio/mpeg"
                )

            with st.expander("Show spectrograms (original vs modified)"):
                col1, col2 = st.columns(2)

                with col1:
                    st.write("Original")
                    fig1, ax1 = plt.subplots(figsize=(8, 4))
                    librosa.display.specshow(
                        librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max),
                        sr=sr, x_axis="time", y_axis="hz", ax=ax1
                    )
                    ax1.set_title("Original")
                    fig1.tight_layout()
                    st.pyplot(fig1)
                    plt.close(fig1)

                with col2:
                    st.write("Modified")
                    fig2, ax2 = plt.subplots(figsize=(8, 4))
                    librosa.display.specshow(
                        librosa.amplitude_to_db(np.abs(librosa.stft(y_mod)), ref=np.max),
                        sr=sr, x_axis="time", y_axis="hz", ax=ax2
                    )
                    ax2.set_title("Modified")
                    fig2.tight_layout()
                    st.pyplot(fig2)
                    plt.close(fig2)

        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.info("Make sure the file is a valid MP3 or WAV. For MP3 support (upload and download), ffmpeg must be installed on the system.")

st.markdown("---")
st.caption("Effects are approximate DSP transformations (pitch + spectral shaping). For higher quality voice conversion consider specialized models such as RVC. API keys are accepted but not required for these effects.")
