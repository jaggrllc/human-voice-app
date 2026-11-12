import streamlit as st
import librosa
import numpy as np
import soundfile as sf
from io import BytesIO
import tempfile
import os
from pydub import AudioSegment
import matplotlib.pyplot as plt
import shutil

st.set_page_config(page_title="Voice Modifier", page_icon="🎙️", layout="wide")

st.title("🎙️ Voice Modifier")
st.caption("Upload MP3/WAV → apply effects, noise reduction, stereo, or keep original. All processing is local.")

# Sidebar for optional API keys
with st.sidebar:
    st.header("Optional API Keys")
    openai_key = st.text_input("OpenAI API Key", type="password", help="Optional – not used by core effects")
    serp_key = st.text_input("SerpAPI Key", type="password", help="Optional – not used by core effects")
    st.info("Core effects run entirely locally for security and offline use.")

uploaded = st.file_uploader(
    "Upload voice recording (MP3 or WAV)",
    type=["mp3", "wav"],
    help="Max recommended size ~50 MB. Only audio files are accepted."
)

EFFECTS = {
    "Original (no modification)": "original",
    "Deep & Rich (Morgan Freeman / Don LaFontaine style)": "deep",
    "Male → Female": "male_to_female",
    "Female → Male": "female_to_male",
    "Robot Voice": "robot",
}

effect = st.selectbox("Choose effect", list(EFFECTS.keys()))
reduce_noise = st.checkbox("Remove background noise and normalize audio", value=False)
make_stereo = st.checkbox("Output as stereo", value=True)

# Check if ffmpeg is available
ffmpeg_available = shutil.which("ffmpeg") is not None
if not ffmpeg_available:
    st.warning("MP3 download is unavailable (ffmpeg not found). WAV download works normally.")

def to_stereo(y, sr):
    """Convert mono to stereo with light widening, or keep stereo."""
    if y.ndim == 1:
        delay_samples = int(0.012 * sr)
        left = y
        right = np.pad(y, (delay_samples, 0))[:len(y)]
        right = right * 0.97
        y_stereo = np.vstack([left, right])
        return y_stereo
    else:
        return y

def reduce_noise_and_normalize(y, sr):
    """Simple spectral noise reduction + peak normalization (works on mono or stereo)."""
    if y.ndim == 2:
        cleaned = [reduce_noise_and_normalize(ch, sr) for ch in y]
        return np.vstack(cleaned)

    noise_samples = int(0.5 * sr)
    if len(y) > noise_samples:
        noise_clip = y[:noise_samples]
        noise_stft = np.abs(librosa.stft(noise_clip))
        noise_profile = np.mean(noise_stft, axis=1)

        D = librosa.stft(y)
        mag = np.abs(D)
        phase = np.angle(D)
        noise_mag = noise_profile[:, np.newaxis]
        mag_clean = np.maximum(mag - 1.5 * noise_mag, 0.05 * mag)
        D_clean = mag_clean * np.exp(1j * phase)
        y_clean = librosa.istft(D_clean)
    else:
        y_clean = y

    y_clean = librosa.effects.preemphasis(y_clean, coef=0.95)

    peak = np.max(np.abs(y_clean))
    if peak > 0:
        y_clean = y_clean / peak * 0.95
    return y_clean

def apply_effect(y, sr, effect_name):
    """Apply the selected voice transformation (supports mono or stereo)."""
    if y.ndim == 2:
        processed = [apply_effect(ch, sr, effect_name) for ch in y]
        return np.vstack(processed)

    if effect_name == "original":
        y_out = y.copy()
    elif effect_name == "deep":
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

            y, sr = librosa.load(tmp_path, sr=None, mono=False)
            os.unlink(tmp_path)

            if make_stereo:
                y = to_stereo(y, sr)

            if reduce_noise:
                y = reduce_noise_and_normalize(y, sr)

            y_mod = apply_effect(y, sr, EFFECTS[effect])

            # Write WAV
            wav_buf = BytesIO()
            sf.write(wav_buf, y_mod.T if y_mod.ndim == 2 else y_mod, sr, format="WAV")
            wav_buf.seek(0)
            wav_bytes = wav_buf.getvalue()

            st.success("Processing complete")
            st.audio(wav_bytes, format="audio/wav")

            # Download format selector (MP3 only if ffmpeg exists)
            format_options = ["WAV"]
            if ffmpeg_available:
                format_options.append("MP3")

            download_format = st.radio(
                "Download format",
                options=format_options,
                horizontal=True,
                index=0
            )

            base_name = uploaded.name.rsplit(".", 1)[0]
            effect_key = EFFECTS[effect]
            noise_tag = "_denoised" if reduce_noise else ""
            stereo_tag = "_stereo" if make_stereo else ""

            if download_format == "WAV":
                st.download_button(
                    label="Download audio (WAV)",
                    data=wav_bytes,
                    file_name=f"{effect_key}{noise_tag}{stereo_tag}_{base_name}.wav",
                    mime="audio/wav"
                )
            else:
                audio_segment = AudioSegment.from_wav(BytesIO(wav_bytes))
                mp3_buf = BytesIO()
                audio_segment.export(mp3_buf, format="mp3", bitrate="192k")
                mp3_bytes = mp3_buf.getvalue()

                st.download_button(
                    label="Download audio (MP3)",
                    data=mp3_bytes,
                    file_name=f"{effect_key}{noise_tag}{stereo_tag}_{base_name}.mp3",
                    mime="audio/mpeg"
                )

            with st.expander("Show spectrograms (original vs processed)"):
                y_display = y.mean(axis=0) if y.ndim == 2 else y
                y_mod_display = y_mod.mean(axis=0) if y_mod.ndim == 2 else y_mod

                col1, col2 = st.columns(2)

                with col1:
                    st.write("Original")
                    fig1, ax1 = plt.subplots(figsize=(8, 4))
                    librosa.display.specshow(
                        librosa.amplitude_to_db(np.abs(librosa.stft(y_display)), ref=np.max),
                        sr=sr, x_axis="time", y_axis="hz", ax=ax1
                    )
                    ax1.set_title("Original")
                    fig1.tight_layout()
                    st.pyplot(fig1)
                    plt.close(fig1)

                with col2:
                    st.write("Processed")
                    fig2, ax2 = plt.subplots(figsize=(8, 4))
                    librosa.display.specshow(
                        librosa.amplitude_to_db(np.abs(librosa.stft(y_mod_display)), ref=np.max),
                        sr=sr, x_axis="time", y_axis="hz", ax=ax2
                    )
                    ax2.set_title("Processed")
                    fig2.tight_layout()
                    st.pyplot(fig2)
                    plt.close(fig2)

        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.info("WAV download always works. MP3 requires ffmpeg (not available on Streamlit Cloud). Noise reduction works best when the first half-second contains mostly background noise.")

st.markdown("---")
st.caption("Select 'Original (no modification)' to keep the voice unchanged. Stereo option converts mono to stereo with light widening. Effects are approximate DSP transformations. API keys are accepted but not required.")
