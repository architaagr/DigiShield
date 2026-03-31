# ============================================================
# realtime_asr.py  —  DIGISHIELD integrated version
#
# What changed from the original:
#   - BERTScorer is loaded at startup
#   - Every transcribed segment is immediately scored
#   - Risk level + flagged phrases printed to console
#   - Everything else (audio threads, VAD, silence filter)
#     is IDENTICAL to your original code
#
# Run:
#   python realtime_asr.py
#
# Prerequisites:
#   1. Run train_model.py first (builds model_artifacts/)
#   2. pip install faster-whisper soundcard sounddevice
#              transformers torch datasets pandas
# ============================================================

import soundcard as sc
import sounddevice as sd
import numpy as np
import threading
import queue
import sys
import time

import warnings
from soundcard import SoundcardRuntimeWarning
warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning)

from faster_whisper import WhisperModel

# ── Load BERT scorer first (before audio starts) ──────────────
print("Loading BERT scorer...")
from nlp_scorer import BERTScorer
scorer = BERTScorer()
scorer.load()
print("BERT scorer ready.\n")

# ── Load Whisper ───────────────────────────────────────────────
SAMPLE_RATE = 16000
BLOCK_SIZE  = 4096

print("Loading Whisper model...")
whisper_model = WhisperModel("small", device="cuda", compute_type="float16")
print("Whisper loaded.")
print("System audio: ACTIVE  |  Microphone: OFF by default")
print("Type 'm' + Enter to toggle mic.\n")
print("-" * 60)

# ── Shared state ───────────────────────────────────────────────
audio_queue = queue.Queue()
stop_event  = threading.Event()
mic_enabled = False


# ─────────────────────────────────────────────────────────────
# SYSTEM AUDIO (Loopback) — unchanged from original
# ─────────────────────────────────────────────────────────────
def record_system_audio():
    speaker  = sc.default_speaker()
    loopback = sc.get_microphone(id=str(speaker.name), include_loopback=True)

    with loopback.recorder(samplerate=SAMPLE_RATE, channels=1) as recorder:
        while not stop_event.is_set():
            data = recorder.record(numframes=BLOCK_SIZE)
            audio_queue.put(("SYSTEM", data.copy()))

    print("System audio stopped.")


# ─────────────────────────────────────────────────────────────
# MICROPHONE AUDIO — unchanged from original
# ─────────────────────────────────────────────────────────────
def record_mic():
    global mic_enabled

    with sd.InputStream(samplerate=SAMPLE_RATE,
                        channels=1,
                        blocksize=BLOCK_SIZE) as stream:
        while not stop_event.is_set():
            if mic_enabled:
                data, _ = stream.read(BLOCK_SIZE)
                audio_queue.put(("MIC", data.copy()))
            else:
                time.sleep(0.1)

    print("Microphone stopped.")


# ─────────────────────────────────────────────────────────────
# MIC TOGGLE — unchanged from original
# ─────────────────────────────────────────────────────────────
def mic_control():
    global mic_enabled

    while not stop_event.is_set():
        try:
            cmd = input("\nType 'm' to toggle mic ON/OFF: ").strip().lower()
            if cmd == 'm':
                mic_enabled = not mic_enabled
                print(f"Microphone: {'ON' if mic_enabled else 'OFF'}")
        except EOFError:
            break


# ─────────────────────────────────────────────────────────────
# TRANSCRIPTION + SCORING
# This is where your original print() is replaced with scoring
# ─────────────────────────────────────────────────────────────
def transcribe_and_score():
    buffer = np.zeros((0, 1), dtype=np.float32)

    while not stop_event.is_set():
        try:
            source, chunk = audio_queue.get(timeout=1)
            buffer = np.concatenate((buffer, chunk))

            # Wait until 3 seconds of audio is buffered
            if len(buffer) <= SAMPLE_RATE * 3:
                continue

            # ── Silence filter (unchanged) ────────────────────
            energy = np.sqrt(np.mean(buffer ** 2))
            if energy < 0.01:
                buffer = np.zeros((0, 1), dtype=np.float32)
                continue

            # ── Whisper transcription (unchanged) ─────────────
            segments, _ = whisper_model.transcribe(
                buffer.flatten(),
                language="en",
                vad_filter=True,
            )

            full_text = ""
            for segment in segments:
                if segment.text.strip():
                    full_text += segment.text + " "

            buffer = np.zeros((0, 1), dtype=np.float32)

            if not full_text.strip():
                continue

            # ── BERT scoring (new) ────────────────────────────
            result = scorer.score(full_text.strip())

            if result is None:
                continue

            # ── Console output ────────────────────────────────
            risk      = result["risk_level"]
            score     = result["smoothed_score"]
            raw       = result["raw_score"]
            phrases   = result["flagged_phrases"]

            # Colour codes for terminal
            colours = {"HIGH": "\033[91m", "MEDIUM": "\033[93m", "LOW": "\033[92m"}
            reset   = "\033[0m"
            colour  = colours.get(risk, "")

            print(f"\n[{source}] {full_text.strip()}")
            print(f"  {colour}▶ RISK: {risk}  |  score: {score:.4f}  (raw: {raw:.4f}){reset}")
            if phrases:
                print(f"  ⚠ Flagged: {', '.join(phrases)}")
            print("-" * 60)

        except queue.Empty:
            continue

    print("Transcription stopped.")


# ─────────────────────────────────────────────────────────────
# START THREADS
# ─────────────────────────────────────────────────────────────
system_thread    = threading.Thread(target=record_system_audio, daemon=True)
mic_thread       = threading.Thread(target=record_mic,          daemon=True)
control_thread   = threading.Thread(target=mic_control,         daemon=True)
transcribe_thread = threading.Thread(target=transcribe_and_score, daemon=True)

system_thread.start()
mic_thread.start()
control_thread.start()
transcribe_thread.start()

# ─────────────────────────────────────────────────────────────
# CLEAN SHUTDOWN
# ─────────────────────────────────────────────────────────────
try:
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("\n\nShutting down safely...")
    stop_event.set()

    system_thread.join(timeout=3)
    mic_thread.join(timeout=3)
    transcribe_thread.join(timeout=3)

    print("All audio devices released.")
    sys.exit(0)
