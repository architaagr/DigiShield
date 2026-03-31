# ============================================================
# app.py  —  DigiShield Real-Time Scam Detection Dashboard
#
# Architecture:
#   AudioThread → audio_queue → TranscribeThread → result_queue → UI
#
# Thread-safety contract:
#   - Worker threads NEVER touch st.session_state directly
#   - All inter-thread communication goes through Queue objects
#   - stop_event (threading.Event) is the single source of truth
#     for shutting down both worker threads
#   - Engine instances live in session_state so they survive
#     Streamlit reruns without being recreated (no duplicate threads,
#     no duplicate model loads)
# ============================================================

import queue
import threading
import time

import numpy as np
import streamlit as st

# ── Page config (must be first Streamlit call) ───────────────
st.set_page_config(
    page_title="DigiShield — Scam Detection",
    page_icon="🛡️",
    layout="wide",
)

# ── Constants ────────────────────────────────────────────────
SAMPLE_RATE      = 16000
BLOCK_SIZE       = 4096
BUFFER_SECONDS   = 3          # seconds of audio before we transcribe
SILENCE_THRESH   = 0.01       # RMS below this → skip transcription
MAX_TRANSCRIPTS  = 15         # keep last N in the history list
UI_POLL_INTERVAL = 0.5        # seconds between UI refreshes


# ════════════════════════════════════════════════════════════
# 1.  MODEL SINGLETON — loaded once per session
#     get_scorer() already handles singleton logic internally,
#     but we cache it in session_state so Streamlit reruns
#     never trigger a second load.
# ════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading BERT scorer…")
def load_scorer():
    from nlp_scorer import get_scorer
    return get_scorer()


@st.cache_resource(show_spinner="Loading Whisper model…")
def load_whisper():
    import torch
    from faster_whisper import WhisperModel
    device       = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel("small", device=device, compute_type=compute_type)


# ════════════════════════════════════════════════════════════
# 2.  WORKER THREADS
#     Both threads are created fresh on each START click.
#     They share:
#       - audio_queue   : raw numpy chunks
#       - result_queue  : scored result dicts
#       - stop_event    : signals both threads to exit
# ════════════════════════════════════════════════════════════

def audio_capture_thread(audio_queue: queue.Queue,
                          stop_event: threading.Event,
                          source: str):
    """
    Pushes raw audio chunks onto audio_queue.
    source = "SYSTEM" → Windows loopback via soundcard
    source = "MIC"    → default microphone via sounddevice
    """
    if source == "SYSTEM":
        try:
            import soundcard as sc
            from soundcard import SoundcardRuntimeWarning
            import warnings
            warnings.filterwarnings("ignore", category=SoundcardRuntimeWarning)

            speaker  = sc.default_speaker()
            loopback = sc.get_microphone(
                id=str(speaker.name), include_loopback=True
            )
            with loopback.recorder(samplerate=SAMPLE_RATE, channels=1) as recorder:
                while not stop_event.is_set():
                    data = recorder.record(numframes=BLOCK_SIZE)
                    audio_queue.put(data.copy())
        except Exception as exc:
            # Push sentinel so the transcribe thread knows audio died
            audio_queue.put(None)
            st.session_state.get("_errors", []).append(str(exc))

    else:  # MIC
        try:
            import sounddevice as sd
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                blocksize=BLOCK_SIZE,
            ) as stream:
                while not stop_event.is_set():
                    data, _ = stream.read(BLOCK_SIZE)
                    audio_queue.put(data.copy())
        except Exception as exc:
            audio_queue.put(None)


def transcribe_score_thread(audio_queue: queue.Queue,
                             result_queue: queue.Queue,
                             stop_event: threading.Event,
                             whisper_model,
                             scorer):
    """
    Pulls audio chunks, buffers BUFFER_SECONDS, transcribes with Whisper,
    scores with BERT, and pushes result dicts onto result_queue.

    Thread-safety note: result_queue is the ONLY channel used to
    communicate back to the main thread. No session_state writes here.
    """
    buffer = np.zeros((0, 1), dtype=np.float32)

    while not stop_event.is_set():
        try:
            chunk = audio_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        # Sentinel → audio thread died; stop gracefully
        if chunk is None:
            break

        buffer = np.concatenate((buffer, chunk))

        # Wait until we have at least BUFFER_SECONDS of audio
        if len(buffer) < SAMPLE_RATE * BUFFER_SECONDS:
            continue

        # ── Silence filter ────────────────────────────────────
        rms = float(np.sqrt(np.mean(buffer ** 2)))
        if rms < SILENCE_THRESH:
            buffer = np.zeros((0, 1), dtype=np.float32)
            continue

        # ── Whisper transcription ─────────────────────────────
        try:
            segments, _ = whisper_model.transcribe(
                buffer.flatten(),
                language="en",
                vad_filter=True,
            )
            full_text = " ".join(
                seg.text for seg in segments if seg.text.strip()
            ).strip()
        except Exception:
            buffer = np.zeros((0, 1), dtype=np.float32)
            continue

        buffer = np.zeros((0, 1), dtype=np.float32)

        if not full_text:
            continue

        # ── BERT scoring ──────────────────────────────────────
        try:
            result = scorer.score(full_text)
        except Exception:
            continue

        if result is None:
            continue

        # Push to result_queue for the UI to consume
        result_queue.put({
            "text":      full_text,
            "timestamp": time.strftime("%H:%M:%S"),
            **result,      # raw_score, smoothed_score, risk_level, flagged_phrases
        })


# ════════════════════════════════════════════════════════════
# 3.  SESSION STATE INITIALISATION
#     Runs once per browser session (not on every rerun).
# ════════════════════════════════════════════════════════════
def init_session():
    defaults = {
        "engine_started":  False,
        "audio_thread":    None,
        "transcribe_thread": None,
        "stop_event":      None,
        "audio_queue":     None,
        "result_queue":    None,
        "transcripts":     [],   # list of result dicts, newest first
        "latest_result":   None, # most recent scored result
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ════════════════════════════════════════════════════════════
# 4.  ENGINE START / STOP
# ════════════════════════════════════════════════════════════
def start_engine(source: str, whisper_model, scorer):
    """Spin up audio + transcription threads."""
    # Guard: do nothing if already running
    if st.session_state.engine_started:
        return

    stop_event   = threading.Event()
    audio_q      = queue.Queue(maxsize=200)   # bounded — prevents unbounded RAM
    result_q     = queue.Queue()

    at = threading.Thread(
        target=audio_capture_thread,
        args=(audio_q, stop_event, source),
        daemon=True,
        name="AudioCaptureThread",
    )
    tt = threading.Thread(
        target=transcribe_score_thread,
        args=(audio_q, result_q, stop_event, whisper_model, scorer),
        daemon=True,
        name="TranscribeScoreThread",
    )

    at.start()
    tt.start()

    st.session_state.stop_event        = stop_event
    st.session_state.audio_queue       = audio_q
    st.session_state.result_queue      = result_q
    st.session_state.audio_thread      = at
    st.session_state.transcribe_thread = tt
    st.session_state.engine_started    = True


def stop_engine():
    """Signal threads to stop and wait briefly for clean exit."""
    if not st.session_state.engine_started:
        return

    stop_event = st.session_state.stop_event
    if stop_event:
        stop_event.set()

    # Give threads a moment to exit cleanly (non-blocking for UI)
    for thread_key in ("audio_thread", "transcribe_thread"):
        t = st.session_state.get(thread_key)
        if t and t.is_alive():
            t.join(timeout=3)

    # Reset state — keep transcripts and latest_result for display
    st.session_state.engine_started    = False
    st.session_state.audio_thread      = None
    st.session_state.transcribe_thread = None
    st.session_state.stop_event        = None
    st.session_state.audio_queue       = None
    st.session_state.result_queue      = None


# ════════════════════════════════════════════════════════════
# 5.  DRAIN RESULT QUEUE → update session_state
#     Called once per rerun from the main thread.
#     This is the ONLY place session_state is written after init.
# ════════════════════════════════════════════════════════════
def drain_results():
    rq = st.session_state.result_queue
    if rq is None:
        return False

    updated = False
    while True:
        try:
            result = rq.get_nowait()
        except queue.Empty:
            break

        st.session_state.transcripts.insert(0, result)
        # Cap history to MAX_TRANSCRIPTS
        st.session_state.transcripts = st.session_state.transcripts[:MAX_TRANSCRIPTS]
        st.session_state.latest_result = result
        updated = True

    return updated


# ════════════════════════════════════════════════════════════
# 6.  UI HELPERS
# ════════════════════════════════════════════════════════════
RISK_COLOURS = {
    "HIGH":   "#FF4B4B",
    "MEDIUM": "#FFA500",
    "LOW":    "#21C354",
}
RISK_BAR_VALUES = {"HIGH": 1.0, "MEDIUM": 0.5, "LOW": 0.15}


def render_risk_panel(result: dict):
    risk    = result["risk_level"]
    score   = result["smoothed_score"]
    raw     = result["raw_score"]
    phrases = result["flagged_phrases"]
    colour  = RISK_COLOURS.get(risk, "#CCCCCC")

    st.markdown(
        f"""
        <div style="
            background:{colour}22;
            border-left: 6px solid {colour};
            border-radius:8px;
            padding:16px 20px;
            margin-bottom:12px;
        ">
            <span style="font-size:2rem;font-weight:700;color:{colour};">{risk} RISK</span>
            <span style="margin-left:16px;font-size:1rem;color:#888;">
                smoothed: {score:.4f} &nbsp;|&nbsp; raw: {raw:.4f}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(RISK_BAR_VALUES.get(risk, 0.0), text=f"Risk score: {score:.2%}")

    if phrases:
        st.warning(f"⚠️ Flagged phrases: **{', '.join(phrases)}**")


def render_transcript_history(transcripts: list):
    if not transcripts:
        st.info("No transcripts yet. Start the engine and speak or play audio.")
        return

    for item in transcripts:
        risk    = item["risk_level"]
        colour  = RISK_COLOURS.get(risk, "#888")
        ts      = item.get("timestamp", "")
        text    = item["text"]
        score   = item["smoothed_score"]
        phrases = item["flagged_phrases"]

        badge = (
            f'<span style="'
            f'background:{colour};color:#fff;padding:2px 8px;'
            f'border-radius:4px;font-size:0.75rem;font-weight:600;">'
            f'{risk}</span>'
        )
        phrase_html = (
            f'<br><small style="color:#aaa;">⚠ {", ".join(phrases)}</small>'
            if phrases else ""
        )

        st.markdown(
            f"""
            <div style="
                border:1px solid #2a2a2a;
                border-radius:6px;
                padding:10px 14px;
                margin-bottom:6px;
                background:#1a1a1a;
            ">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    {badge}
                    <small style="color:#666;">{ts} &nbsp;|&nbsp; {score:.4f}</small>
                </div>
                <p style="margin:6px 0 0;color:#e0e0e0;">{text}</p>
                {phrase_html}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ════════════════════════════════════════════════════════════
# 7.  MAIN APP
# ════════════════════════════════════════════════════════════
def main():
    init_session()

    # Load models once (cached across reruns and sessions)
    scorer        = load_scorer()
    whisper_model = load_whisper()

    # ── Header ───────────────────────────────────────────────
    st.markdown(
        "<h1 style='margin-bottom:0'>🛡️ DigiShield</h1>"
        "<p style='color:#888;margin-top:4px;'>Real-Time Scam Call Detection</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Sidebar: controls ─────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Controls")

        source = st.radio(
            "Audio source",
            options=["SYSTEM", "MIC"],
            format_func=lambda x: "🔊 System loopback" if x == "SYSTEM" else "🎙️ Microphone",
            disabled=st.session_state.engine_started,
        )

        col1, col2 = st.columns(2)
        start_clicked = col1.button(
            "▶ Start",
            use_container_width=True,
            disabled=st.session_state.engine_started,
            type="primary",
        )
        stop_clicked = col2.button(
            "⏹ Stop",
            use_container_width=True,
            disabled=not st.session_state.engine_started,
        )

        if start_clicked:
            start_engine(source, whisper_model, scorer)
            st.rerun()

        if stop_clicked:
            stop_engine()
            st.rerun()

        st.divider()

        if st.button("🗑️ Clear history", use_container_width=True):
            st.session_state.transcripts  = []
            st.session_state.latest_result = None
            scorer.reset_buffer()
            st.rerun()

        st.divider()
        st.markdown(
            "**Status:** "
            + ("🟢 Running" if st.session_state.engine_started else "🔴 Stopped")
        )
        if st.session_state.engine_started:
            st.caption(f"Source: {source}")

    # ── Main area ─────────────────────────────────────────────
    left_col, right_col = st.columns([1, 2], gap="large")

    # LEFT: current risk panel
    with left_col:
        st.subheader("Current Risk")
        if st.session_state.latest_result:
            render_risk_panel(st.session_state.latest_result)
        else:
            st.markdown(
                "<div style='color:#555;padding:20px 0;'>Waiting for audio…</div>",
                unsafe_allow_html=True,
            )

    # RIGHT: transcript history
    with right_col:
        st.subheader(f"Transcript History  (last {MAX_TRANSCRIPTS})")
        render_transcript_history(st.session_state.transcripts)

    # ── Auto-refresh while engine is running ──────────────────
    # We drain the result queue, then schedule a rerun after
    # UI_POLL_INTERVAL seconds. This keeps the UI live without
    # blocking the main thread or freezing Streamlit.
    if st.session_state.engine_started:
        # Drain any pending results into session_state
        drain_results()

        # st_autorefresh is not available in all builds; use time.sleep + rerun.
        # We use a short sleep so Streamlit doesn't spin at 100 % CPU.
        time.sleep(UI_POLL_INTERVAL)
        st.rerun()
    else:
        # Even when stopped, drain any last results that arrived before stop
        if drain_results():
            st.rerun()


if __name__ == "__main__":
    main()