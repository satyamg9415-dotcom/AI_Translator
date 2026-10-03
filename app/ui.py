"""
ui.py — Streamlit UI for the GenAI Language Translator.
Two tabs: Text Translation and Voice Translation.
Features: auto language detection, LoRA fine-tuned model toggle, side-by-side comparison,
interactive 3D globe (three.js) that reacts to the selected language pair.
"""
import html
import json
import logging
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GenAI Translator",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS (ink + saffron professional theme) ─────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Instrument+Sans:wght@400;500;600&display=swap');

:root{--ink:#0a1020;--panel:#101a30;--line:#26324d;--text:#e8edf6;--mute:#8d9bb8;--saf:#ffb02e;--sky:#5ab0ff}

html, body, [class*="css"] { font-family: 'Instrument Sans', 'Segoe UI', sans-serif; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.stApp { background: radial-gradient(1200px 600px at 80% -10%, #16254a 0%, var(--ink) 60%); color: var(--text); }
.block-container { max-width: 1180px; padding-top: 1.2rem; }

/* ── Top bar ── */
.topbar { display:flex; justify-content:space-between; align-items:center; padding:.4rem 0 .2rem; }
.brand { font:700 1.15rem 'Bricolage Grotesque',sans-serif; display:flex; gap:10px; align-items:center; }
.brand i { width:12px; height:12px; border-radius:50%; background:var(--saf); box-shadow:0 0 14px var(--saf); }
.status { border:1px solid var(--line); border-radius:99px; padding:5px 14px; font-size:.82rem; color:var(--mute); display:inline-flex; gap:8px; align-items:center; }
.status b { width:8px; height:8px; border-radius:50%; background:#fbbf24; }
.status.on b { background:#4ade80; }

/* ── Hero ── */
.hero { padding: 3.2rem 0 1rem 0; }
.hero-title { font:800 clamp(2.4rem,4.6vw,3.9rem)/1.03 'Bricolage Grotesque',sans-serif; letter-spacing:-.03em; margin:0; }
.hero-subtitle { font-size:1.12rem; color:var(--mute); margin-top:1.1rem; max-width:44ch; line-height:1.6; }
.hero-stack { margin-top:1.6rem; font-size:.85rem; color:var(--mute); }

/* ── Cards ── */
.glass-card { background:rgba(16,26,48,.7); border:1px solid var(--line); border-radius:14px; padding:1.5rem; backdrop-filter:blur(12px); margin-bottom:1rem; }

.lang-badge { display:inline-block; border:1px solid var(--line); border-radius:6px; padding:3px 10px; font-size:.82rem; margin:.4rem .4rem 0 0; }
.lang-badge.high { border-color:rgba(74,222,128,.4);  color:#4ade80; }
.lang-badge.med  { border-color:rgba(251,191,36,.4);  color:#fbbf24; }
.lang-badge.low  { border-color:rgba(248,113,113,.4); color:#f87171; }
.hinglish-badge { display:inline-block; border:1px solid rgba(255,176,46,.45); color:var(--saf); border-radius:6px; padding:3px 10px; font-size:.82rem; font-weight:600; margin-top:.4rem; }

.translation-box { background:var(--ink); border:1px solid var(--line); border-left:3px solid var(--sky); border-radius:10px; padding:1.1rem 1.3rem; font-size:1.06rem; color:var(--text); line-height:1.7; min-height:76px; margin-top:.5rem; white-space:pre-wrap; }
.translation-box.finetuned { border-left-color:var(--saf); }

.section-label { font-size:.85rem; font-weight:600; color:var(--mute); margin-bottom:.4rem; }
.section-label.purple { color:var(--saf); }
.section-label.blue   { color:var(--sky); }

/* ── Streamlit widget overrides ── */
.stTextArea textarea { background:var(--ink) !important; border:1px solid var(--line) !important; border-radius:8px !important; color:var(--text) !important; font-family:'Instrument Sans',sans-serif !important; }
.stTextArea textarea:focus { border-color:var(--saf) !important; box-shadow:none !important; }
.stSelectbox > div > div { background:var(--ink) !important; border:1px solid var(--line) !important; border-radius:8px !important; color:var(--text) !important; }
.stButton > button { background:var(--saf) !important; border:none !important; border-radius:8px !important; color:#1a1100 !important; font-weight:600 !important; font-family:'Instrument Sans',sans-serif !important; padding:.55rem 2rem !important; transition:transform .15s, box-shadow .15s !important; }
.stButton > button:hover { transform:translateY(-1px) !important; box-shadow:0 8px 26px rgba(255,176,46,.35) !important; }
.stTabs [data-baseweb="tab-list"] { gap:6px; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"] { color:var(--mute) !important; font-family:'Instrument Sans',sans-serif !important; font-weight:600; padding:12px 22px; background:transparent; }
.stTabs [aria-selected="true"] { color:var(--text) !important; }
.stTabs [data-baseweb="tab-highlight"] { background:var(--saf) !important; }
div[data-testid="stAlert"] { border-radius:10px; }
hr { border-color:var(--line); margin:1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Imports (deferred for faster startup) ─────────────────────────────────────
from app.config import SOURCE_LANGUAGES, TARGET_LANGUAGES, NLLB_TO_DISPLAY, WHISPER_TO_NLLB
from app.lang_detector import detect_language, is_hinglish
from app.text_translator import translate, is_finetuned_available


# ── Helpers ───────────────────────────────────────────────────────────────────
def confidence_class(conf: float) -> str:
    if conf >= 0.80:
        return "high"
    elif conf >= 0.50:
        return "med"
    return "low"


def lang_badge_html(label: str, conf: float) -> str:
    return (
        f'<span class="lang-badge {confidence_class(conf)}">'
        f'Detected <strong>{html.escape(label)}</strong> · {conf*100:.0f}% confidence'
        f'</span>'
    )


def hinglish_badge_html() -> str:
    return '<span class="hinglish-badge">Hinglish detected — fine-tuned model available</span>'


def box(text: str, finetuned: bool = False) -> None:
    """Render a translation/transcript box (HTML-escaped)."""
    cls = "translation-box finetuned" if finetuned else "translation-box"
    st.markdown(f'<div class="{cls}">{html.escape(text)}</div>', unsafe_allow_html=True)


def speak(text: str, nllb_code: str) -> None:
    """Text-to-speech that never breaks the page if gTTS fails."""
    try:
        from app.text_to_speech import synthesize
        st.audio(synthesize(text, nllb_code), format="audio/mp3")
    except Exception as e:  # network / unsupported language
        st.caption(f"Audio unavailable: {e}")


# ── 3D globe (three.js inside a Streamlit component) ──────────────────────────
GLOBE_HTML = """
<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;height:100%;background:transparent;overflow:hidden;font-family:'Segoe UI',system-ui,sans-serif}
canvas{width:100%;height:100%;display:block;cursor:grab}canvas:active{cursor:grabbing}
#cap{position:absolute;left:8px;bottom:8px;color:#8d9bb8;font-size:13px;letter-spacing:.01em}
#cap b{color:#ffb02e;font-weight:600}
</style></head><body>
<canvas id="c"></canvas><div id="cap"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const SRC=__SRC__,TGT=__TGT__,NAMES=__NAMES__;
const still=matchMedia('(prefers-reduced-motion:reduce)').matches;
document.getElementById('cap').innerHTML=SRC===TGT?'<b>'+SRC+'</b>':'<b>'+SRC+'</b> &rarr; <b>'+TGT+'</b> &nbsp;· drag to rotate';
const cv=document.getElementById('c'),R=new THREE.WebGLRenderer({canvas:cv,antialias:true,alpha:true});
R.setPixelRatio(Math.min(devicePixelRatio,2));
const S=new THREE.Scene(),C=new THREE.PerspectiveCamera(36,1,.1,50);C.position.z=4.1;
const G=new THREE.Group();S.add(G);
const ll=(a,o,r=1)=>{const p=(90-a)*Math.PI/180,t=(o+180)*Math.PI/180;return new THREE.Vector3(-r*Math.sin(p)*Math.cos(t),r*Math.cos(p),r*Math.sin(p)*Math.sin(t))};
G.add(new THREE.Mesh(new THREE.SphereGeometry(.985,48,48),new THREE.MeshBasicMaterial({color:0x0a1020})));
const N=2600,pos=new Float32Array(N*3);
for(let i=0;i<N;i++){const y=1-2*(i+.5)/N,r=Math.sqrt(1-y*y),t=i*2.399963;pos.set([Math.cos(t)*r,y,Math.sin(t)*r],i*3)}
const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(pos,3));
G.add(new THREE.Points(g,new THREE.PointsMaterial({color:0x5b7aa8,size:.012})));
const tex=(()=>{const c=document.createElement('canvas');c.width=c.height=64;const x=c.getContext('2d'),q=x.createRadialGradient(32,32,0,32,32,32);
q.addColorStop(0,'rgba(255,176,46,1)');q.addColorStop(.25,'rgba(255,176,46,.5)');q.addColorStop(1,'rgba(255,176,46,0)');x.fillStyle=q;x.fillRect(0,0,64,64);return new THREE.CanvasTexture(c)})();
const CITY={hindi:[28.6,77.2],english:[51.5,-.1],spanish:[40.4,-3.7],french:[48.9,2.3],german:[52.5,13.4],chinese:[39.9,116.4],japanese:[35.7,139.7],russian:[55.8,37.6],arabic:[24.7,46.7],portuguese:[-15.8,-47.9],korean:[37.6,127],bengali:[23.8,90.4],tamil:[13.1,80.3],italian:[41.9,12.5],hinglish:[26.9,75.8]};
const where=n=>{const k=Object.keys(CITY).find(k=>n.toLowerCase().startsWith(k));if(k)return CITY[k];let h=0;for(const c of n)h=(h*31+c.charCodeAt(0))%997;return[(h%120)-60,(h*7)%340-170]};
const nodes={};
NAMES.forEach(n=>{const s=new THREE.Sprite(new THREE.SpriteMaterial({map:tex,blending:THREE.AdditiveBlending,depthWrite:false}));
const p=ll(...where(n),1.005);s.position.copy(p);const hot=(n===SRC||n===TGT);s.scale.setScalar(hot?.26:.09);G.add(s);nodes[n]=p});
let arc=null,face=null,t0=performance.now();
if(nodes[SRC]&&nodes[TGT]&&SRC!==TGT){const A=nodes[SRC],B=nodes[TGT],m=A.clone().add(B).normalize(),d=A.distanceTo(B),
cu=new THREE.QuadraticBezierCurve3(A,m.clone().multiplyScalar(1+d*.55),B);
arc=new THREE.Line(new THREE.BufferGeometry().setFromPoints(cu.getPoints(80)),new THREE.LineBasicMaterial({color:0xffb02e}));
arc.geometry.setDrawRange(0,still?81:0);G.add(arc);face=m}
else if(nodes[SRC])face=nodes[SRC].clone().normalize();
const up=new THREE.Vector3(0,.2,1).normalize(),q=new THREE.Quaternion();
let drag=false,lx=0,ly=0;
cv.addEventListener('pointerdown',e=>{drag=true;face=null;lx=e.clientX;ly=e.clientY;cv.setPointerCapture(e.pointerId)});
cv.addEventListener('pointerup',()=>drag=false);
cv.addEventListener('pointermove',e=>{if(!drag)return;
G.quaternion.premultiply(new THREE.Quaternion().setFromEuler(new THREE.Euler((e.clientY-ly)*.006,(e.clientX-lx)*.006,0)));lx=e.clientX;ly=e.clientY});
function size(){const w=cv.clientWidth,h=cv.clientHeight;R.setSize(w,h,false);C.aspect=w/h;C.updateProjectionMatrix()}
addEventListener('resize',size);size();
(function loop(now){requestAnimationFrame(loop);
if(face){q.setFromUnitVectors(face,up);G.quaternion.slerp(q,.04)}else if(!drag&&!still)G.rotateY(.0016);
if(arc)arc.geometry.setDrawRange(0,Math.floor(Math.min(1,(now-t0)/1300)*81));
R.render(S,C)})(0);
</script></body></html>
"""


def render_globe(src_name: str, tgt_name: str) -> None:
    names = sorted(set(SOURCE_LANGUAGES) | set(TARGET_LANGUAGES))
    page = (
        GLOBE_HTML
        .replace("__SRC__", json.dumps(src_name))
        .replace("__TGT__", json.dumps(tgt_name))
        .replace("__NAMES__", json.dumps(names))
    )
    components.html(page, height=430)


# ── Top bar ────────────────────────────────────────────────────────────────────
ft_available = is_finetuned_available()
st.markdown(
    f'<div class="topbar"><div class="brand"><i></i>GenAI Translator</div>'
    f'<div class="status {"on" if ft_available else ""}"><b></b>'
    f'{"Fine-tuned model ready" if ft_available else "Base model only"}</div></div>',
    unsafe_allow_html=True,
)

# ── Hero: headline + 3D globe (reads last run's language choice) ──────────────
_prev_src = st.session_state.get("src_lang_select", "✨ Auto-Detect")
if _prev_src == "✨ Auto-Detect":
    _prev_src = st.session_state.get("_detected", "English")
if _prev_src not in SOURCE_LANGUAGES:
    _prev_src = "English"
_prev_tgt = st.session_state.get("tgt_lang_select", "Hindi")

hero_l, hero_r = st.columns([1, 1], gap="large")
with hero_l:
    st.markdown("""
    <div class="hero">
        <h1 class="hero-title">Translate the way people actually write.</h1>
        <p class="hero-subtitle">
            Detects your language, handles Hinglish with a fine-tuned model,
            transcribes speech, and reads the result aloud.
        </p>
        <div class="hero-stack">NLLB-200 · Whisper · gTTS · LoRA fine-tuning</div>
    </div>
    """, unsafe_allow_html=True)
with hero_r:
    render_globe(_prev_src, _prev_tgt)

if not ft_available:
    st.info(
        "Fine-tuned adapter not found. Run `python ml/finetune_lora.py` to enable "
        "the Hinglish→English fine-tuned model. Using base NLLB-200 for all translations.",
        icon="🔬",
    )

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_text, tab_voice = st.tabs(["Text", "Voice"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — TEXT TRANSLATION
# ─────────────────────────────────────────────────────────────────────────────
with tab_text:
    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown('<div class="section-label">Text to translate</div>', unsafe_allow_html=True)
        input_text = st.text_area(
            label="input_text_area",
            label_visibility="collapsed",
            placeholder="Type or paste text here… (any of 10 supported languages)",
            height=180,
            key="text_input",
        )

        # ── Language detection ────────────────────────────────────────────────
        detected_label, detected_conf = "Unknown", 0.0
        hinglish_detected = False

        if input_text and len(input_text.strip()) >= 2:
            detected_label, detected_conf = detect_language(input_text)
            hinglish_detected = is_hinglish(input_text)
            if detected_label != "Unknown":
                st.markdown(lang_badge_html(detected_label, detected_conf), unsafe_allow_html=True)
            if hinglish_detected:
                st.markdown(hinglish_badge_html(), unsafe_allow_html=True)

        # Remember detection so the globe can follow it on the next rerun
        st.session_state["_detected"] = detected_label if detected_label in SOURCE_LANGUAGES else "English"

        # ── Source language selector ──────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        src_lang_options = ["✨ Auto-Detect"] + list(SOURCE_LANGUAGES.keys())

        col_src, col_tgt = st.columns(2)
        with col_src:
            st.markdown('<div class="section-label">From</div>', unsafe_allow_html=True)
            selected_src_option = st.selectbox(
                "source_lang",
                label_visibility="collapsed",
                options=src_lang_options,
                index=0,
                key="src_lang_select",
            )

        # Resolve effective source language
        if selected_src_option == "✨ Auto-Detect":
            if detected_label != "Unknown" and detected_label in SOURCE_LANGUAGES:
                effective_src_lang = detected_label
            else:
                effective_src_lang = "English"
        else:
            effective_src_lang = selected_src_option

        with col_tgt:
            st.markdown('<div class="section-label">To</div>', unsafe_allow_html=True)
            # Default target to English unless source is English, then Hindi
            default_tgt_lang = "Hindi" if effective_src_lang == "English" else "English"
            tgt_options = list(TARGET_LANGUAGES.keys())
            tgt_index = tgt_options.index(default_tgt_lang) if default_tgt_lang in tgt_options else 0

            tgt_lang_name = st.selectbox(
                "target_lang",
                label_visibility="collapsed",
                options=tgt_options,
                index=tgt_index,
                key="tgt_lang_select",
            )

        # ── Fine-tuned model toggle ───────────────────────────────────────────
        can_use_finetuned = (
            ft_available
            and (effective_src_lang in ("Hindi", "Hinglish → English") or hinglish_detected)
            and tgt_lang_name == "English"
        )

        use_finetuned = False
        if ft_available:
            use_finetuned = st.toggle(
                "Use fine-tuned Hinglish→EN model",
                value=can_use_finetuned,
                disabled=not can_use_finetuned,
                help="Only available when source is Hindi/Hinglish and target is English. "
                     "Trained on code-switched Hinglish corpus with LoRA.",
                key="finetuned_toggle",
            )

        # ── Translate button ──────────────────────────────────────────────────
        translate_btn = st.button("Translate", key="translate_text_btn", use_container_width=True)

    # ── Right column: output ──────────────────────────────────────────────────
    with col_right:
        src_nllb = SOURCE_LANGUAGES[effective_src_lang][0]
        tgt_nllb = TARGET_LANGUAGES[tgt_lang_name][0]

        if translate_btn:
            if not input_text.strip():
                st.warning("Please enter some text to translate.")
            else:
                if use_finetuned:
                    # Side-by-side comparison
                    st.markdown('<div class="section-label blue">Base model (NLLB-200)</div>', unsafe_allow_html=True)
                    with st.spinner("Translating with base model…"):
                        base_result = translate(input_text, src_nllb, tgt_nllb, use_finetuned=False)
                    box(base_result)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="section-label purple">Fine-tuned model (LoRA)</div>', unsafe_allow_html=True)
                    with st.spinner("Translating with fine-tuned model…"):
                        ft_result = translate(input_text, src_nllb, tgt_nllb, use_finetuned=True)
                    box(ft_result, finetuned=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="section-label">Listen (fine-tuned)</div>', unsafe_allow_html=True)
                    speak(ft_result, tgt_nllb)

                else:
                    st.markdown('<div class="section-label blue">Translation</div>', unsafe_allow_html=True)
                    with st.spinner("Translating…"):
                        result = translate(input_text, src_nllb, tgt_nllb, use_finetuned=False)
                    box(result)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="section-label">Listen</div>', unsafe_allow_html=True)
                    speak(result, tgt_nllb)
        else:
            st.markdown("""
            <div class="glass-card" style="margin-top:1.6rem; text-align:center; color:#8d9bb8; padding:3rem 1rem;">
                Your translation will appear here.<br>
                <span style="font-size:.85rem;">
                    10 languages · auto-detects the source · fine-tuned Hinglish→English
                </span>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — VOICE TRANSLATION
# ─────────────────────────────────────────────────────────────────────────────
with tab_voice:
    st.markdown("<br>", unsafe_allow_html=True)
    col_v_left, col_v_right = st.columns(2, gap="large")

    with col_v_left:
        st.markdown('<div class="section-label">Upload audio</div>', unsafe_allow_html=True)
        uploaded_audio = st.file_uploader(
            "audio_upload",
            label_visibility="collapsed",
            type=["wav", "mp3", "m4a", "ogg", "flac"],
            key="audio_upload",
            help="Upload a WAV, MP3, M4A, OGG, or FLAC file",
        )

        if uploaded_audio:
            st.audio(uploaded_audio, format=f"audio/{uploaded_audio.name.split('.')[-1]}")

        st.markdown('<div class="section-label" style="margin-top:1.2rem;">Translate to</div>', unsafe_allow_html=True)
        tgt_lang_name_v = st.selectbox(
            "voice_target_lang",
            label_visibility="collapsed",
            options=list(TARGET_LANGUAGES.keys()),
            index=0,
            key="voice_tgt_lang",
        )

        voice_translate_btn = st.button("Transcribe and translate", key="voice_translate_btn", use_container_width=True)

    with col_v_right:
        if voice_translate_btn:
            if not uploaded_audio:
                st.warning("Please upload an audio file first.")
            else:
                # ── STT ───────────────────────────────────────────────────────
                st.markdown('<div class="section-label">Transcription</div>', unsafe_allow_html=True)
                with st.spinner("Transcribing audio…"):
                    from app.speech_to_text import transcribe
                    audio_bytes = uploaded_audio.read()
                    audio_ext = uploaded_audio.name.split(".")[-1]
                    transcribed_text, whisper_lang = transcribe(audio_bytes, audio_ext)

                box(transcribed_text)

                # Map Whisper lang code → NLLB source code
                src_nllb_v = WHISPER_TO_NLLB.get(whisper_lang, "eng_Latn")
                src_display = NLLB_TO_DISPLAY.get(src_nllb_v, whisper_lang)
                st.markdown(lang_badge_html(src_display, 0.95), unsafe_allow_html=True)

                # ── Translation ───────────────────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-label blue">Translation</div>', unsafe_allow_html=True)
                tgt_nllb_v = TARGET_LANGUAGES[tgt_lang_name_v][0]

                # Auto-use fine-tuned if Hinglish→EN
                _hinglish_v = is_hinglish(transcribed_text)
                _use_ft_v = ft_available and _hinglish_v and tgt_lang_name_v == "English"

                with st.spinner("Translating…"):
                    translated_v = translate(transcribed_text, src_nllb_v, tgt_nllb_v, use_finetuned=_use_ft_v)

                box(translated_v, finetuned=_use_ft_v)

                if _use_ft_v:
                    st.markdown('<span class="hinglish-badge">Used fine-tuned model</span>', unsafe_allow_html=True)

                # ── TTS ───────────────────────────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-label">Translated audio</div>', unsafe_allow_html=True)
                speak(translated_v, tgt_nllb_v)

        else:
            st.markdown("""
            <div class="glass-card" style="margin-top:1.6rem; text-align:center; color:#8d9bb8; padding:3rem 1rem;">
                Upload audio to see the transcript and translation.<br>
                <span style="font-size:.85rem;">Whisper STT → NLLB translation → gTTS audio</span>
            </div>
            """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#8d9bb8; font-size:.8rem; padding-bottom:1rem;">
    GenAI Translator · NLLB-200 + Whisper + gTTS + LoRA fine-tuning · Built with Streamlit
</div>
""", unsafe_allow_html=True)
