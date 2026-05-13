import os
import re

import gradio as gr
from dotenv import load_dotenv

from src.finetune import TutorLLM, LLMConfig

load_dotenv()

_FINETUNED_MODEL_ID = os.getenv("FINETUNED_MODEL_ID", "").strip()

LANGUAGES = [
    "English", "German", "French", "Spanish", "Italian",
    "Portuguese", "Turkish", "Arabic", "Chinese", "Japanese",
]
LEVELS = ["Beginner (A1-A2)", "Intermediate (B1-B2)", "Advanced (C1-C2)"]
QUIZ_TOPICS = [
    "General", "Food & Dining", "Travel", "Business",
    "Family", "Sports", "Health", "Technology",
]

_llm: TutorLLM | None = None


def get_llm():
    global _llm
    if _llm is None:
        if _FINETUNED_MODEL_ID:
            try:
                from src.local_model import LocalTutorLLM
                _llm = LocalTutorLLM(
                    model_id=_FINETUNED_MODEL_ID,
                    hf_token=os.getenv("HF_API_TOKEN"),
                )
                print(f"[app] Using fine-tuned model: {_FINETUNED_MODEL_ID}")
            except Exception as e:
                print(f"[app] Fine-tuned model failed to load ({e}), falling back to Groq.")
                _llm = None

        if _llm is None:
            cfg = LLMConfig(
                groq_api_key=os.getenv("GROQ_API_KEY"),
                groq_model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                finetuned_model_id=_FINETUNED_MODEL_ID or None,
            )
            _llm = TutorLLM(cfg)
            print("[app] Using Groq backend.")
    return _llm


# ── Score parser ──────────────────────────────────────────────────────────────

def parse_quiz_score(text: str) -> float:
    """Extract score as 0.0-1.0 from AI feedback text. Returns -1 if not found."""
    m = re.search(r'(\d+)\s*/\s*5', text)
    if m:
        return int(m.group(1)) / 5.0
    m = re.search(r'(\d+)\s+out\s+of\s+5', text, re.IGNORECASE)
    if m:
        return int(m.group(1)) / 5.0
    return -1.0


# ── Handlers ──────────────────────────────────────────────────────────────────

def handle_chat(message: str, history: list, native: str, language: str, level: str):
    message = message.strip()
    if not message:
        yield history, ""
        return
    # Convert messages format [{role, content}] to tuples [[user, bot]] for the LLM
    history_tuples = []
    msgs = history or []
    i = 0
    while i < len(msgs) - 1:
        if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
            history_tuples.append([msgs[i]["content"], msgs[i + 1]["content"]])
            i += 2
        else:
            i += 1
    new_history = list(msgs) + [{"role": "user", "content": message}]
    yield new_history, ""
    reply = ""
    try:
        for chunk in get_llm().chat_stream(
            message, history=history_tuples, native=native, language=language, level=level
        ):
            reply += chunk
            yield new_history + [{"role": "assistant", "content": reply}], ""
    except Exception as e:
        yield new_history + [{"role": "assistant", "content": f"Error: {e}"}], ""


def handle_grammar(text: str, native: str, language: str, level: str) -> str:
    if not text.strip():
        return "Please enter some text first."
    try:
        return get_llm().check_grammar(text, native=native, language=language, level=level)
    except Exception as e:
        return f"Error: {e}"


def handle_quiz(native: str, language: str, level: str, topic_dd: str, topic_custom: str):
    topic = topic_custom.strip() or topic_dd
    try:
        questions = get_llm().generate_quiz(
            native=native, language=language, level=level, topic=topic
        )
    except Exception as e:
        questions = f"Error: {e}"
    return questions, "", ""


def handle_quiz_check(quiz_text: str, user_answers: str,
                      native: str, language: str, level: str):
    if not quiz_text.strip():
        return "Please generate a quiz first.", -1.0
    if not user_answers.strip():
        return "Please write your answers before checking.", -1.0
    try:
        feedback = get_llm().check_quiz_answers(
            quiz_text, user_answers, native=native, language=language, level=level
        )
        return feedback, parse_quiz_score(feedback)
    except Exception as e:
        return f"Error: {e}", -1.0


def handle_quick_translate(text: str, from_lang: str, to_lang: str) -> str:
    if not text.strip():
        return "Please enter some text to translate."
    try:
        return get_llm().quick_translate(text, from_lang=from_lang, to_lang=to_lang)
    except Exception as e:
        return f"Error: {e}"


def handle_translation(original: str, translation: str,
                       from_lang: str, to_lang: str) -> str:
    if not original.strip() or not translation.strip():
        return "Please fill in both fields."
    try:
        return get_llm().check_translation(
            original, translation, from_lang=from_lang, to_lang=to_lang
        )
    except Exception as e:
        return f"Error: {e}"


# ── Animation JS — loaded via demo.load() so it actually executes ─────────────
# gr.HTML innerHTML does NOT execute <script> tags (browser security).
# demo.load(fn=None, js=...) runs the JS string as a function when the page loads.

ANIM_JS_DEF = r"""
() => {
window.triggerLinguaAnimation = function(score) {
    score = parseFloat(score);
    if (isNaN(score) || score < 0) return;

    var tier, colors, msg, msgColor, duration;
    if (score >= 0.8) {
        tier = 'legendary';
        colors = ['#f0d080','#fbbf24','#ffffff','#fef3c7','#f59e0b','#fcd34d'];
        msg = '🏆  Excellent Mastery!  🏆';
        msgColor = '#f0d080';
        duration = 320;
    } else if (score >= 0.5) {
        tier = 'worthy';
        colors = ['#60a5fa','#a78bfa','#34d399','#f472b6','#fbbf24','#38bdf8'];
        msg = '⭐  Great Work!  ⭐';
        msgColor = '#60a5fa';
        duration = 240;
    } else if (score >= 0.2) {
        tier = 'learning';
        colors = ['#6366f1','#8b5cf6','#4f46e5','#a78bfa','#7c3aed'];
        msg = '📚  Keep Improving!  📚';
        msgColor = '#a78bfa';
        duration = 200;
    } else {
        tier = 'defeated';
        colors = ['#4b5563','#6b7280','#9ca3af','#374151','#52525b'];
        msg = '💪  More Practice Ahead!  💪';
        msgColor = '#9ca3af';
        duration = 200;
    }

    /* canvas */
    var canvas = document.createElement('canvas');
    canvas.style.cssText = 'position:fixed;top:0;left:0;pointer-events:none;z-index:9998;';
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    canvas.style.width = '100vw';
    canvas.style.height = '100vh';
    document.body.appendChild(canvas);
    var ctx = canvas.getContext('2d');
    var W = canvas.width, H = canvas.height;

    /* flash overlay */
    var flash = document.createElement('div');
    var flashBg = tier==='legendary'?'rgba(240,208,128,0.14)':
                  tier==='worthy'?'rgba(96,165,250,0.1)':
                  tier==='learning'?'rgba(139,92,246,0.08)':'rgba(75,85,99,0.1)';
    flash.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:9997;transition:opacity 0.9s ease;background:'+flashBg+';';
    document.body.appendChild(flash);
    setTimeout(function(){ flash.style.opacity = '0'; }, 300);

    /* message */
    var msgEl = document.createElement('div');
    msgEl.textContent = msg;
    msgEl.style.cssText =
        'position:fixed;top:44%;left:50%;transform:translate(-50%,-50%);z-index:9999;'+
        'pointer-events:none;font-weight:800;letter-spacing:0.14em;text-align:center;'+
        'font-family:Inter,system-ui,sans-serif;white-space:nowrap;'+
        'font-size:clamp(1rem,2.8vw,1.9rem);'+
        'color:'+msgColor+';text-shadow:0 0 30px '+msgColor+',0 0 60px '+msgColor+';'+
        'opacity:0;transition:opacity 0.4s ease;';
    document.body.appendChild(msgEl);
    setTimeout(function(){ msgEl.style.opacity = '1'; }, 120);
    setTimeout(function(){ msgEl.style.opacity = '0'; }, (duration - 60) * 16);

    /* particles */
    var particles = [];

    function spawnBurst(x, y, n, spd, life, grav) {
        for (var i = 0; i < n; i++) {
            var a = Math.random() * Math.PI * 2;
            var s = spd * (0.3 + Math.random() * 0.7);
            var l = life * (0.6 + Math.random() * 0.8);
            particles.push({
                x:x, y:y,
                vx:Math.cos(a)*s, vy:Math.sin(a)*s,
                color:colors[Math.floor(Math.random()*colors.length)],
                size:1+Math.random()*3,
                life:l, maxLife:l, gravity:grav
            });
        }
    }

    /* initial burst */
    if (tier === 'legendary') {
        spawnBurst(W/2, H/2,  120, 15, 130, 0.18);
        spawnBurst(W*0.25, H*0.4, 60, 11, 100, 0.16);
        spawnBurst(W*0.75, H*0.4, 60, 11, 100, 0.16);
    } else if (tier === 'worthy') {
        spawnBurst(W/2, H/3,  80, 12, 100, 0.15);
        spawnBurst(W*0.2, H*0.5, 40, 10, 80, 0.13);
        spawnBurst(W*0.8, H*0.5, 40, 10, 80, 0.13);
    }

    var frame = 0;

    function animate() {
        if (frame >= duration) {
            canvas.remove(); flash.remove(); msgEl.remove();
            return;
        }
        ctx.clearRect(0, 0, W, H);

        /* ongoing spawning */
        if (tier === 'legendary') {
            if (frame % 14 === 0 && frame < 130)
                spawnBurst(W*0.1+Math.random()*W*0.8, Math.random()*H*0.55, 35, 11, 90, 0.17);
            if (frame % 3 === 0 && frame < 90)
                particles.push({x:Math.random()*W, y:-6,
                    vx:(Math.random()-0.5)*2, vy:3+Math.random()*4,
                    color:colors[Math.floor(Math.random()*colors.length)],
                    size:2+Math.random()*4, life:70+Math.random()*50, maxLife:120, gravity:0.09});
        } else if (tier === 'worthy') {
            if (frame % 22 === 0 && frame < 160)
                spawnBurst(W*0.1+Math.random()*W*0.8, H*0.1+Math.random()*H*0.55, 30, 10, 80, 0.14);
        } else if (tier === 'learning') {
            if (frame % 5 === 0)
                for (var j = 0; j < 5; j++)
                    particles.push({x:W/2+(Math.random()-0.5)*180, y:H*0.88,
                        vx:(Math.random()-0.5)*4, vy:-(2+Math.random()*6),
                        color:colors[Math.floor(Math.random()*colors.length)],
                        size:1+Math.random()*2.5, life:45+Math.random()*55, maxLife:100, gravity:0.03});
        } else {
            if (frame % 7 === 0)
                for (var k = 0; k < 4; k++)
                    particles.push({x:W/2+(Math.random()-0.5)*350, y:H*0.75+Math.random()*H*0.2,
                        vx:(Math.random()-0.5)*1.5, vy:-(0.4+Math.random()*1.5),
                        color:colors[Math.floor(Math.random()*colors.length)],
                        size:5+Math.random()*14, life:100+Math.random()*100, maxLife:200, gravity:-0.012});
        }

        /* draw */
        for (var i = particles.length - 1; i >= 0; i--) {
            var p = particles[i];
            p.x += p.vx; p.y += p.vy; p.vy += p.gravity;
            p.vx *= (tier === 'defeated' ? 0.994 : 0.989);
            p.life--;
            if (p.life <= 0) { particles.splice(i, 1); continue; }
            var alpha = Math.pow(p.life / p.maxLife, tier === 'defeated' ? 0.4 : 1.0);
            var radius = p.size * (tier === 'defeated' ? (1 + (1 - p.life/p.maxLife) * 2.5) : 1);
            ctx.globalAlpha = alpha;
            ctx.fillStyle = p.color;
            ctx.beginPath();
            ctx.arc(p.x, p.y, Math.max(0.1, radius), 0, Math.PI * 2);
            ctx.fill();
        }
        ctx.globalAlpha = 1;
        frame++;
        requestAnimationFrame(animate);
    }

    animate();
};
}
"""


# ── Theme & CSS ───────────────────────────────────────────────────────────────

CSS = """
footer { display: none !important; }

body { background: #07050e !important; }

/* Page-level ambient colour bleed — gives depth to the otherwise flat dark bg */
.gradio-container {
    background:
        radial-gradient(ellipse 75% 45% at 10% 5%,  rgba(59,130,246,0.055) 0%, transparent 55%),
        radial-gradient(ellipse 65% 50% at 90% 95%, rgba(139,92,246,0.05)  0%, transparent 55%),
        radial-gradient(ellipse 90% 35% at 50% -8%, rgba(196,148,20,0.07)  0%, transparent 50%),
        transparent !important;
    max-width: 1080px !important;
    margin: 0 auto !important;
    padding: 24px 20px !important;
}

/* ── Global transitions ── */
*, *::before, *::after {
    transition: background-color 0.25s ease, border-color 0.25s ease,
                box-shadow 0.25s ease, color 0.2s ease !important;
}
button {
    transition: transform 0.18s cubic-bezier(0.34,1.56,0.64,1),
                box-shadow 0.2s ease, background 0.22s ease !important;
}
button:hover  { transform: translateY(-2px) !important; }
button:active { transform: translateY(0) scale(0.97) !important; }

/* ── Scrollbars ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.03); border-radius: 3px; }
::-webkit-scrollbar-thumb { background: rgba(196,148,20,0.35); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(196,148,20,0.6); }

/* ── Header ── */
.app-header {
    background:
        radial-gradient(ellipse at 50% 0%, rgba(196,148,20,0.13) 0%, transparent 65%),
        linear-gradient(170deg, #1c1409 0%, #2a1c07 50%, #1a1005 100%);
    border: 1px solid rgba(196,148,20,0.5);
    border-radius: 20px;
    padding: 36px 40px 30px;
    text-align: center;
    margin-bottom: 0;
    overflow: hidden;
    position: relative;
    animation: breathe 5s ease-in-out infinite alternate;
}
.app-header::after {
    content: '';
    position: absolute;
    bottom: 0; left: 10%; right: 10%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(196,148,20,0.6), transparent);
}
@keyframes breathe {
    from { box-shadow: 0 0 40px rgba(196,148,20,0.14), 0 0 0 1px rgba(196,148,20,0.3); }
    to   { box-shadow: 0 0 80px rgba(196,148,20,0.28), 0 0 0 1px rgba(196,148,20,0.6); }
}
.app-header h1 {
    color: #f0d080 !important;
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.07em !important;
    margin: 0 0 8px 0 !important;
    text-shadow: 0 0 40px rgba(240,208,128,0.55), 0 2px 4px rgba(0,0,0,0.5);
}
.app-header p {
    color: #8a6c2a !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.25em !important;
    text-transform: uppercase !important;
    margin: 0 !important;
}

/* ── Stats bar ── */
.stats-row {
    display: flex; gap: 10px; justify-content: center;
    padding: 16px 0 4px; flex-wrap: wrap;
}
.stat-pill {
    background: rgba(18,12,6,0.9);
    border: 1px solid rgba(196,148,20,0.28);
    border-radius: 50px; padding: 7px 16px;
    display: inline-flex; align-items: center; gap: 7px;
    backdrop-filter: blur(8px); cursor: default;
}
.stat-pill:hover {
    border-color: rgba(196,148,20,0.65) !important;
    box-shadow: 0 0 18px rgba(196,148,20,0.2);
    transform: translateY(-1px) !important;
}
.stat-icon { font-size: 1rem; }
.stat-text { color: #c8a85a; font-size: 0.82rem; font-weight: 600; letter-spacing: 0.03em; }
.stat-text b { color: #f0d080; }

/* ── Language selector bar ── */
.lang-bar {
    background: rgba(14,10,22,0.85) !important;
    border: 1px solid rgba(196,148,20,0.25) !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important;
}

/* ── Section description card base ── */
.section-card {
    border-left: 3px solid rgba(196,148,20,0.7);
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    margin-bottom: 8px;
}
.section-card b    { font-size: 0.95rem; display: block; margin-bottom: 2px; }
.section-card span { font-size: 0.84rem; display: block; margin-top: 2px; opacity: 0.85; }

/* ── Tab nav ── */
.tab-nav {
    background: rgba(10,7,18,0.95) !important;
    border: 1px solid rgba(196,148,20,0.2) !important;
    border-bottom: none !important;
    border-radius: 14px 14px 0 0 !important;
    padding: 6px 8px 0 !important;
    gap: 3px !important;
    backdrop-filter: blur(8px);
}
.tab-nav button {
    color: #5a4a28 !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 10px 18px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.02em !important;
    border: none !important;
    background: transparent !important;
}
.tab-nav button:hover {
    color: #c8a85a !important;
    background: rgba(196,148,20,0.08) !important;
    transform: none !important;
    box-shadow: none !important;
}
.tab-nav button:nth-child(1).selected {
    color: #60a5fa !important; border-bottom: 2px solid #3b82f6 !important;
    background: rgba(59,130,246,0.12) !important;
}
.tab-nav button:nth-child(2).selected {
    color: #34d399 !important; border-bottom: 2px solid #10b981 !important;
    background: rgba(16,185,129,0.12) !important;
}
.tab-nav button:nth-child(3).selected {
    color: #a78bfa !important; border-bottom: 2px solid #8b5cf6 !important;
    background: rgba(139,92,246,0.12) !important;
}
.tab-nav button:nth-child(4).selected {
    color: #22d3ee !important; border-bottom: 2px solid #06b6d4 !important;
    background: rgba(6,182,212,0.12) !important;
}
.tab-nav button:nth-child(5).selected {
    color: #fbbf24 !important; border-bottom: 2px solid #f59e0b !important;
    background: rgba(245,158,11,0.12) !important;
}

/* ── Tab panel wrapper — transparent; each inner column is the styled container ── */
.tabitem {
    background: transparent !important;
    border: none !important;
    border-radius: 0 0 14px 14px !important;
    padding: 8px !important;
}

/* ═══════════════════════════════════════════
   TAB 1 — SPEAK  (Deep Ocean Blue)
═══════════════════════════════════════════ */
.tab-speak {
    background: linear-gradient(155deg, #060e1f 0%, #0c1c3e 55%, #050b18 100%) !important;
    border: 1px solid rgba(59,130,246,0.28) !important;
    border-radius: 10px !important;
    padding: 20px !important;
    position: relative !important;
    box-shadow: inset 0 2px 8px rgba(59,130,246,0.12), 0 0 55px rgba(59,130,246,0.06) !important;
}
.tab-speak::before {
    content: '';
    position: absolute;
    top: 0; left: 8%; right: 8%; height: 2px;
    background: linear-gradient(90deg, transparent, #2563eb 25%, #60a5fa 50%, #2563eb 75%, transparent);
    border-radius: 2px;
}
.tab-speak .section-card {
    background: linear-gradient(135deg, rgba(29,78,216,0.14), rgba(15,40,100,0.08)) !important;
    border: 1px solid rgba(59,130,246,0.2) !important;
    border-left: 3px solid #3b82f6 !important;
}
.tab-speak .section-card b    { color: #60a5fa !important; }
.tab-speak .section-card span { color: #93c5fd !important; }
.tab-speak button.primary {
    background: linear-gradient(135deg, #1e3a8a, #2563eb) !important;
    color: #fff !important;
    border: 1px solid rgba(96,165,250,0.35) !important;
    border-radius: 10px !important;
}
.tab-speak button.primary:hover {
    background: linear-gradient(135deg, #1d4ed8, #3b82f6) !important;
    box-shadow: 0 8px 28px rgba(59,130,246,0.5) !important;
}
.tab-speak button.secondary {
    border-color: rgba(59,130,246,0.4) !important;
    color: #60a5fa !important;
}
.tab-speak button.secondary:hover {
    border-color: rgba(59,130,246,0.75) !important;
    box-shadow: 0 4px 14px rgba(59,130,246,0.25) !important;
    color: #93c5fd !important;
}
.tab-speak textarea, .tab-speak input {
    border-color: rgba(59,130,246,0.25) !important;
    background: rgba(4,8,20,0.92) !important;
}
.tab-speak textarea:focus, .tab-speak input:focus {
    border-color: rgba(59,130,246,0.72) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.18) !important;
}
.tab-speak .chatbot {
    background: rgba(3,6,18,0.95) !important;
    border: 1px solid rgba(59,130,246,0.2) !important;
    border-radius: 12px !important;
}

/* ═══════════════════════════════════════════
   TAB 2 — GRAMMAR  (Forest Emerald)
═══════════════════════════════════════════ */
.tab-grammar {
    background: linear-gradient(155deg, #030e08 0%, #071a0e 55%, #030c06 100%) !important;
    border: 1px solid rgba(16,185,129,0.28) !important;
    border-radius: 10px !important;
    padding: 20px !important;
    position: relative !important;
    box-shadow: inset 0 2px 8px rgba(16,185,129,0.12), 0 0 55px rgba(16,185,129,0.05) !important;
}
.tab-grammar::before {
    content: '';
    position: absolute;
    top: 0; left: 8%; right: 8%; height: 2px;
    background: linear-gradient(90deg, transparent, #059669 25%, #34d399 50%, #059669 75%, transparent);
    border-radius: 2px;
}
.tab-grammar .section-card {
    background: linear-gradient(135deg, rgba(5,150,105,0.13), rgba(3,80,50,0.08)) !important;
    border: 1px solid rgba(16,185,129,0.2) !important;
    border-left: 3px solid #10b981 !important;
}
.tab-grammar .section-card b    { color: #34d399 !important; }
.tab-grammar .section-card span { color: #6ee7b7 !important; }
.tab-grammar button.primary {
    background: linear-gradient(135deg, #064e3b, #059669) !important;
    color: #fff !important;
    border: 1px solid rgba(52,211,153,0.35) !important;
    border-radius: 10px !important;
}
.tab-grammar button.primary:hover {
    background: linear-gradient(135deg, #065f46, #10b981) !important;
    box-shadow: 0 8px 28px rgba(16,185,129,0.5) !important;
}
.tab-grammar button.secondary {
    border-color: rgba(16,185,129,0.4) !important;
    color: #34d399 !important;
}
.tab-grammar button.secondary:hover {
    border-color: rgba(16,185,129,0.75) !important;
    box-shadow: 0 4px 14px rgba(16,185,129,0.25) !important;
    color: #6ee7b7 !important;
}
.tab-grammar textarea, .tab-grammar input {
    border-color: rgba(16,185,129,0.25) !important;
    background: rgba(2,7,4,0.93) !important;
}
.tab-grammar textarea:focus, .tab-grammar input:focus {
    border-color: rgba(16,185,129,0.72) !important;
    box-shadow: 0 0 0 3px rgba(16,185,129,0.18) !important;
}

/* ═══════════════════════════════════════════
   TAB 3 — QUIZ  (Cosmic Violet)
═══════════════════════════════════════════ */
.tab-quiz {
    background: linear-gradient(155deg, #080518 0%, #12082c 55%, #060412 100%) !important;
    border: 1px solid rgba(139,92,246,0.28) !important;
    border-radius: 10px !important;
    padding: 20px !important;
    position: relative !important;
    box-shadow: inset 0 2px 8px rgba(139,92,246,0.14), 0 0 55px rgba(139,92,246,0.07) !important;
}
.tab-quiz::before {
    content: '';
    position: absolute;
    top: 0; left: 8%; right: 8%; height: 2px;
    background: linear-gradient(90deg, transparent, #7c3aed 25%, #a78bfa 50%, #7c3aed 75%, transparent);
    border-radius: 2px;
}
.tab-quiz .section-card {
    background: linear-gradient(135deg, rgba(109,40,217,0.13), rgba(60,20,120,0.08)) !important;
    border: 1px solid rgba(139,92,246,0.2) !important;
    border-left: 3px solid #8b5cf6 !important;
}
.tab-quiz .section-card b    { color: #a78bfa !important; }
.tab-quiz .section-card span { color: #c4b5fd !important; }
.tab-quiz button.primary {
    background: linear-gradient(135deg, #4c1d95, #7c3aed) !important;
    color: #fff !important;
    border: 1px solid rgba(167,139,250,0.35) !important;
    border-radius: 10px !important;
}
.tab-quiz button.primary:hover {
    background: linear-gradient(135deg, #5b21b6, #8b5cf6) !important;
    box-shadow: 0 8px 28px rgba(139,92,246,0.5) !important;
}
.tab-quiz button.secondary {
    border-color: rgba(139,92,246,0.4) !important;
    color: #a78bfa !important;
}
.tab-quiz button.secondary:hover {
    border-color: rgba(139,92,246,0.75) !important;
    box-shadow: 0 4px 14px rgba(139,92,246,0.25) !important;
    color: #c4b5fd !important;
}
.tab-quiz textarea, .tab-quiz input {
    border-color: rgba(139,92,246,0.25) !important;
    background: rgba(4,2,12,0.93) !important;
}
.tab-quiz textarea:focus, .tab-quiz input:focus {
    border-color: rgba(139,92,246,0.72) !important;
    box-shadow: 0 0 0 3px rgba(139,92,246,0.18) !important;
}

/* ═══════════════════════════════════════════
   TAB 4 — TRANSLATE  (Arctic Cyan)
═══════════════════════════════════════════ */
.tab-qtranslate {
    background: linear-gradient(155deg, #030e16 0%, #071a24 55%, #030c10 100%) !important;
    border: 1px solid rgba(6,182,212,0.28) !important;
    border-radius: 10px !important;
    padding: 20px !important;
    position: relative !important;
    box-shadow: inset 0 2px 8px rgba(6,182,212,0.12), 0 0 55px rgba(6,182,212,0.05) !important;
}
.tab-qtranslate::before {
    content: '';
    position: absolute;
    top: 0; left: 8%; right: 8%; height: 2px;
    background: linear-gradient(90deg, transparent, #0891b2 25%, #22d3ee 50%, #0891b2 75%, transparent);
    border-radius: 2px;
}
.tab-qtranslate .section-card {
    background: linear-gradient(135deg, rgba(8,145,178,0.13), rgba(3,80,100,0.08)) !important;
    border: 1px solid rgba(6,182,212,0.2) !important;
    border-left: 3px solid #06b6d4 !important;
}
.tab-qtranslate .section-card b    { color: #22d3ee !important; }
.tab-qtranslate .section-card span { color: #67e8f9 !important; }
.tab-qtranslate button.primary {
    background: linear-gradient(135deg, #164e63, #0e7490) !important;
    color: #fff !important;
    border: 1px solid rgba(34,211,238,0.35) !important;
    border-radius: 10px !important;
}
.tab-qtranslate button.primary:hover {
    background: linear-gradient(135deg, #155e75, #0891b2) !important;
    box-shadow: 0 8px 28px rgba(6,182,212,0.5) !important;
}
.tab-qtranslate button.secondary {
    border-color: rgba(6,182,212,0.4) !important;
    color: #22d3ee !important;
}
.tab-qtranslate button.secondary:hover {
    border-color: rgba(6,182,212,0.75) !important;
    box-shadow: 0 4px 14px rgba(6,182,212,0.25) !important;
    color: #67e8f9 !important;
}
.tab-qtranslate textarea, .tab-qtranslate input {
    border-color: rgba(6,182,212,0.25) !important;
    background: rgba(2,7,10,0.93) !important;
}
.tab-qtranslate textarea:focus, .tab-qtranslate input:focus {
    border-color: rgba(6,182,212,0.72) !important;
    box-shadow: 0 0 0 3px rgba(6,182,212,0.18) !important;
}

/* ═══════════════════════════════════════════
   TAB 5 — PRACTICE  (Warm Amber)
═══════════════════════════════════════════ */
.tab-practice {
    background: linear-gradient(155deg, #130a02 0%, #201204 55%, #0e0701 100%) !important;
    border: 1px solid rgba(245,158,11,0.28) !important;
    border-radius: 10px !important;
    padding: 20px !important;
    position: relative !important;
    box-shadow: inset 0 2px 8px rgba(245,158,11,0.12), 0 0 55px rgba(245,158,11,0.06) !important;
}
.tab-practice::before {
    content: '';
    position: absolute;
    top: 0; left: 8%; right: 8%; height: 2px;
    background: linear-gradient(90deg, transparent, #b45309 25%, #f59e0b 50%, #b45309 75%, transparent);
    border-radius: 2px;
}
.tab-practice .section-card {
    background: linear-gradient(135deg, rgba(180,83,9,0.13), rgba(100,40,4,0.08)) !important;
    border: 1px solid rgba(245,158,11,0.2) !important;
    border-left: 3px solid #f59e0b !important;
}
.tab-practice .section-card b    { color: #fbbf24 !important; }
.tab-practice .section-card span { color: #fde68a !important; }
.tab-practice button.primary {
    background: linear-gradient(135deg, #92400e, #b45309) !important;
    color: #fff !important;
    border: 1px solid rgba(251,191,36,0.35) !important;
    border-radius: 10px !important;
}
.tab-practice button.primary:hover {
    background: linear-gradient(135deg, #b45309, #d97706) !important;
    box-shadow: 0 8px 28px rgba(245,158,11,0.5) !important;
}
.tab-practice button.secondary {
    border-color: rgba(245,158,11,0.4) !important;
    color: #fbbf24 !important;
}
.tab-practice button.secondary:hover {
    border-color: rgba(245,158,11,0.75) !important;
    box-shadow: 0 4px 14px rgba(245,158,11,0.25) !important;
    color: #fde68a !important;
}
.tab-practice textarea, .tab-practice input {
    border-color: rgba(245,158,11,0.25) !important;
    background: rgba(9,4,1,0.93) !important;
}
.tab-practice textarea:focus, .tab-practice input:focus {
    border-color: rgba(245,158,11,0.72) !important;
    box-shadow: 0 0 0 3px rgba(245,158,11,0.18) !important;
}
"""

THEME = gr.themes.Base(
    primary_hue="amber",
    secondary_hue="yellow",
    neutral_hue="stone",
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#09070f",
    block_background_fill="rgba(18,13,28,0.82)",
    block_border_color="rgba(196,148,20,0.2)",
    input_background_fill="rgba(8,6,14,0.92)",
    button_primary_background_fill="linear-gradient(135deg,#b8890e,#d4a820)",
    button_primary_background_fill_hover="linear-gradient(135deg,#c89818,#e8b830)",
    button_primary_text_color="#1a1005",
    button_secondary_background_fill="rgba(20,14,30,0.9)",
    button_secondary_border_color="rgba(196,148,20,0.4)",
    button_secondary_text_color="#c8a85a",
    color_accent="#c49414",
    color_accent_soft="rgba(196,148,20,0.15)",
    border_color_primary="rgba(196,148,20,0.2)",
    border_color_accent="rgba(196,148,20,0.5)",
    shadow_drop_lg="0 8px 32px rgba(0,0,0,0.6)",
    block_label_text_color="#c8a85a",
    block_title_text_color="#f0d080",
    body_text_color="#e8dcc8",
    body_text_color_subdued="#8a7040",
    input_border_color="rgba(196,148,20,0.28)",
    input_border_color_focus="rgba(196,148,20,0.75)",
    input_shadow_focus="0 0 0 3px rgba(196,148,20,0.2)",
)

STATS_HTML = """
<div class="stats-row">
  <div class="stat-pill"><span class="stat-icon">🌐</span><span class="stat-text"><b>10</b> Languages</span></div>
  <div class="stat-pill"><span class="stat-icon">🤖</span><span class="stat-text"><b>LLaMA 3.1</b> Powered</span></div>
  <div class="stat-pill"><span class="stat-icon">⚡</span><span class="stat-text"><b>Real-time</b> Streaming</span></div>
  <div class="stat-pill"><span class="stat-icon">📊</span><span class="stat-text"><b>ML</b> Classifier</span></div>
  <div class="stat-pill"><span class="stat-icon">🎓</span><span class="stat-text"><b>Adaptive</b> Learning</span></div>
</div>
"""

def tab_card(icon: str, title: str, desc: str) -> str:
    return (
        f'<div class="section-card">'
        f'<b>{icon} {title}</b>'
        f'<span>{desc}</span>'
        f'</div>'
    )


# ── UI ────────────────────────────────────────────────────────────────────────

def create_app() -> gr.Blocks:
    with gr.Blocks(title="Lingua Arcana") as demo:

        # Header
        gr.Markdown(
            '<div class="app-header">'
            '<h1>📜 LINGUA ARCANA</h1>'
            '<p>AI-Powered Language Tutor &nbsp;·&nbsp; 10 Languages &nbsp;·&nbsp; Powered by LLaMA 3.1</p>'
            '</div>'
        )
        gr.Markdown(STATS_HTML)

        # Language selectors
        with gr.Row(elem_classes=["lang-bar"]):
            native_sel = gr.Dropdown(choices=LANGUAGES, value="English",
                                     label="Your Language", scale=2)
            lang_sel   = gr.Dropdown(choices=LANGUAGES, value="German",
                                     label="Language to Learn", scale=2)
            level_sel  = gr.Dropdown(choices=LEVELS, value="Beginner (A1-A2)",
                                     label="Your Level", scale=2)

        with gr.Tabs():

            # ── Speak ──────────────────────────────────────────────────────
            with gr.Tab("💬 Speak"):
                with gr.Column(elem_classes=["tab-speak"]):
                    gr.Markdown(tab_card("💬", "Conversational Practice",
                        "Chat with Arcana in your target language. Errors are corrected gently "
                        "with explanations in your native language."))
                    chatbot = gr.Chatbot(height=400, show_label=False)
                    chat_input = gr.Textbox(
                        placeholder="Write something and press Enter to begin…",
                        label="Your message", lines=2)
                    with gr.Row():
                        send_btn  = gr.Button("Send",  variant="primary",    scale=3)
                        clear_btn = gr.Button("Clear", variant="secondary",  scale=1)
                    send_btn.click(handle_chat,
                        [chat_input, chatbot, native_sel, lang_sel, level_sel],
                        [chatbot, chat_input])
                    chat_input.submit(handle_chat,
                        [chat_input, chatbot, native_sel, lang_sel, level_sel],
                        [chatbot, chat_input])
                    clear_btn.click(lambda: ([], ""), outputs=[chatbot, chat_input])

            # ── Grammar ────────────────────────────────────────────────────
            with gr.Tab("✏️ Grammar"):
                with gr.Column(elem_classes=["tab-grammar"]):
                    gr.Markdown(tab_card("✏️", "Grammar Analysis",
                        "Submit any sentence in your target language. Arcana identifies errors, "
                        "explains the rules broken, and gives you a personalised grammar tip."))
                    grammar_input = gr.Textbox(
                        label="Your text",
                        placeholder="Write a sentence in your target language…",
                        lines=4)
                    check_btn = gr.Button("Analyse Grammar", variant="primary")
                    grammar_out = gr.Textbox(label="Analysis & Feedback",
                                             lines=13, interactive=False)
                    check_btn.click(handle_grammar,
                        [grammar_input, native_sel, lang_sel, level_sel], grammar_out)

            # ── Quiz ───────────────────────────────────────────────────────
            with gr.Tab("📚 Quiz"):
                with gr.Column(elem_classes=["tab-quiz"]):
                    gr.Markdown(tab_card("📚", "Vocabulary Quiz",
                        "Questions are generated with answers hidden. Submit your answers to "
                        "receive a score — and unlock a spell animation based on how well you did!"))
                    with gr.Row():
                        topic_dd = gr.Dropdown(choices=QUIZ_TOPICS, value="General",
                                               label="Topic", scale=2)
                        topic_custom = gr.Textbox(
                            label="Custom topic (optional)",
                            placeholder="e.g. weather, colours…", scale=3)
                    quiz_btn  = gr.Button("Generate Quiz", variant="primary")
                    quiz_out  = gr.Textbox(label="Questions", lines=12, interactive=False)
                    user_answers = gr.Textbox(
                        label="Your Answers",
                        placeholder="1. answer\n2. a / b / c / d\n3. …",
                        lines=5)
                    check_btn_quiz = gr.Button("Submit Answers ✨", variant="primary")
                    quiz_feedback  = gr.Textbox(label="Results & Feedback",
                                                lines=13, interactive=False)
                    quiz_score = gr.Number(value=-1, visible=False)

                    quiz_btn.click(handle_quiz,
                        [native_sel, lang_sel, level_sel, topic_dd, topic_custom],
                        [quiz_out, user_answers, quiz_feedback])

                    check_btn_quiz.click(
                        handle_quiz_check,
                        [quiz_out, user_answers, native_sel, lang_sel, level_sel],
                        [quiz_feedback, quiz_score],
                    ).then(
                        fn=None,
                        inputs=[quiz_score],
                        outputs=[],
                        js="(score) => { if(window.triggerLinguaAnimation) window.triggerLinguaAnimation(score); }",
                    )

            # ── Quick Translate ────────────────────────────────────────────
            with gr.Tab("🔄 Translate"):
                with gr.Column(elem_classes=["tab-qtranslate"]):
                    gr.Markdown(tab_card("🔄", "Quick Translation",
                        "Instantly translate any text between any two supported languages. "
                        "Includes vocabulary breakdown and a grammar note to help you learn."))
                    with gr.Row():
                        qt_from = gr.Dropdown(choices=LANGUAGES, value="English",
                                              label="From", scale=1)
                        qt_to   = gr.Dropdown(choices=LANGUAGES, value="German",
                                              label="To", scale=1)
                    qt_input = gr.Textbox(label="Text to translate",
                                          placeholder="Type anything you want to translate…",
                                          lines=3)
                    qt_btn = gr.Button("Translate", variant="primary")
                    qt_out = gr.Textbox(label="Translation + Vocabulary Notes",
                                        lines=17, interactive=False)
                    qt_btn.click(handle_quick_translate, [qt_input, qt_from, qt_to], qt_out)

            # ── Translation Practice ───────────────────────────────────────
            with gr.Tab("🌍 Practice"):
                with gr.Column(elem_classes=["tab-practice"]):
                    gr.Markdown(tab_card("🌍", "Translation Practice",
                        "Test yourself — write your own translation attempt and receive a "
                        "scored evaluation with corrections, strengths, and alternative phrasings."))
                    with gr.Row():
                        from_lang = gr.Dropdown(choices=LANGUAGES, value="English",
                                                label="From", scale=1)
                        to_lang   = gr.Dropdown(choices=LANGUAGES, value="German",
                                                label="To", scale=1)
                    original_txt = gr.Textbox(label="Original text",
                                               placeholder="The sentence you want to translate…",
                                               lines=3)
                    user_trans = gr.Textbox(label="Your translation",
                                             placeholder="Write your attempt here…",
                                             lines=3)
                    trans_btn = gr.Button("Evaluate Translation", variant="primary")
                    trans_out = gr.Textbox(label="Score & Feedback",
                                           lines=14, interactive=False)
                    trans_btn.click(handle_translation,
                        [original_txt, user_trans, from_lang, to_lang], trans_out)

        # Define animation function in browser on page load
        demo.load(fn=None, inputs=[], outputs=[], js=ANIM_JS_DEF)

    return demo


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=False,
        theme=THEME,
        css=CSS,
    )
