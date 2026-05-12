import os

import gradio as gr
from dotenv import load_dotenv

from src.finetune import TutorLLM, LLMConfig

load_dotenv()

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


def get_llm() -> TutorLLM:
    global _llm
    if _llm is None:
        cfg = LLMConfig(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            groq_model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        )
        _llm = TutorLLM(cfg)
    return _llm


# ── Handlers ──────────────────────────────────────────────────────────────────

def handle_chat(message: str, history: list, native: str, language: str, level: str):
    message = message.strip()
    if not message:
        yield history, ""
        return
    new_history = history + [[message, None]]
    yield new_history, ""
    reply = ""
    try:
        for chunk in get_llm().chat_stream(
            message, history=history, native=native, language=language, level=level
        ):
            reply += chunk
            new_history[-1][1] = reply
            yield new_history, ""
    except Exception as e:
        new_history[-1][1] = f"Error: {e}"
        yield new_history, ""


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


def handle_quiz_check(quiz_text: str, user_answers: str, native: str, language: str, level: str) -> str:
    if not quiz_text.strip():
        return "Please generate a quiz first."
    if not user_answers.strip():
        return "Please write your answers before checking."
    try:
        return get_llm().check_quiz_answers(
            quiz_text, user_answers, native=native, language=language, level=level
        )
    except Exception as e:
        return f"Error: {e}"


def handle_quick_translate(text: str, from_lang: str, to_lang: str) -> str:
    if not text.strip():
        return "Please enter some text to translate."
    try:
        return get_llm().quick_translate(text, from_lang=from_lang, to_lang=to_lang)
    except Exception as e:
        return f"Error: {e}"


def handle_translation(original: str, translation: str, from_lang: str, to_lang: str) -> str:
    if not original.strip() or not translation.strip():
        return "Please fill in both fields."
    try:
        return get_llm().check_translation(
            original, translation, from_lang=from_lang, to_lang=to_lang
        )
    except Exception as e:
        return f"Error: {e}"


# ── Theme & CSS ───────────────────────────────────────────────────────────────

CSS = """
/* ── Variables & page ───────────────────────────────────────────── */
:root {
    --body-background-fill: #09070f;
    --block-background-fill: rgba(18, 13, 28, 0.82);
    --block-border-color: rgba(196, 148, 20, 0.18);
    --block-border-width: 1px;
    --block-radius: 14px;
    --block-shadow: 0 4px 24px rgba(0,0,0,0.5);
    --input-background-fill: rgba(8, 6, 14, 0.92);
    --input-border-color: rgba(196, 148, 20, 0.28);
    --input-border-color-focus: rgba(196, 148, 20, 0.75);
    --input-shadow-focus: 0 0 0 3px rgba(196, 148, 20, 0.2);
    --border-color-primary: rgba(196, 148, 20, 0.2);
    --color-text-body: #e8dcc8;
    --color-text-label: #c8a85a;
    --button-primary-background-fill: linear-gradient(135deg, #b8890e 0%, #d4a820 100%);
    --button-primary-background-fill-hover: linear-gradient(135deg, #c89818 0%, #e8b830 100%);
    --button-primary-text-color: #1a1005;
    --button-secondary-background-fill: rgba(20, 14, 30, 0.9);
    --button-secondary-border-color: rgba(196, 148, 20, 0.4);
    --button-secondary-text-color: #c8a85a;
    --color-accent: #c49414;
    --slider-color: #c49414;
    --table-even-background-fill: rgba(18,13,28,0.6);
    --table-odd-background-fill: rgba(12,9,20,0.6);
}

footer { display: none !important; }

body {
    background: #09070f !important;
}

.gradio-container {
    background: transparent !important;
    max-width: 1080px !important;
    margin: 0 auto !important;
    padding: 24px 20px !important;
}

/* ── Global smooth transitions ─────────────────────────────────── */
*, *::before, *::after {
    transition: background-color 0.25s ease, border-color 0.25s ease,
                box-shadow 0.25s ease, color 0.2s ease !important;
}

button {
    transition: transform 0.18s cubic-bezier(0.34, 1.56, 0.64, 1),
                box-shadow 0.2s ease,
                background 0.2s ease !important;
}
button:hover {
    transform: translateY(-2px) !important;
}
button:active {
    transform: translateY(0px) scale(0.97) !important;
}
button.primary:hover {
    box-shadow: 0 8px 28px rgba(196, 148, 20, 0.45) !important;
}
button.secondary:hover {
    box-shadow: 0 4px 14px rgba(196, 148, 20, 0.2) !important;
    border-color: rgba(196, 148, 20, 0.7) !important;
    color: #f0d080 !important;
}

/* Dropdown / select animation */
.wrap, select, input, textarea {
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

/* ── Scrollbars ─────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.03); border-radius: 3px; }
::-webkit-scrollbar-thumb { background: rgba(196,148,20,0.35); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(196,148,20,0.6); }

/* ── Header (breathing Erdtree glow) ───────────────────────────── */
.app-header {
    background:
        radial-gradient(ellipse at 50% 0%, rgba(196,148,20,0.12) 0%, transparent 65%),
        linear-gradient(170deg, #1c1409 0%, #2a1c07 50%, #1a1005 100%);
    border: 1px solid rgba(196,148,20,0.5);
    border-radius: 20px;
    padding: 36px 40px 30px;
    text-align: center;
    margin-bottom: 0;
    position: relative;
    overflow: hidden;
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

/* ── Stats badges ──────────────────────────────────────────────── */
.stats-row {
    display: flex;
    gap: 10px;
    justify-content: center;
    padding: 18px 0 4px;
    flex-wrap: wrap;
}
.stat-pill {
    background: rgba(18, 12, 6, 0.9);
    border: 1px solid rgba(196,148,20,0.28);
    border-radius: 50px;
    padding: 7px 16px;
    display: inline-flex;
    align-items: center;
    gap: 7px;
    backdrop-filter: blur(8px);
    cursor: default;
    transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease !important;
}
.stat-pill:hover {
    border-color: rgba(196,148,20,0.65);
    box-shadow: 0 0 18px rgba(196,148,20,0.2);
    transform: translateY(-1px);
}
.stat-icon { font-size: 1rem; }
.stat-text { color: #c8a85a; font-size: 0.82rem; font-weight: 600; letter-spacing: 0.03em; }
.stat-text b { color: #f0d080; }

/* ── Language selector bar ─────────────────────────────────────── */
.lang-bar {
    background: rgba(14, 10, 22, 0.85) !important;
    border: 1px solid rgba(196,148,20,0.25) !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important;
}

/* ── Section description cards ─────────────────────────────────── */
.section-card {
    background: rgba(26, 18, 9, 0.6);
    border: 1px solid rgba(196,148,20,0.18);
    border-left: 3px solid rgba(196,148,20,0.7);
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    margin-bottom: 4px;
}
.section-card b { color: #f0d080; font-size: 0.95rem; }
.section-card span { color: #8a7040; font-size: 0.84rem; display: block; margin-top: 3px; }

/* ── Tab nav ───────────────────────────────────────────────────── */
.tab-nav {
    background: rgba(12, 9, 20, 0.9) !important;
    border: 1px solid rgba(196,148,20,0.2) !important;
    border-bottom: none !important;
    border-radius: 14px 14px 0 0 !important;
    padding: 6px 8px 0 !important;
    gap: 3px !important;
    backdrop-filter: blur(8px);
}
.tab-nav button {
    color: #6a5428 !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 10px 20px !important;
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
.tab-nav button.selected {
    color: #f0d080 !important;
    background: rgba(196,148,20,0.15) !important;
    border-bottom: 2px solid #c49414 !important;
}

/* ── Tab content panel ─────────────────────────────────────────── */
.tabitem {
    background: rgba(12, 9, 20, 0.9) !important;
    border: 1px solid rgba(196,148,20,0.2) !important;
    border-top: none !important;
    border-radius: 0 0 14px 14px !important;
    padding: 20px !important;
    backdrop-filter: blur(8px);
}

/* ── Chatbot bubble area ────────────────────────────────────────── */
.chatbot {
    background: rgba(8, 6, 14, 0.95) !important;
    border: 1px solid rgba(196,148,20,0.18) !important;
    border-radius: 12px !important;
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
    with gr.Blocks(title="Lingua Arcana", theme=THEME, css=CSS) as demo:

        # Header
        gr.Markdown(
            '<div class="app-header">'
            '<h1>📜 LINGUA ARCANA</h1>'
            '<p>Master the arcane art of language &nbsp;·&nbsp; Ten tongues await the worthy</p>'
            '</div>'
        )

        # Stats bar
        gr.Markdown(STATS_HTML)

        # Language selectors
        with gr.Row(elem_classes=["lang-bar"]):
            native_sel = gr.Dropdown(
                choices=LANGUAGES, value="English",
                label="Your Language", scale=2,
            )
            lang_sel = gr.Dropdown(
                choices=LANGUAGES, value="German",
                label="Language to Learn", scale=2,
            )
            level_sel = gr.Dropdown(
                choices=LEVELS, value="Beginner (A1-A2)",
                label="Your Level", scale=2,
            )

        with gr.Tabs():

            # ── Chat ──────────────────────────────────────────────────────
            with gr.Tab("💬 Speak"):
                gr.Markdown(tab_card(
                    "💬", "Conversational Practice",
                    "Chat with Arcana in your target language. Errors are corrected gently "
                    "with explanations delivered in your native language."
                ))
                chatbot = gr.Chatbot(
                    height=400,
                    show_label=False,
                    type="tuples",
                    bubble_full_width=False,
                )
                chat_input = gr.Textbox(
                    placeholder="Write something and press Enter to begin…",
                    label="Your message",
                    lines=2,
                )
                with gr.Row():
                    send_btn  = gr.Button("Send", variant="primary", scale=3)
                    clear_btn = gr.Button("Clear", variant="secondary", scale=1)

                send_btn.click(
                    handle_chat,
                    [chat_input, chatbot, native_sel, lang_sel, level_sel],
                    [chatbot, chat_input],
                )
                chat_input.submit(
                    handle_chat,
                    [chat_input, chatbot, native_sel, lang_sel, level_sel],
                    [chatbot, chat_input],
                )
                clear_btn.click(lambda: ([], ""), outputs=[chatbot, chat_input])

            # ── Grammar ────────────────────────────────────────────────────
            with gr.Tab("✏️ Grammar"):
                gr.Markdown(tab_card(
                    "✏️", "Grammar Analysis",
                    "Submit any sentence in your target language. Arcana identifies errors, "
                    "explains the rules broken, and gives a personalised grammar tip."
                ))
                grammar_input = gr.Textbox(
                    label="Your text",
                    placeholder="Write a sentence in your target language…",
                    lines=4,
                )
                check_btn = gr.Button("Analyse Grammar", variant="primary")
                grammar_out = gr.Textbox(
                    label="Analysis & Feedback",
                    lines=13,
                    interactive=False,
                )
                check_btn.click(
                    handle_grammar,
                    [grammar_input, native_sel, lang_sel, level_sel],
                    grammar_out,
                )

            # ── Quiz ───────────────────────────────────────────────────────
            with gr.Tab("📚 Quiz"):
                gr.Markdown(tab_card(
                    "📚", "Vocabulary Quiz",
                    "Generate a quiz in your target language — fill-in-the-blank, "
                    "multiple choice, and sentence construction. Answers are hidden until you submit."
                ))
                with gr.Row():
                    topic_dd = gr.Dropdown(
                        choices=QUIZ_TOPICS, value="General",
                        label="Topic", scale=2,
                    )
                    topic_custom = gr.Textbox(
                        label="Custom topic (optional)",
                        placeholder="e.g. weather, colours…",
                        scale=3,
                    )
                quiz_btn = gr.Button("Generate Quiz", variant="primary")
                quiz_out = gr.Textbox(
                    label="Questions",
                    lines=12,
                    interactive=False,
                )
                user_answers = gr.Textbox(
                    label="Your Answers",
                    placeholder="Write your answers here, e.g.:\n1. Kaffee\n2. b\n3. a\n4. Sie trinkt Kaffee.",
                    lines=5,
                )
                check_btn_quiz = gr.Button("Submit Answers", variant="primary")
                quiz_feedback = gr.Textbox(
                    label="Results & Feedback",
                    lines=13,
                    interactive=False,
                )
                quiz_btn.click(
                    handle_quiz,
                    [native_sel, lang_sel, level_sel, topic_dd, topic_custom],
                    [quiz_out, user_answers, quiz_feedback],
                )
                check_btn_quiz.click(
                    handle_quiz_check,
                    [quiz_out, user_answers, native_sel, lang_sel, level_sel],
                    quiz_feedback,
                )

            # ── Quick Translate ────────────────────────────────────────────
            with gr.Tab("🔄 Translate"):
                gr.Markdown(tab_card(
                    "🔄", "Quick Translation",
                    "Instantly translate any text between any two supported languages. "
                    "Includes key vocabulary breakdown and a grammar note to help you learn."
                ))
                with gr.Row():
                    qt_from = gr.Dropdown(
                        choices=LANGUAGES, value="English", label="From", scale=1,
                    )
                    qt_to = gr.Dropdown(
                        choices=LANGUAGES, value="German", label="To", scale=1,
                    )
                qt_input = gr.Textbox(
                    label="Text to translate",
                    placeholder="Type anything you want to translate…",
                    lines=3,
                )
                qt_btn = gr.Button("Translate", variant="primary")
                qt_out = gr.Textbox(
                    label="Translation + Vocabulary Notes",
                    lines=17,
                    interactive=False,
                )
                qt_btn.click(
                    handle_quick_translate,
                    [qt_input, qt_from, qt_to],
                    qt_out,
                )

            # ── Translation Practice ───────────────────────────────────────
            with gr.Tab("🌍 Practice"):
                gr.Markdown(tab_card(
                    "🌍", "Translation Practice",
                    "Test yourself — write your own translation attempt and receive a scored "
                    "evaluation with corrections, strengths, and alternative phrasings."
                ))
                with gr.Row():
                    from_lang = gr.Dropdown(
                        choices=LANGUAGES, value="English", label="From", scale=1,
                    )
                    to_lang = gr.Dropdown(
                        choices=LANGUAGES, value="German", label="To", scale=1,
                    )
                original_txt = gr.Textbox(
                    label="Original text",
                    placeholder="The sentence you want to translate…",
                    lines=3,
                )
                user_trans = gr.Textbox(
                    label="Your translation",
                    placeholder="Write your attempt here…",
                    lines=3,
                )
                trans_btn = gr.Button("Evaluate Translation", variant="primary")
                trans_out = gr.Textbox(
                    label="Score & Feedback",
                    lines=14,
                    interactive=False,
                )
                trans_btn.click(
                    handle_translation,
                    [original_txt, user_trans, from_lang, to_lang],
                    trans_out,
                )

    return demo


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=False,
        show_api=False,
    )
