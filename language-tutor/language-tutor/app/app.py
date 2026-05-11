import os
from typing import List

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
    "Family", "Sports", "Health", "Technology", "Nature", "Emotions",
]

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;900&display=swap');

*, *::before, *::after { box-sizing: border-box; }
body {
    font-family: 'Nunito', 'Segoe UI', sans-serif !important;
    background: #0d1117 !important;
    color: #e6edf3 !important;
}
.gradio-container {
    max-width: 980px !important;
    margin: 0 auto !important;
    padding: 16px !important;
    background: transparent !important;
}

/* ── Hero ── */
#hero {
    background: linear-gradient(135deg, #1a2a1a 0%, #0d1f0d 40%, #101c2a 100%);
    border: 1px solid #2ea043;
    border-radius: 24px;
    padding: 36px 40px 28px;
    margin-bottom: 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
#hero::before {
    content: '';
    position: absolute; top: -60px; left: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(46,160,67,.18) 0%, transparent 70%);
    border-radius: 50%; pointer-events: none;
}
#hero h1 {
    font-size: 2.8em; font-weight: 900; color: #58cc02;
    letter-spacing: -1px; text-shadow: 0 0 40px rgba(88,204,2,.4);
    margin: 0 0 8px;
}
#hero p { color: #8b949e; font-size: 1.05em; line-height: 1.6; margin: 0; }
.hero-badges { display: flex; justify-content: center; gap: 10px; margin-top: 16px; flex-wrap: wrap; }
.hb {
    background: rgba(88,204,2,.12); border: 1px solid rgba(88,204,2,.3);
    color: #58cc02; border-radius: 100px; padding: 5px 14px;
    font-size: .82em; font-weight: 700;
}

/* ── Settings row ── */
#settings-row {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 16px; padding: 14px 20px; margin-bottom: 16px;
}

/* ── Tabs ── */
.tabs { background: transparent !important; }
.tab-nav {
    background: #161b22 !important; border-radius: 100px !important;
    padding: 4px !important; gap: 2px !important;
    border: 1px solid #30363d !important; margin-bottom: 14px !important;
}
.tab-nav button {
    border-radius: 100px !important; font-weight: 700 !important;
    padding: 8px 18px !important; border: none !important;
    background: transparent !important; color: #8b949e !important;
    font-size: .88em !important; transition: all .2s !important;
}
.tab-nav button.selected {
    background: #58cc02 !important; color: #0d1117 !important;
    box-shadow: 0 2px 8px rgba(88,204,2,.4) !important;
}
.tabitem { background: transparent !important; border: none !important; padding: 0 !important; }

/* ── Input fields ── */
textarea, input[type="text"] {
    background: #161b22 !important; border: 1.5px solid #30363d !important;
    border-radius: 12px !important; color: #e6edf3 !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #58cc02 !important; box-shadow: 0 0 0 3px rgba(88,204,2,.12) !important;
    outline: none !important;
}
select {
    background: #161b22 !important; color: #e6edf3 !important;
    border: 1.5px solid #30363d !important; border-radius: 10px !important;
}
label { color: #8b949e !important; font-size: .9em !important; font-weight: 600 !important; }

/* ── Chat ── */
#chatbox { background: #161b22 !important; border: 1px solid #30363d !important; border-radius: 16px !important; }

/* ── Buttons ── */
button.primary {
    background: linear-gradient(180deg, #58cc02, #46a302) !important;
    color: #fff !important; border: none !important;
    border-radius: 100px !important; font-weight: 800 !important;
    padding: 11px 28px !important;
    box-shadow: 0 4px 0 #2d6901, 0 6px 16px rgba(88,204,2,.25) !important;
    transition: transform .12s, box-shadow .12s !important;
}
button.primary:hover { transform: translateY(-2px) !important; }
button.primary:active { transform: translateY(2px) !important; box-shadow: 0 2px 0 #2d6901 !important; }
button.secondary {
    background: #21262d !important; color: #c9d1d9 !important;
    border: 1px solid #30363d !important; border-radius: 100px !important;
    font-weight: 700 !important; padding: 9px 22px !important;
}
button.secondary:hover { background: #2d333b !important; }

/* ── Output markdown ── */
.out-md {
    background: #161b22 !important; border: 1px solid #2ea043 !important;
    border-radius: 16px !important; padding: 20px 22px !important;
    color: #e6edf3 !important; line-height: 1.75 !important; min-height: 80px;
}
.out-md strong { color: #58cc02 !important; }
.out-md code { background: #0d1117 !important; border-radius: 6px !important; padding: 2px 6px !important; color: #1CB0F6 !important; }

/* ── Section hints ── */
.hint { color: #58a0cc; font-size: .9em; border-left: 3px solid #1CB0F6; padding-left: 10px; margin-bottom: 12px; }

/* ── Examples ── */
.examples { background: #161b22 !important; border-radius: 12px !important; border: 1px solid #30363d !important; }

/* ── Blocks ── */
.block { background: #161b22 !important; border-color: #30363d !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #58cc02; }
"""

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


# ── Handlers ─────────────────────────────────────────────────────────────────

def handle_chat(message: str, history: List, language: str, level: str):
    message = message.strip()
    if not message:
        return history, ""
    try:
        reply = get_llm().chat(message, history=history, language=language, level=level)
    except Exception as e:
        reply = f"**Error:** {e}"
    return history + [[message, reply]], ""


def handle_grammar(text: str, language: str, level: str) -> str:
    if not text.strip():
        return "_Enter some text above and click **Check Grammar**._"
    try:
        return get_llm().check_grammar(text, language=language, level=level)
    except Exception as e:
        return f"**Error:** {e}"


def handle_quiz(language: str, level: str, topic_dd: str, topic_custom: str) -> str:
    topic = topic_custom.strip() if topic_custom.strip() else topic_dd
    try:
        return get_llm().generate_quiz(language=language, level=level, topic=topic)
    except Exception as e:
        return f"**Error:** {e}"


def handle_translation(original: str, translation: str, from_lang: str, to_lang: str) -> str:
    if not original.strip() or not translation.strip():
        return "_Fill in both fields above._"
    try:
        return get_llm().check_translation(original, translation, from_lang=from_lang, to_lang=to_lang)
    except Exception as e:
        return f"**Error:** {e}"


# ── UI ────────────────────────────────────────────────────────────────────────

def create_app() -> gr.Blocks:
    with gr.Blocks(title="LinguaBot - AI Language Tutor") as demo:

        # Hero
        gr.HTML("""
        <div id="hero">
            <h1>🦜 LinguaBot</h1>
            <p>An immersive AI language tutor powered by LLaMA 3.1 &nbsp;·&nbsp; Speak, practice, master.</p>
            <div class="hero-badges">
                <span class="hb">10 Languages</span>
                <span class="hb">Grammar AI</span>
                <span class="hb">Adaptive Quizzes</span>
                <span class="hb">Translation Coach</span>
                <span class="hb">Live Feedback</span>
            </div>
        </div>
        """)

        # Settings
        with gr.Row(elem_id="settings-row"):
            lang_sel = gr.Dropdown(
                choices=LANGUAGES, value="English",
                label="🌍 Target Language", scale=2, interactive=True,
            )
            level_sel = gr.Dropdown(
                choices=LEVELS, value="Beginner (A1-A2)",
                label="📈 Your Level", scale=2, interactive=True,
            )
            gr.HTML(
                "<div style='display:flex;align-items:center;color:#484f58;"
                "font-size:.83em;padding:4px 8px;line-height:1.5;'>"
                "Set your language &amp; level once —<br>every tab uses these settings.</div>",
                scale=2,
            )

        with gr.Tabs():

            # ── Chat ─────────────────────────────────────────────────────
            with gr.Tab("💬 Chat Tutor"):
                gr.HTML("<p class='hint'>Have a real conversation. Ask grammar questions, "
                        "request vocabulary help, or just practice freely.</p>")
                chatbot = gr.Chatbot(
                    label="",
                    height=430,
                    show_label=False,
                    avatar_images=(
                        None,
                        "https://em-content.zobj.net/source/google/387/parrot_1f99c.png",
                    ),
                    elem_id="chatbox",
                    layout="bubbles",
                )
                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder="Ask anything… e.g. 'How do I form questions in German?'",
                        show_label=False, scale=5, lines=1,
                    )
                    send_btn = gr.Button("Send ▶", variant="primary", scale=1)
                with gr.Row():
                    clear_btn = gr.Button("🗑 Clear Chat", variant="secondary")
                    gr.Examples(
                        examples=[
                            "How do I use the past tense in German?",
                            "Correct my sentence: She don't likes coffee.",
                            "Give me 5 French phrases for ordering food.",
                            "What's the difference between 'since' and 'for'?",
                            "Teach me basic Japanese greetings with pronunciation.",
                        ],
                        inputs=chat_input,
                        label="Try one of these starters:",
                    )
                send_btn.click(handle_chat, [chat_input, chatbot, lang_sel, level_sel], [chatbot, chat_input])
                chat_input.submit(handle_chat, [chat_input, chatbot, lang_sel, level_sel], [chatbot, chat_input])
                clear_btn.click(lambda: ([], ""), outputs=[chatbot, chat_input])

            # ── Grammar Check ─────────────────────────────────────────────
            with gr.Tab("✏️ Grammar Check"):
                gr.HTML("<p class='hint'>Paste any text — get a corrected version, "
                        "a numbered error list, and a targeted grammar tip.</p>")
                grammar_input = gr.Textbox(
                    label="Your Text",
                    placeholder="e.g.  Yesterday I have go to school and meted my friends there.",
                    lines=5,
                )
                check_btn = gr.Button("Check Grammar ✓", variant="primary")
                grammar_out = gr.Markdown(
                    value="_Feedback will appear here._", elem_classes="out-md"
                )
                gr.Examples(
                    examples=[
                        "She don't likes coffee but drink tea every morning.",
                        "I am very boring at this class, the teacher explain bad.",
                        "Yesterday we have visited the museum and it was very interested.",
                        "He has less friends than me but more happier than before.",
                        "The childrens was playing in the park when the rain started.",
                    ],
                    inputs=grammar_input, label="Try these error-filled sentences:",
                )
                check_btn.click(handle_grammar, [grammar_input, lang_sel, level_sel], grammar_out)

            # ── Quiz ──────────────────────────────────────────────────────
            with gr.Tab("📚 Vocab Quiz"):
                gr.HTML("<p class='hint'>Get a personalised quiz — fill-in-the-blank, "
                        "multiple choice, and translation — all tailored to your level.</p>")
                with gr.Row():
                    topic_dd = gr.Dropdown(
                        choices=QUIZ_TOPICS, value="General",
                        label="Topic", scale=2, interactive=True,
                    )
                    topic_custom = gr.Textbox(
                        label="Or type a custom topic",
                        placeholder="weather, emotions, city life…",
                        scale=3,
                    )
                    quiz_btn = gr.Button("Generate Quiz 🎯", variant="primary", scale=1)
                quiz_out = gr.Markdown(
                    value="_Click **Generate Quiz** to get your personalised exercise!_",
                    elem_classes="out-md",
                )
                quiz_btn.click(handle_quiz, [lang_sel, level_sel, topic_dd, topic_custom], quiz_out)

            # ── Translation ───────────────────────────────────────────────
            with gr.Tab("🌍 Translate"):
                gr.HTML("<p class='hint'>Write your translation — get a score out of 10, "
                        "specific corrections, and natural-sounding alternatives.</p>")
                with gr.Row():
                    from_lang = gr.Dropdown(
                        choices=LANGUAGES, value="English",
                        label="From", scale=1, interactive=True,
                    )
                    to_lang = gr.Dropdown(
                        choices=LANGUAGES, value="German",
                        label="To", scale=1, interactive=True,
                    )
                original_txt = gr.Textbox(
                    label="Original Text",
                    placeholder="Enter the sentence you want to translate…", lines=3,
                )
                user_trans = gr.Textbox(
                    label="Your Translation",
                    placeholder="Write your translation attempt here…", lines=3,
                )
                trans_btn = gr.Button("Evaluate Translation ✓", variant="primary")
                trans_out = gr.Markdown(
                    value="_Your feedback will appear here._", elem_classes="out-md"
                )
                gr.Examples(
                    examples=[
                        ["I would like a coffee, please.", "Ich hätte gerne einen Kaffee, bitte.", "English", "German"],
                        ["The train arrives at eight o'clock.", "Le train arrive à huit heures.", "English", "French"],
                        ["Can you help me find the station?", "Puedes ayudarme a encontrar la estacion?", "English", "Spanish"],
                    ],
                    inputs=[original_txt, user_trans, from_lang, to_lang],
                    label="Pre-filled example translations:",
                )
                trans_btn.click(
                    handle_translation,
                    [original_txt, user_trans, from_lang, to_lang],
                    trans_out,
                )

            # ── About ─────────────────────────────────────────────────────
            with gr.Tab("ℹ️ About"):
                gr.HTML("""
                <div style="padding:8px 0;line-height:1.7;">
                    <h2 style="color:#58cc02;font-size:1.6em;margin-bottom:10px;">🦜 LinguaBot</h2>
                    <p style="color:#8b949e;">
                        An immersive AI-powered language learning assistant built at
                        <strong style="color:#c9d1d9;">SRH University</strong>.
                        Combines a large-language-model backend (LLaMA 3.1 via Groq) with a
                        scikit-learn ML pipeline for grammar error classification.
                    </p>

                    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin:20px 0;">
                        <div style="background:#1c2a1c;border:1px solid #2ea043;border-radius:14px;padding:16px;text-align:center;">
                            <div style="font-size:2em;">💬</div>
                            <h3 style="color:#58cc02;margin:6px 0 4px;">Chat Tutor</h3>
                            <p style="color:#8b949e;font-size:.84em;">Multi-turn AI conversation with grammar corrections &amp; cultural tips</p>
                        </div>
                        <div style="background:#1a1c2a;border:1px solid #1CB0F6;border-radius:14px;padding:16px;text-align:center;">
                            <div style="font-size:2em;">✏️</div>
                            <h3 style="color:#1CB0F6;margin:6px 0 4px;">Grammar Check</h3>
                            <p style="color:#8b949e;font-size:.84em;">Structured error detection with rule explanations &amp; learning tips</p>
                        </div>
                        <div style="background:#2a1a1a;border:1px solid #f78166;border-radius:14px;padding:16px;text-align:center;">
                            <div style="font-size:2em;">📚</div>
                            <h3 style="color:#f78166;margin:6px 0 4px;">Vocab Quiz</h3>
                            <p style="color:#8b949e;font-size:.84em;">Adaptive fill-in-the-blank, multiple-choice &amp; translation exercises</p>
                        </div>
                        <div style="background:#1a2a2a;border:1px solid #76e3ea;border-radius:14px;padding:16px;text-align:center;">
                            <div style="font-size:2em;">🌍</div>
                            <h3 style="color:#76e3ea;margin:6px 0 4px;">Translation</h3>
                            <p style="color:#8b949e;font-size:.84em;">Scored evaluation with corrections &amp; natural alternatives</p>
                        </div>
                    </div>

                    <h3 style="color:#1CB0F6;margin:24px 0 10px;">🧠 ML Pipeline (Scikit-learn)</h3>
                    <table style="width:100%;border-collapse:collapse;font-size:.9em;">
                        <tr style="border-bottom:1px solid #30363d;">
                            <th style="text-align:left;padding:8px;color:#8b949e;">Model</th>
                            <th style="text-align:left;padding:8px;color:#8b949e;">Features</th>
                            <th style="text-align:left;padding:8px;color:#8b949e;">Outputs</th>
                        </tr>
                        <tr style="border-bottom:1px solid #30363d;">
                            <td style="padding:8px;color:#58cc02;font-weight:700;">Logistic Regression</td>
                            <td style="padding:8px;color:#c9d1d9;">TF-IDF uni+bigrams</td>
                            <td style="padding:8px;color:#c9d1d9;">Accuracy, F1, ROC-AUC, CV, GridSearch</td>
                        </tr>
                        <tr>
                            <td style="padding:8px;color:#1CB0F6;font-weight:700;">Linear SVM</td>
                            <td style="padding:8px;color:#c9d1d9;">TF-IDF uni+bigrams</td>
                            <td style="padding:8px;color:#c9d1d9;">Accuracy, F1, ROC-AUC, CV, GridSearch</td>
                        </tr>
                    </table>

                    <h3 style="color:#1CB0F6;margin:24px 0 10px;">🚀 Run the pipeline</h3>
                    <pre style="background:#0d1117;border:1px solid #30363d;border-radius:12px;padding:16px;color:#58cc02;font-size:.85em;overflow-x:auto;">python -m src.preprocess    # generate 10 000-row dataset
python -m src.classifier    # train LogReg + SVM
python -m src.evaluate      # metrics, ROC, confusion matrix, learning curves</pre>

                    <h3 style="color:#1CB0F6;margin:24px 0 10px;">🐳 Docker</h3>
                    <pre style="background:#0d1117;border:1px solid #30363d;border-radius:12px;padding:16px;color:#58cc02;font-size:.85em;">docker compose up --build   # http://localhost:7860</pre>

                    <p style="color:#484f58;font-size:.82em;margin-top:28px;text-align:center;">
                        SRH University &nbsp;·&nbsp; Artificial Intelligence &nbsp;·&nbsp; 2025
                    </p>
                </div>
                """)

    return demo


# ── Theme (Gradio 6: passed to launch()) ─────────────────────────────────────
def _theme() -> gr.themes.Base:
    return gr.themes.Base(
        primary_hue=gr.themes.colors.green,
        secondary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.gray,
        radius_size=gr.themes.sizes.radius_lg,
        spacing_size=gr.themes.sizes.spacing_md,
    ).set(
        body_background_fill="#0d1117",
        body_text_color="#e6edf3",
        block_background_fill="#161b22",
        block_border_color="#30363d",
        block_label_text_color="#8b949e",
        input_background_fill="#0d1117",
        input_border_color="#30363d",
        input_placeholder_color="#484f58",
        button_primary_background_fill="linear-gradient(180deg, #58cc02, #46a302)",
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="#21262d",
        button_secondary_text_color="#c9d1d9",
    )


if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=False,
        theme=_theme(),
        css=CSS,
    )
