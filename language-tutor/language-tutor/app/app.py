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


# ── Chat handler (streaming generator — Chatbot supports it) ──────────────────

def handle_chat(message: str, history: list, language: str, level: str):
    """history is list of [user, bot] tuples (type='tuples')."""
    message = message.strip()
    if not message:
        yield history, ""
        return
    new_history = history + [[message, None]]
    yield new_history, ""
    reply = ""
    try:
        for chunk in get_llm().chat_stream(
            message, history=history, language=language, level=level
        ):
            reply += chunk
            new_history[-1][1] = reply
            yield new_history, ""
    except Exception as e:
        new_history[-1][1] = f"Error: {e}"
        yield new_history, ""


# ── Other handlers (plain functions — no streaming needed) ────────────────────

def handle_grammar(text: str, language: str, level: str) -> str:
    if not text.strip():
        return "Please enter some text first."
    try:
        return get_llm().check_grammar(text, language=language, level=level)
    except Exception as e:
        return f"Error: {e}"


def handle_quiz(language: str, level: str, topic_dd: str, topic_custom: str) -> str:
    topic = topic_custom.strip() or topic_dd
    try:
        return get_llm().generate_quiz(language=language, level=level, topic=topic)
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


# ── UI ────────────────────────────────────────────────────────────────────────

def create_app() -> gr.Blocks:
    with gr.Blocks(title="LinguaBot - AI Language Tutor") as demo:

        gr.Markdown(
            "# 🦜 LinguaBot — AI Language Tutor\n"
            "Practice grammar, vocabulary, translation and conversation in 10 languages."
        )

        with gr.Row():
            lang_sel = gr.Dropdown(
                choices=LANGUAGES, value="English",
                label="Target Language", scale=2,
            )
            level_sel = gr.Dropdown(
                choices=LEVELS, value="Beginner (A1-A2)",
                label="Your Level", scale=2,
            )

        with gr.Tabs():

            # ── Chat ──────────────────────────────────────────────────────
            with gr.Tab("💬 Chat Tutor"):
                chatbot = gr.Chatbot(
                    height=420,
                    show_label=False,
                    type="tuples",
                    bubble_full_width=False,
                )
                chat_input = gr.Textbox(
                    placeholder="Type a message and press Enter…",
                    label="Your message", lines=2,
                )
                with gr.Row():
                    send_btn  = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear Chat", variant="secondary")

                send_btn.click(
                    handle_chat,
                    [chat_input, chatbot, lang_sel, level_sel],
                    [chatbot, chat_input],
                )
                chat_input.submit(
                    handle_chat,
                    [chat_input, chatbot, lang_sel, level_sel],
                    [chatbot, chat_input],
                )
                clear_btn.click(lambda: ([], ""), outputs=[chatbot, chat_input])

            # ── Grammar ────────────────────────────────────────────────────
            with gr.Tab("✏️ Grammar Check"):
                grammar_input = gr.Textbox(
                    label="Your text",
                    placeholder="e.g. She don't likes coffee.",
                    lines=4,
                )
                check_btn   = gr.Button("Check Grammar", variant="primary")
                grammar_out = gr.Textbox(
                    label="Feedback", lines=12, interactive=False,
                )
                check_btn.click(
                    handle_grammar,
                    [grammar_input, lang_sel, level_sel],
                    grammar_out,
                )

            # ── Quiz ───────────────────────────────────────────────────────
            with gr.Tab("📚 Vocab Quiz"):
                topic_dd = gr.Dropdown(
                    choices=QUIZ_TOPICS, value="General", label="Topic",
                )
                topic_custom = gr.Textbox(
                    label="Custom topic (optional)",
                    placeholder="e.g. weather, family…",
                )
                quiz_btn = gr.Button("Generate Quiz", variant="primary")
                quiz_out = gr.Textbox(
                    label="Quiz", lines=20, interactive=False,
                )
                quiz_btn.click(
                    handle_quiz,
                    [lang_sel, level_sel, topic_dd, topic_custom],
                    quiz_out,
                )

            # ── Translation ────────────────────────────────────────────────
            with gr.Tab("🌍 Translation"):
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
                    label="Feedback", lines=14, interactive=False,
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
