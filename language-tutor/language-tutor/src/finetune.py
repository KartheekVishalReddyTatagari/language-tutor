import os
from dataclasses import dataclass
from typing import Optional, List

from groq import Groq


@dataclass
class LLMConfig:
    groq_api_key: Optional[str] = None
    groq_model: str = "llama-3.1-8b-instant"
    local_model_dir: Optional[str] = None
    hf_token: Optional[str] = None
    finetuned_model_id: Optional[str] = None  # HuggingFace model ID for the fine-tuned model


class TutorLLM:
    """Groq-backed language tutor with multi-turn chat, grammar check, quiz, and translation."""

    def __init__(self, cfg: LLMConfig):
        api_key = cfg.groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing. Set it in .env or as an environment variable.")
        self.cfg = cfg
        self.client = Groq(api_key=api_key)

    def _call(self, messages: list, temperature: float = 0.5, max_tokens: int = 600) -> str:
        completion = self.client.chat.completions.create(
            model=self.cfg.groq_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content.strip()

    def _stream(self, messages: list, temperature: float = 0.5, max_tokens: int = 600):
        """Yield text chunks from a streaming Groq completion."""
        stream = self.client.chat.completions.create(
            model=self.cfg.groq_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def chat_stream(
        self,
        message: str,
        history: List[List[str]],
        native: str = "English",
        language: str = "German",
        level: str = "Beginner (A1-A2)",
    ):
        """Stream chat tokens one chunk at a time."""
        system = (
            f"You are Arcana, an AI language tutor for the app Lingua Arcana. "
            f"The student speaks {native} natively and is learning {language} at {level} level.\n"
            f"RULES — follow these strictly:\n"
            f"1. Practice sentences and exercises are written in {language}.\n"
            f"2. ALL explanations, corrections, encouragement, and instructions MUST be written in {native}. "
            f"NEVER use English unless {native} is English.\n"
            f"3. If the student writes in the wrong language, gently redirect them in {native}.\n"
            "4. Be warm, concise, and use occasional emojis. "
            "Keep responses under 160 words unless a detailed grammar explanation is requested."
        )
        messages = [{"role": "system", "content": system}]
        for pair in history[-8:]:
            if pair[0]:
                messages.append({"role": "user", "content": pair[0]})
            if pair[1]:
                messages.append({"role": "assistant", "content": pair[1]})
        messages.append({"role": "user", "content": message})
        yield from self._stream(messages, temperature=0.6, max_tokens=400)

    def check_grammar(
        self,
        text: str,
        native: str = "English",
        language: str = "German",
        level: str = "Beginner (A1-A2)",
    ) -> str:
        """Return structured grammar feedback with correction, explanation, and a tip."""
        system = (
            f"You are an expert {language} grammar coach for {level} learners whose native language is {native}. "
            f"You MUST write ALL feedback, explanations, and tips in {native}. "
            f"NEVER use English unless {native} is English."
        )
        user = (
            f"Check the following {language} text for grammar errors:\n\n\"{text}\"\n\n"
            "Reply using **exactly** this format (no extra sections):\n\n"
            "**✅ Corrected Version:**\n[corrected text, or 'No errors found!' if correct]\n\n"
            "**🔍 Errors Found:**\n[numbered list — each error, what rule was broken, and the fix; "
            "if none, write 'None — great job!']\n\n"
            "**💡 Grammar Tip:**\n[one actionable tip related to the errors, or a reinforcement if correct]"
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self._call(messages, temperature=0.3, max_tokens=600)

    def generate_quiz(
        self,
        native: str = "English",
        language: str = "German",
        level: str = "Beginner (A1-A2)",
        topic: str = "",
    ) -> str:
        """Generate a quiz — questions only, no answers."""
        topic_str = f" on the topic of **{topic}**" if topic.strip() else ""
        system = (
            f"You are a creative language quiz designer for {language} learners at {level} level. "
            "Create fun, Duolingo-style exercises. NEVER reveal answers in this step."
        )
        user = (
            f"Create a short {language} vocabulary quiz{topic_str} for a {level} learner "
            f"whose native language is {native}.\n\n"
            f"All exercises must test {language} vocabulary and grammar. "
            f"Write the questions in {language}. "
            f"You may add a short {native} instruction line at the top of each section so the student knows what to do.\n\n"
            "Include these **three** exercise types:\n\n"
            f"**📝 Fill in the Blank** (2 questions — use a {language} word in a {language} sentence):\n"
            f"[{language} sentence with ___ blank — do NOT include the answer]\n\n"
            f"**🔤 Multiple Choice** (2 questions — do NOT mark the correct answer):\n"
            f"[{language} question about word meaning or usage]\na) ... b) ... c) ... d) ...\n\n"
            f"**✍️ Use It in a Sentence** (1 vocabulary word):\n"
            f"[give one {language} vocabulary word and ask the learner to write an original {language} sentence using it]\n\n"
            "Show ONLY the questions. No answers, no hints, no correct-answer markers."
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self._call(messages, temperature=0.7, max_tokens=600)

    def check_quiz_answers(
        self,
        quiz_text: str,
        user_answers: str,
        native: str = "English",
        language: str = "German",
        level: str = "Beginner (A1-A2)",
    ) -> str:
        """Evaluate student answers against the quiz questions."""
        system = (
            f"You are a friendly {language} quiz evaluator for {level} learners whose native language is {native}. "
            f"You MUST write ALL feedback and explanations in {native}. "
            f"NEVER use English unless {native} is English."
        )
        user = (
            f"Quiz questions:\n{quiz_text}\n\n"
            f"Student's answers:\n{user_answers}\n\n"
            "For each question:\n"
            "- Fill in the blank / Multiple choice: reveal the correct answer, say right or wrong with a brief explanation.\n"
            "- 'Use It in a Sentence': check if the sentence is grammatically correct and uses the word properly; "
            "suggest an improvement if needed.\n\n"
            "End with a score out of 5 and a short motivating message."
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self._call(messages, temperature=0.4, max_tokens=700)

    def quick_translate(
        self,
        text: str,
        from_lang: str = "English",
        to_lang: str = "German",
    ) -> str:
        """Translate text and explain key vocabulary so the learner picks something up."""
        system = (
            f"You are a helpful {from_lang}–{to_lang} translator and language teacher."
        )
        user = (
            f"Translate the following from {from_lang} to {to_lang}:\n\n\"{text}\"\n\n"
            "Reply using **exactly** this format:\n\n"
            f"**🌍 Translation ({to_lang}):**\n[the translation]\n\n"
            "**📖 Key Vocabulary:**\n"
            "[a numbered list of 3–5 important words or phrases from the translation — "
            f"show the {to_lang} word, its {from_lang} meaning, and one example sentence in {to_lang}]\n\n"
            "**💡 Note:**\n[one short tip about grammar or phrasing that is useful to remember]"
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self._call(messages, temperature=0.3, max_tokens=600)

    def check_translation(
        self,
        original: str,
        translation: str,
        from_lang: str = "English",
        to_lang: str = "German",
    ) -> str:
        """Evaluate a learner's translation attempt."""
        system = (
            f"You are a skilled {from_lang}–{to_lang} translation coach. "
            "Give constructive, encouraging feedback."
        )
        user = (
            f"Original ({from_lang}): \"{original}\"\n"
            f"Student's translation ({to_lang}): \"{translation}\"\n\n"
            "Reply using **exactly** this format:\n\n"
            "**⭐ Score:** [X / 10]\n\n"
            "**✅ Improved Version:**\n[corrected or 'Your translation is perfect!']\n\n"
            "**👍 What you did well:**\n[specific positive feedback]\n\n"
            "**🔧 Improvements:**\n[numbered list of specific corrections with brief explanations]\n\n"
            "**💬 Alternative phrasing:**\n[1–2 natural alternatives]"
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self._call(messages, temperature=0.4, max_tokens=600)

    def generate(self, user_message: str, mode: str = "grammar", language: str = "English") -> str:
        """Legacy single-turn method kept for backward compatibility."""
        level_hint = "Intermediate (B1-B2)" if mode == "dialogue" else "Beginner (A1-A2)"
        return self.chat_stream(user_message, history=[], language=language, level=level_hint)
