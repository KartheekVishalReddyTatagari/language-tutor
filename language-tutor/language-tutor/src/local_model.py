"""
LocalTutorLLM — runs the fine-tuned LLaMA model locally.

Loads a merged (LoRA-fused) model from HuggingFace Hub in 4-bit quantization
so it fits in 4 GB VRAM (RTX 2050 / 3050 etc.).

Drop-in replacement for TutorLLM: same public methods, same signatures.
"""
import os
from typing import Generator, List

import torch


class LocalTutorLLM:
    """Inference wrapper for the fine-tuned language tutor model."""

    def __init__(self, model_id: str, hf_token: str | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        token = hf_token or os.getenv("HF_API_TOKEN")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[LocalTutorLLM] Loading '{model_id}' on {device} ...")

        quant_cfg = None
        if device == "cuda":
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_cfg,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            token=token,
        )
        self.model.eval()
        self.device = device
        print("[LocalTutorLLM] Model ready.")

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _generate(self, messages: list, max_new_tokens: int = 500, temperature: float = 0.6) -> str:
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        if self.device == "cuda":
            input_ids = input_ids.to("cuda")

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_ids = output_ids[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    def _stream(self, messages: list, max_new_tokens: int = 500, temperature: float = 0.6) -> Generator[str, None, None]:
        """Yield the full response as a single chunk (streaming not supported locally without extra libs)."""
        yield self._generate(messages, max_new_tokens=max_new_tokens, temperature=temperature)

    # ── Public API — mirrors TutorLLM ─────────────────────────────────────────

    def chat_stream(
        self,
        message: str,
        history: List[List[str]],
        native: str = "English",
        language: str = "German",
        level: str = "Beginner (A1-A2)",
    ) -> Generator[str, None, None]:
        system = (
            f"You are Arcana, an AI language tutor. "
            f"The student speaks {native} natively and is learning {language} at {level} level.\n"
            f"Practice sentences are in {language}. "
            f"All explanations MUST be in {native}. "
            "Be warm, concise, and use occasional emojis. Keep responses under 160 words."
        )
        messages = [{"role": "system", "content": system}]
        for pair in history[-6:]:
            if pair[0]:
                messages.append({"role": "user", "content": pair[0]})
            if pair[1]:
                messages.append({"role": "assistant", "content": pair[1]})
        messages.append({"role": "user", "content": message})
        yield from self._stream(messages, max_new_tokens=400, temperature=0.6)

    def check_grammar(
        self,
        text: str,
        native: str = "English",
        language: str = "German",
        level: str = "Beginner (A1-A2)",
    ) -> str:
        system = (
            f"You are Arcana, an AI language tutor and expert {language} grammar coach "
            f"for {level} learners whose native language is {native}. "
            f"Write ALL feedback in {native}."
        )
        user = (
            f"You are Arcana, an AI language tutor. "
            f"Check the following {language} sentence for grammar errors and provide structured feedback.\n\n"
            f'Check this sentence: "{text}"\n\n'
            "Reply using exactly this format:\n\n"
            "**✅ Corrected Version:**\n[corrected text or 'No errors found!']\n\n"
            "**🔍 Errors Found:**\n[numbered list of errors with rule + fix]\n\n"
            "**💡 Grammar Tip:**\n[one actionable tip]"
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self._generate(messages, max_new_tokens=500, temperature=0.3)

    def generate_quiz(
        self,
        native: str = "English",
        language: str = "German",
        level: str = "Beginner (A1-A2)",
        topic: str = "",
    ) -> str:
        topic_str = f" on the topic of {topic}" if topic.strip() else ""
        system = f"You are Arcana, a creative language quiz designer for {language} learners at {level} level."
        user = (
            f"You are Arcana, a language quiz designer. Generate a {language} vocabulary quiz{topic_str} "
            f"for a {level} learner whose native language is {native}.\n\n"
            "Include:\n"
            "**📝 Fill in the Blank** (2 questions)\n"
            "**🔤 Multiple Choice** (2 questions — do NOT mark the correct answer)\n"
            "**✍️ Use It in a Sentence** (1 vocabulary word)\n\n"
            "Show ONLY the questions. No answers."
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self._generate(messages, max_new_tokens=500, temperature=0.7)

    def check_quiz_answers(
        self,
        quiz_text: str,
        user_answers: str,
        native: str = "English",
        language: str = "German",
        level: str = "Beginner (A1-A2)",
    ) -> str:
        system = (
            f"You are Arcana, a friendly {language} quiz evaluator for {level} learners "
            f"whose native language is {native}. Write ALL feedback in {native}."
        )
        user = (
            f"Quiz:\n{quiz_text}\n\n"
            f"Student's answers:\n{user_answers}\n\n"
            "Evaluate each answer, reveal correct answers with explanations, "
            "then give a score out of 5 and a short motivating message."
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self._generate(messages, max_new_tokens=600, temperature=0.4)

    def quick_translate(self, text: str, from_lang: str = "English", to_lang: str = "German") -> str:
        system = f"You are a helpful {from_lang}–{to_lang} translator and language teacher."
        user = (
            f"Translate from {from_lang} to {to_lang}:\n\n\"{text}\"\n\n"
            "Reply in this format:\n\n"
            f"**🌍 Translation ({to_lang}):**\n[translation]\n\n"
            "**📖 Key Vocabulary:**\n[3–5 key words with meaning + example sentence]\n\n"
            "**💡 Note:**\n[one grammar or phrasing tip]"
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self._generate(messages, max_new_tokens=500, temperature=0.3)

    def check_translation(
        self,
        original: str,
        translation: str,
        from_lang: str = "English",
        to_lang: str = "German",
    ) -> str:
        system = f"You are Arcana, a skilled {from_lang}–{to_lang} translation coach."
        user = (
            f"Original ({from_lang}): \"{original}\"\n"
            f"Student's translation ({to_lang}): \"{translation}\"\n\n"
            "Reply in this format:\n\n"
            "**⭐ Score:** [X / 10]\n\n"
            "**✅ Improved Version:**\n[corrected or 'Your translation is perfect!']\n\n"
            "**👍 What you did well:**\n[specific positive feedback]\n\n"
            "**🔧 Improvements:**\n[numbered list of corrections]\n\n"
            "**💬 Alternative phrasing:**\n[1–2 natural alternatives]"
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self._generate(messages, max_new_tokens=500, temperature=0.4)


if __name__ == "__main__":
    # Quick test — run as: python -m src.local_model
    import os
    from dotenv import load_dotenv
    load_dotenv()

    model_id = os.getenv("FINETUNED_MODEL_ID")
    if not model_id:
        print("Set FINETUNED_MODEL_ID in .env first.")
    else:
        llm = LocalTutorLLM(model_id)
        result = llm.check_grammar(
            "Yesterday I go to the market and buy many vegetable.",
            native="English",
            language="English",
        )
        print(result)
