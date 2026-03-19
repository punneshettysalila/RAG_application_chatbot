# llm_handler.py
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
import torch
from dotenv import load_dotenv, dotenv_values
import os
from importlib.util import find_spec
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def _supports_4bit_quantization():
    return (
        torch.cuda.is_available()
        and os.name != "nt"
        and find_spec("bitsandbytes") is not None
    )


def _get_hf_token():
    """Read HF token from env or local .env file."""
    env_token = (
        os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("HUGGINGFACE_API_TOKEN")
        or os.getenv("HF_TOKEN")
    )
    if env_token:
        return env_token.strip()

    env_values = dotenv_values(ENV_PATH)
    file_token = (
        env_values.get("HUGGINGFACEHUB_API_TOKEN")
        or env_values.get("HUGGINGFACE_API_TOKEN")
        or env_values.get("HF_TOKEN")
    )
    return file_token.strip() if isinstance(file_token, str) else None


def load_llm():
    """Load LLaMA model via HuggingFace transformers pipeline."""
    token = _get_hf_token()
    if not token:
        raise RuntimeError(
            "Missing Hugging Face token. Add HUGGINGFACEHUB_API_TOKEN to .env."
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "token": token,
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
    if _supports_4bit_quantization():
        model_kwargs["load_in_4bit"] = True

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        **model_kwargs
    )

    pipeline_kwargs = {
        # Keep generation bounded for responsive Streamlit inference.
        "max_new_tokens": 100,
        "temperature": 0.3,
        "do_sample": False,
        "top_p": 0.9,
        "repetition_penalty": 1.0,
        "return_full_text": False,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if not torch.cuda.is_available():
        pipeline_kwargs["device"] = -1

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        **pipeline_kwargs
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


def build_prompt(context, chat_history, question):
    """Build a structured prompt with context and history."""
    history_text = ""
    for turn in chat_history[-4:]:  # Use last 4 turns for context window
        history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"

    prompt = f"""You are a helpful assistant. Answer the user's question based on the provided context from PDF documents. Be concise and accurate.

Context from PDFs:
{context}

Conversation so far:
{history_text}
User: {question}
Assistant:"""
    return prompt


def get_answer(llm, prompt):
    """Get answer from LLM with timeout protection."""
    timeout_seconds = int(os.getenv("LLM_TIMEOUT_SECONDS", "90"))

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            raw = executor.submit(llm.invoke, prompt).result(timeout=timeout_seconds)
    except FuturesTimeoutError as exc:
        raise TimeoutError(
            f"LLM response timed out after {timeout_seconds}s. "
            "Try a shorter question or use a smaller model."
        ) from exc

    # Handle possible return shapes defensively.
    if isinstance(raw, str):
        text = raw
    elif isinstance(raw, dict):
        text = str(raw.get("text") or raw.get("generated_text") or raw)
    elif isinstance(raw, list) and raw:
        first = raw[0]
        if isinstance(first, dict):
            text = str(first.get("generated_text") or first.get("text") or first)
        else:
            text = str(first)
    else:
        text = str(raw)

    answer = text.strip()
    if not answer:
        return "I could not generate a response. Please try rephrasing your question."
    return answer
