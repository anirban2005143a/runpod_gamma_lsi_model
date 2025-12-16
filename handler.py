# import os
# import re
# import json
# from typing import List, Set, Dict, Any

# import torch
# import runpod
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from peft import PeftModel

# # ---------------------------
# # Config (RunPod env)
# # ---------------------------
# MODEL_ID     = os.getenv("MODEL_ID", "google/gemma-3-12b-it")
# PEFT_PATH    = os.getenv("PEFT_PATH", "")
# USE_4BIT     = os.getenv("USE_4BIT", "1") not in {"0", "false", "False"}
# CTX_TOKENS   = int(os.getenv("CONTEXT_TOKENS", "32000"))
# TEMP         = float(os.getenv("TEMPERATURE", "0.1"))
# MAX_NEW      = int(os.getenv("MAX_NEW_TOKENS", "128"))
# NUM_BEAMS    = int(os.getenv("NUM_BEAMS", "4"))
# CHUNK_BUDGET = int(os.getenv("CHUNK_BUDGET", "12000"))

# # ---------------------------
# # System prompt
# # ---------------------------
# SYSTEM_INSTRUCTIONS = """
# You are a legal expert on Indian law. Given the text below, identify all applicable legal sections and their full act names.
# Output only one list in square brackets like:
# ["Section X of Act Name"; "Section Y of Act Name"]
# """.strip()

# # ---------------------------
# # Helpers
# # ---------------------------
# def whitespace_handler(text: str) -> str:
#     text = re.sub(r"\s+", " ", re.sub(r"\n+", " ", text.strip()))
#     return text.replace("\xad", "")

# def build_prompt_llama3(tokenizer, src: str) -> str:
#     messages = [
#         {"role": "system", "content": SYSTEM_INSTRUCTIONS},
#         {"role": "user", "content": whitespace_handler(src)},
#     ]
#     return tokenizer.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     ).strip()

# def build_prompt_gemma3(tokenizer, src: str) -> str:
#     messages = [{
#         "role": "user",
#         "content": f"{SYSTEM_INSTRUCTIONS}\n\n{whitespace_handler(src)}"
#     }]
#     return tokenizer.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     ).strip()

# def get_builder(mid: str):
#     mid = (mid or "").lower()
#     if "llama" in mid:
#         return build_prompt_llama3
#     if "gemma" in mid:
#         return build_prompt_gemma3
#     return build_prompt_llama3

# _section_item_re = re.compile(r'"([^"]+?)"')

# def parse_section_act_list(text: str) -> List[str]:
#     m = re.search(r"\[(.*?)\]", text, flags=re.DOTALL)
#     segment = m.group(1) if m else text
#     items = _section_item_re.findall(segment)

#     seen, cleaned = set(), []
#     for it in items:
#         c = whitespace_handler(it)
#         if c and c not in seen:
#             seen.add(c)
#             cleaned.append(c)
#     return cleaned

# # --------------------------------------------------
# # Model initialization (runs ONCE per container)
# # --------------------------------------------------
# print("🔹 Loading model...")

# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "left"
# tokenizer.model_max_length = CTX_TOKENS

# quant_cfg = None
# if USE_4BIT:
#     quant_cfg = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16,
#     )

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID,
#     trust_remote_code=True,
#     device_map="auto",
#     torch_dtype=torch.bfloat16,
#     quantization_config=quant_cfg,
# )

# if PEFT_PATH:
#     model = PeftModel.from_pretrained(model, PEFT_PATH, device_map="auto")

# model.eval()
# model.config.use_cache = True

# prompt_builder = get_builder(MODEL_ID)

# # --------------------------------------------------
# # Core inference
# # --------------------------------------------------
# def chunk_text(text: str, max_tokens: int) -> List[str]:
#     paras = [p.strip() for p in text.split("\n\n") if p.strip()]
#     chunks, cur, cur_tokens = [], "", 0

#     for p in paras:
#         t = tokenizer(p, return_tensors="pt")["input_ids"].shape[1]
#         if cur_tokens + t > max_tokens and cur:
#             chunks.append(cur.strip())
#             cur, cur_tokens = p, t
#         else:
#             cur = f"{cur}\n\n{p}" if cur else p
#             cur_tokens += t

#     if cur:
#         chunks.append(cur.strip())

#     return chunks or [""]

# def generate(prompt: str) -> str:
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     with torch.inference_mode():
#         out = model.generate(
#             **inputs,
#             max_new_tokens=MAX_NEW,
#             temperature=TEMP,
#             num_beams=NUM_BEAMS,
#             do_sample=False,
#             pad_token_id=tokenizer.eos_token_id,
#             no_repeat_ngram_size=8,
#         )
#     return tokenizer.decode(
#         out[0][inputs["input_ids"].shape[-1]:],
#         skip_special_tokens=True
#     ).strip()

# def classify(text: str) -> str:
#     all_items = set()
#     for chunk in chunk_text(text, CHUNK_BUDGET):
#         prompt = prompt_builder(tokenizer, chunk)
#         gen = generate(prompt)
#         all_items.update(parse_section_act_list(gen))

#     merged = "; ".join(f"\"{it}\"" for it in sorted(all_items))
#     return f"[{merged}]"

# # --------------------------------------------------
# # RunPod handler
# # --------------------------------------------------
# def handler(event: Dict[str, Any]) -> Dict[str, Any]:
#     try:
#         inputs = event["input"].get("inputs")
#         if not inputs:
#             return {"error": "Missing 'inputs' field"}

#         if isinstance(inputs, list):
#             results = []
#             for i, text in enumerate(inputs, 1):
#                 results.append({
#                     "query_no": i,
#                     "model_output": classify(text)
#                 })
#             return {"count": len(results), "results": results}

#         return {"output": classify(inputs)}

#     except Exception as e:
#         return {"error": f"{type(e).__name__}: {e}"}

# # --------------------------------------------------
# # RunPod bootstrap
# # --------------------------------------------------
# runpod.serverless.start({"handler": handler})




import os
import re
import json
from typing import List, Dict, Any

import torch
import runpod

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HFValidationError

# --------------------------------------------------
# CONFIG (RunPod + HF Cache)
# --------------------------------------------------
MODEL_ID     = os.getenv("MODEL_ID", "AnirbanDas2005/Gemma-LSI-ft")
PEFT_PATH    = os.getenv("PEFT_PATH", "")
USE_4BIT     = os.getenv("USE_4BIT", "1") not in {"0", "false", "False"}
CTX_TOKENS   = int(os.getenv("CONTEXT_TOKENS", "32000"))
TEMP         = float(os.getenv("TEMPERATURE", "0.1"))
MAX_NEW      = int(os.getenv("MAX_NEW_TOKENS", "128"))
NUM_BEAMS    = int(os.getenv("NUM_BEAMS", "4"))
CHUNK_BUDGET = int(os.getenv("CHUNK_BUDGET", "12000"))

# Persistent HF cache on RunPod volume
CACHE_DIR = "/runpod-volume/huggingface-cache"
os.environ["HF_HOME"] = CACHE_DIR

# --------------------------------------------------
# SYSTEM PROMPT
# --------------------------------------------------
SYSTEM_INSTRUCTIONS = """
You are a legal expert on Indian law. Given the text below, identify all applicable legal sections and their full act names.
Output only one list in square brackets like:
["Section X of Act Name"; "Section Y of Act Name"]
""".strip()

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def whitespace_handler(text: str) -> str:
    text = re.sub(r"\s+", " ", re.sub(r"\n+", " ", text.strip()))
    return text.replace("\xad", "")

def build_prompt_gemma3(tokenizer, src: str) -> str:
    messages = [{
        "role": "user",
        "content": f"{SYSTEM_INSTRUCTIONS}\n\n{whitespace_handler(src)}"
    }]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ).strip()

def get_builder(mid: str):
    return build_prompt_gemma3

_section_item_re = re.compile(r'"([^"]+?)"')

def parse_section_act_list(text: str) -> List[str]:
    m = re.search(r"\[(.*?)\]", text, flags=re.DOTALL)
    segment = m.group(1) if m else text
    items = _section_item_re.findall(segment)

    seen, cleaned = set(), []
    for it in items:
        c = whitespace_handler(it)
        if c and c not in seen:
            seen.add(c)
            cleaned.append(c)
    return cleaned

# --------------------------------------------------
# MODEL LOAD WITH CACHE LOGGING (RUNS ONCE)
# --------------------------------------------------
print("🔍 Checking Hugging Face cache...")

try:
    snapshot_path = snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=CACHE_DIR,
        local_files_only=True
    )
    print(f"✅ Using cached model at: {snapshot_path}")

except (FileNotFoundError, HFValidationError):
    print("⬇️ Model not found in cache. Downloading now...")

    snapshot_path = snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=CACHE_DIR,
        local_files_only=False
    )
    print(f"✅ Model downloaded and cached at: {snapshot_path}")

print("🚀 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    snapshot_path,
    trust_remote_code=True,
    cache_dir=CACHE_DIR
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"
tokenizer.model_max_length = CTX_TOKENS

quant_cfg = None
if USE_4BIT:
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

print("🚀 Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    snapshot_path,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=quant_cfg,
    cache_dir=CACHE_DIR
)

if PEFT_PATH:
    print("🎯 Loading PEFT adapter...")
    model = PeftModel.from_pretrained(
        model,
        PEFT_PATH,
        device_map="auto"
    )

model.eval()
model.config.use_cache = True

prompt_builder = get_builder(MODEL_ID)

print("🎯 Model ready")

# --------------------------------------------------
# CORE INFERENCE
# --------------------------------------------------
def chunk_text(text: str, max_tokens: int) -> List[str]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur, cur_tokens = [], "", 0

    for p in paras:
        t = tokenizer(p, return_tensors="pt")["input_ids"].shape[1]
        if cur_tokens + t > max_tokens and cur:
            chunks.append(cur.strip())
            cur, cur_tokens = p, t
        else:
            cur = f"{cur}\n\n{p}" if cur else p
            cur_tokens += t

    if cur:
        chunks.append(cur.strip())

    return chunks or [""]

def generate(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW,
            temperature=TEMP,
            num_beams=NUM_BEAMS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=8,
        )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()

def classify(text: str) -> str:
    all_items = set()
    for chunk in chunk_text(text, CHUNK_BUDGET):
        prompt = prompt_builder(tokenizer, chunk)
        gen = generate(prompt)
        all_items.update(parse_section_act_list(gen))

    merged = "; ".join(f"\"{it}\"" for it in sorted(all_items))
    return f"[{merged}]"

# --------------------------------------------------
# RUNPOD HANDLER
# --------------------------------------------------
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    try:
        inputs = event["input"].get("inputs")
        if not inputs:
            return {"error": "Missing 'inputs' field"}

        if isinstance(inputs, list):
            results = []
            for i, text in enumerate(inputs, 1):
                results.append({
                    "query_no": i,
                    "model_output": classify(text)
                })
            return {"count": len(results), "results": results}

        return {"output": classify(inputs)}

    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

# --------------------------------------------------
# RUNPOD BOOTSTRAP
# --------------------------------------------------
runpod.serverless.start({"handler": handler})
