import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import requests
from PIL import Image
from io import BytesIO

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "unsloth/gemma-3-4b-it-unsloth-bnb-4bit"

app = FastAPI()

# Load model at startup
model = Gemma3ForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)

class RequestData(BaseModel):
    image_url: str
    prompt_text: str

@app.get("/")
def home():
    return {"status": "Gemma REST API is running!"}

@app.post("/generate")
def generate_text(data: RequestData):
    response = requests.get(data.image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": data.prompt_text}]}
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(DEVICE, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)

    return {"generated_text": decoded}
