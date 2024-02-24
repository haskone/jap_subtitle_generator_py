from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-ja-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)


def translate_one(japanese_text):
    inputs = tokenizer(
        japanese_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)
