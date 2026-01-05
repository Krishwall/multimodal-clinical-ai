import torch
import torch.nn.functional as F
from PIL import Image

def run_inference(
    model,
    tokenizer,
    image_transform,
    image,
    text,
    labels,
    device
):
    # Image
    image = image_transform(image).unsqueeze(0).to(device)

    # Text
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(image, input_ids, attention_mask)
        probs = F.softmax(logits, dim=1)

    top2_prob, top2_idx = torch.topk(probs, k=2, dim=1)

    primary = labels[top2_idx[0, 0].item()]
    secondary = labels[top2_idx[0, 1].item()]

    return {
        "primary": primary,
        "secondary": secondary,
        "primary_prob": float(top2_prob[0, 0]),
        "secondary_prob": float(top2_prob[0, 1]),
    }
