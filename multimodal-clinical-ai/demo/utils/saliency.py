import torch
import html

SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]"}

def compute_text_saliency(model, input_ids, attention_mask, target_class):
    model.zero_grad()

    embeddings = model.text_encoder.embeddings(input_ids)
    embeddings.requires_grad_(True)

    outputs = model.text_encoder(
        inputs_embeds=embeddings,
        attention_mask=attention_mask
    )

    cls_emb = outputs.last_hidden_state[:, 0, :]

    # Dummy image (image explainability handled separately)
    dummy_image = torch.zeros(1, 3, 224, 224).to(input_ids.device)
    img_feat = model.image_encoder(dummy_image)

    fused = torch.cat([img_feat, cls_emb], dim=1)
    logits = model.fusion(fused)

    score = logits[:, target_class]
    score.backward()

    grads = embeddings.grad
    saliency = grads.abs().sum(dim=-1).squeeze(0)

    # ⬅️ return BOTH saliency + attention mask
    return saliency.detach().cpu().numpy(), attention_mask.squeeze(0).cpu().numpy()


def merge_wordpieces(tokens, scores):
    merged_tokens = []
    merged_scores = []

    current_token = ""
    current_score = 0.0

    for tok, score in zip(tokens, scores):
        if tok.startswith("##"):
            current_token += tok[2:]
            current_score += score
        else:
            if current_token:
                merged_tokens.append(current_token)
                merged_scores.append(current_score)
            current_token = tok
            current_score = score

    if current_token:
        merged_tokens.append(current_token)
        merged_scores.append(current_score)

    return merged_tokens, merged_scores
def filter_tokens(tokens, scores, attention_mask):
    clean_tokens = []
    clean_scores = []

    for tok, score, mask in zip(tokens, scores, attention_mask):
        if mask == 0:
            continue
        if tok in SPECIAL_TOKENS:
            continue
        clean_tokens.append(tok)
        clean_scores.append(score)

    return clean_tokens, clean_scores


def highlight_text(tokens, scores, min_ratio=0.15):
    max_score = max(scores) + 1e-8
    highlighted = []

    for tok, score in zip(tokens, scores):
        if score < min_ratio * max_score:
            highlighted.append(html.escape(tok))
        else:
            intensity = score / max_score
            color = f"rgba(255, 0, 0, {intensity:.2f})"
            safe_tok = html.escape(tok)
            highlighted.append(
                f"<span style='background-color:{color}'>{safe_tok}</span>"
            )

    return " ".join(highlighted)
