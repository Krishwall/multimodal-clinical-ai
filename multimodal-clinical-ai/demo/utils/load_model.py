import torch
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

LABELS = [
    "NORMAL",
    "PNEUMONIA",
    "EFFUSION",
    "PNEUMOTHORAX",
    "CARDIOMEGALY",
    "EDEMA",
    "OTHER"
]

class MultimodalFusionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Image encoder
        self.image_encoder = models.resnet18(pretrained=False)
        self.image_encoder.fc = nn.Identity()

        # Freeze all
        for p in self.image_encoder.parameters():
            p.requires_grad = False

        # Unfreeze layer4
        for p in self.image_encoder.layer4.parameters():
            p.requires_grad = True

        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(MODEL_NAME)

        for p in self.text_encoder.parameters():
            p.requires_grad = False
        for layer in self.text_encoder.encoder.layer[-2:]:
            for p in layer.parameters():
                p.requires_grad = True

        self.fusion = nn.Sequential(
            nn.Linear(512 + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        img_feat = self.image_encoder(image)
        txt_out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        txt_feat = txt_out.last_hidden_state[:, 0, :]
        fused = torch.cat([img_feat, txt_feat], dim=1)
        return self.fusion(fused)


def load_fusion_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultimodalFusionModel(num_classes=len(LABELS))
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return model, tokenizer, image_transform, LABELS, device
