https://huggingface.co/spaces/Krishwall/multimodal-clinical-ai


<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/def84582-bfcb-44d7-9cd8-75b3ce1e3846" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/9ec05624-bad0-4017-8712-4ac25c914b5c" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/cc16cec9-a9cc-4516-9a04-3c9d1c893828" />
---
title: Multimodal Clinical AI
emoji: 🏥
colorFrom: purple
colorTo: pink
sdk: streamlit
sdk_version: 1.29.0
app_file: app.py
pinned: false
license: mit
---

# 🏥 Multimodal Clinical AI - Chest X-ray Analysis

An advanced AI system that combines chest X-ray images with radiology text reports to provide accurate diagnostic predictions with explainability.

## 🎯 Features

- **Multimodal Fusion**: Combines ResNet18 (image) + Bio_ClinicalBERT (text) for comprehensive analysis
- **7-Class Diagnosis**: Detects NORMAL, PNEUMONIA, EFFUSION, PNEUMOTHORAX, CARDIOMEGALY, EDEMA, and OTHER conditions
- **Explainability**: 
  - Grad-CAM heatmaps showing image regions influencing diagnosis
  - Text saliency highlighting important clinical terms
- **Beautiful UI**: Modern, intuitive interface with gradient designs and interactive elements
- **Real-time Analysis**: Fast inference with confidence scores for top-3 diagnoses

## 🚀 Quick Start

1. Upload a chest X-ray image (PNG, JPG, or JPEG)
2. Enter radiology findings or clinical notes
3. Click "Analyze Case" to get diagnostic predictions
4. Explore explainability visualizations to understand model decisions

## 🏗️ Model Architecture

- **Image Encoder**: ResNet18 (pretrained, layer4 fine-tuned)
- **Text Encoder**: Bio_ClinicalBERT (pretrained, last 2 layers fine-tuned)
- **Fusion Layer**: Concatenated features → 512-dim hidden → 7-class output
- **Explainability**: Grad-CAM for images, gradient-based saliency for text

## 📋 Requirements

See `requirements.txt` for full dependencies. Key packages:
- PyTorch >= 2.0.1
- Transformers (for Bio_ClinicalBERT)
- Streamlit >= 1.29.0
- OpenCV, Pillow, NumPy

## ⚠️ Disclaimer

**This tool is for educational and research purposes only. Not intended for clinical diagnosis or treatment decisions. Always consult qualified healthcare professionals.**

## 📚 Citation

If you use this model in your research, please cite:

```bibtex
@software{multimodal_clinical_ai,
  title={Multimodal Clinical AI: Chest X-ray Analysis with Radiology Text},
  author={Your Name},
  year={2024},
  url={https://huggingface.co/spaces/your-username/multimodal-clinical-ai}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

