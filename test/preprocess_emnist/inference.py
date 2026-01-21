import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from emnist_cnn import EMNIST_CNN
from preprocess_emnist import EMNIST_Preprocessor


class EMNIST_Inference:
    def __init__(self, model_path, vocab_path, device="cpu"):
        self.device = torch.device(device)

        self.model = EMNIST_CNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        with open(vocab_path, "r") as f:
            self.class_vocab = json.load(f)
        self.preprocessor = EMNIST_Preprocessor()

    def predict(self, pil_img: Image.Image):
        x = self.preprocessor(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)[0]

        idx = int(torch.argmax(probs).item())
        char = self.class_vocab[idx]
        confidence = float(probs[idx])

        return char, confidence, probs.cpu().numpy()
