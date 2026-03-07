import torch
from PIL import Image
from torchvision import transforms
from src.model.network import MLP

# BEST MODEL: MINIBATCH con 256 HIDDEN SIZE e 16 BATCH SIZE

model_path = "models/model_minibatch_bs16_h256.pth"
image_path = "images/numero9.png"
hidden_size = 256

model = MLP(hidden_size=hidden_size)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

print(f"\npath del Modello caricato: {model_path}\n")

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

img = Image.open(image_path)
x = transform(img).unsqueeze(0)

with torch.no_grad():
    logits = model(x)
    prob = torch.softmax(logits, dim=1)
    conf, pred = torch.max(prob, dim=1)

print("*"*20)
print("Il numero è:", pred.item())
print("Confidence: ",round(conf.item() * 100, 2), "%")

print("*"*20)
print("\nProbabilità delle classi:\n")
for i in range(10):
    print(f"Classe {i}: {prob[0][i].item() * 100:.2f}%")

print("\n")