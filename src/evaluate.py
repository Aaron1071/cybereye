import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns  # Seaborn = viz lib (heatmaps—teach: Builds on matplotlib for pretty plots)
import matplotlib.pyplot as plt
from src.dataset import get_loaders
from src.model import CNNBaseline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, val_loader, _, classes = get_loaders()
num_classes = len(classes)
from src.model import HybridCNNViT
model = HybridCNNViT(num_classes).to(device)
model.load_state_dict(torch.load("best_model.pth")) # Load = restore weights
model.eval()

preds, trues = [], []
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images.to(device))
        pred = outputs.argmax(dim=1).cpu().numpy()
        preds.extend(pred)
        trues.extend(labels.numpy())  # Numpy = to array for sklearn

print(classification_report(trues, preds, target_names=classes))  # Report = table prec/recall/F1 per class (connect: Like forensic report metrics for accuracy in evidence classification)

cm = confusion_matrix(trues, preds)  # Matrix = 26x26 array (true rows, pred cols—diagonal correct)
plt.figure(figsize=(12,10))  # Figsize = size
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)  # Heatmap = colored grid (annot=numbers, fmt=decimal, cmap=color scheme; xticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

if __name__ == "__main__":
    print("Eval complete.")