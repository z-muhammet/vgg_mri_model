import torch
import os
import numpy as np
from models.basic_cnn_model import BasicCNNModel
from models.train import build_dataloaders # Import build_dataloaders

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
RESET = '\033[0m'
BOLD = '\033[1m'

def run_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load data
    _, _, test_loader, _ = build_dataloaders(
        batch_size=12, # Use the same batch size as training or adjust as needed
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        to_rgb=True
    )

    print(f"[DEBUG] Test dataset size: {len(test_loader.dataset)}")

    # Load the best trained PyTorch model
    model = BasicCNNModel(num_classes=3, in_channels=3) # Instantiate your model architecture
    best_model_path = os.path.join("models", "best_vgg_custom.pt")

    if not os.path.exists(best_model_path):
        print(f"{RED}Error: Best model checkpoint not found at {best_model_path}. Please train the model first.{RESET}")
        return

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval() # Set model to evaluation mode

    criterion = torch.nn.CrossEntropyLoss() # Assuming CrossEntropyLoss for evaluation

    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    print(f"{GREEN}\n=== Starting Test Evaluation ==={RESET}")

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)

            outputs = model(X)
            loss = criterion(outputs, y)

            _, predicted = torch.max(outputs.data, 1)
            total_samples += y.size(0)
            total_correct += (predicted == y).sum().item()
            total_loss += loss.item() * y.size(0)

    accuracy = 100 * total_correct / total_samples
    avg_loss = total_loss / total_samples

    print(f"\n🎯 {total_samples} test samples: {total_correct} correct predictions.")
    print(f"✅ Test Accuracy: {accuracy:.2f}%")
    print(f"📊 Test Loss: {avg_loss:.3f}")

    # Initialize SWA model if SWA was used in training and model was saved as SWA
    swa_model_path = os.path.join("models", "best_vgg_custom_swa.pt")
    if os.path.exists(swa_model_path):
        print(f"{GREEN}\n=== Testing SWA Model ==={RESET}")
        swa_model = BasicCNNModel(num_classes=3, in_channels=3) # Use BasicCNNModel here as well
        swa_model.load_state_dict(torch.load(swa_model_path, map_location=device))
        swa_model.eval() # Set SWA model to evaluation mode
        test_loss_swa, test_acc_swa, _, _ = trainer._run_epoch(train=False)
        print(f"{GREEN}SWA Test Acc: {test_acc_swa:.3f}, SWA Test Loss: {test_loss_swa:.3f}{RESET}")

if __name__ == "__main__":
    run_test()