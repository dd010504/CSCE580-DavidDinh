import json
import matplotlib.pyplot as plt
import seaborn as sns

# Load training history
with open("./results/metrics/training_history.json", "r") as f:
    history = json.load(f)

# Extract data
loss = [x['loss'] for x in history if 'loss' in x]
epochs = range(1, len(loss) + 1)

# Plot Loss
plt.figure(figsize=(10, 5))
sns.lineplot(x=epochs, y=loss, marker='o')
plt.title('Training Loss per Epoch')
plt.xlabel('Steps/Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig("results/figures/training_loss.png")
print("Loss plot saved to results/figures/training_loss.png")