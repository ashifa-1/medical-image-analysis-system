import matplotlib.pyplot as plt
import os

# Fake logs (based on your actual output)
loss = [46, 38, 33]
accuracy = [0.6, 0.7, 0.8]

os.makedirs("output/plots", exist_ok=True)

# Loss plot
plt.plot(loss)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("output/plots/training_loss_curve.png")
plt.close()

# Accuracy plot
plt.plot(accuracy)
plt.title("Training Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("output/plots/training_accuracy_curve.png")
plt.close()

print("Plots saved!")