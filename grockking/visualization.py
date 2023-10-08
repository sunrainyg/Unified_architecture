import wandb
import matplotlib.pyplot as plt

api = wandb.Api()

# Fetch runs from a specific project
runs = api.runs("YOUR_USERNAME/YOUR_PROJECT_NAME")

for run in runs:
    # Here, we assume you have logged a metric called "accuracy". Change as needed.
    history = run.history(keys=["accuracy"], samples=1000)
    plt.plot(history["step"], history["accuracy"], label=run.name)

plt.legend()
plt.xlabel("Steps")
plt.ylabel("Accuracy")
plt.title("Accuracy from multiple experiments")
plt.show()
