from datasets import load_dataset

print("Downloading MNIST")
dataset = load_dataset("mnist")
print("Saving MNIST to 'data/{test,train}'/..")
dataset.save_to_disk("data/")
print("MNIST saved to disk")