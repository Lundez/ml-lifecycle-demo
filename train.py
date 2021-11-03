# %%
import joblib
import yaml
import json

import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

from datasets import load_from_disk
# %%
def get_params() -> str:
    with open("params.yml") as f:
        return yaml.safe_load(f)

def reshape_data(example):
    example['image'] = np.array(example['image']).reshape(28*28)
    return example

# %%
dataset = load_from_disk("data/")
params = get_params()


# %%
train = dataset['train'].map(reshape_data)
test = dataset['test'].map(reshape_data)

clf = LogisticRegression(random_state=42)
clf.fit(train['image'], train['label'])

y_pred = clf.predict(test['image'])

accuracy = accuracy_score(test['label'], y_pred)
disp = ConfusionMatrixDisplay.from_predictions(test['label'], y_pred, normalize='true', cmap=plt.cm.Blues)

print(f"Accuracy: {accuracy}")

with open('scores.json', 'w') as f:
    json.dump({'avg_accuracy': accuracy, }, f)

plt.savefig('confusion_matrix.png')
joblib.dump(clf, 'model.joblib')