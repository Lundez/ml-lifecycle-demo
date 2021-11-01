# %%
import joblib
import yaml
import json
from joblib import dump

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

from datasets import load_from_disk

import matplotlib.pyplot as plt
import pandas as pd


# %%
def get_params() -> str:
    with open("params.yml") as f:
        return yaml.safe_load(f)

# %%
dataset = load_from_disk("data/")
params = get_params()


# %%
train = dataset['train'].to_pandas()
test = dataset['test'].to_pandas()

train['flat_img'] = train['image']
test['flat_img'] = test['image']

clf = LogisticRegression(random_state=42)
clf.fit(train['flat_img'].to_numpy().reshape(60000, 28*28), train['label'])

y_pred = clf.predict(test['flat_img'].to_numpy().reshape(10000, 28*28))

accuracy = accuracy_score(test['label'], y_pred)
disp = ConfusionMatrixDisplay.from_predictions(test['label'], y_pred, normalize='true', cmap=plt.cm.Blues)

print(f"Accuracy: {accuracy}")

with open('scores.json', 'w') as f:
    json.dump({'avg_accuracy': accuracy, }, f)

plt.savefig('confusion_matrix.png')
joblib.dump(clf, 'model.joblib')

# VGG16, LinearRegression, Vanilla CNN