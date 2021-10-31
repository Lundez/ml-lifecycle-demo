# %%
import joblib
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from datasets import load_from_disk
from joblib import dump
import matplotlib.pyplot as plt
import json

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

clf = LogisticRegression(random_state=42)
clf.fit([x[0] for x in train['image']], train['label'])

y_pred = clf.predict([x[0] for x in test['image']])

print(f"Accuracy: {accuracy_score(test['label'], y_pred)}")

disp = ConfusionMatrixDisplay.from_predictions(test['label'], y_pred, normalize='true', cmap=plt.cm.Blues)

with open('scores.json', 'w') as f:
    json.dump({'avg_accuracy': accuracy_score(test['label'], y_pred), }, f)

plt.savefig('confusion_matrix.png')
joblib.dump(clf, 'model.joblib')

# VGG16, LinearRegression, Vanilla CNN