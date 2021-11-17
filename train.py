# %%
import joblib
import yaml
import json

import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

from datasets import load_from_disk
# %%
class TODO(Exception):
     pass
# %%
def get_params() -> str:
    with open("params.yaml") as f:
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

clf = LogisticRegression(penalty = params['penalty'])
clf.fit(train['image'], train['label'])

y_pred = clf.predict(test['image'])
accuracy = accuracy_score(y_pred, test['label'])

disp = ConfusionMatrixDisplay.from_predictions(test['label'], y_pred, normalize='true', cmap=plt.cm.Blues)

print(f"Accuracy: {accuracy}")

with open('scores.json', 'w') as f:
    json.dump({'avg_accuracy': accuracy}, f)

pickle.dump(clf, open('model.pkl', 'wb'))
plt.savefig('confusion_matrix.png')

# %%

"""
# FastAI + ResNet
def get_data(source: Dataset):
    reshaped = torch.stack((source['image'],) * 3, axis=1).type(torch.FloatTensor)
    reshaped /= 255.

    result = [PILImage(to_image(x)) for x in reshaped]
    labels = source['label']

    res = []
    for i in range(0, len(result)):
        data = {'x': result[i], 'y': str(labels[i].item())}
        res.append(data)

    return res

datablock = DataBlock(
    blocks=(ImageBlock(cls=PILImage), CategoryBlock),
    get_items=get_data,
    splitter=EndSplitter(len(dataset['test']) / len(all_data)),
    item_tfms=Resize(64, ResizeMethod.Pad, pad_mode='zeros'),
    get_y=(lambda item: item['y']),
    get_x=(lambda item: item['x'])
)
dls = datablock.dataloaders(all_data)

dls.show_batch()

learn = cnn_learner(dls, arch = models.resnet18, metrics=accuracy)
learn.fine_tune(1, cbs=[DvcLiveCallback()])

accuracy = learn.final_record[1]
interpret.plot_confusion_matrix()

plt.savefig('confusion_matrix.png')
learn.save(os.path.abspath('./model'))

"""