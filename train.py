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
class TODO(Exception):
     pass
# %%
def get_params() -> str:
    with open("params.yml") as f:
        raise TODO("Use yaml to load file")

def reshape_data(example):
    raise TODO("example['image'] should be 28*28")

# %%
dataset = load_from_disk("data/")
params = get_params()


# %%
raise TODO("define train & test, then map the data into correct fmt")

raise TODO("Create a LogisticRegression and fit on data")
require(type(clf) == LogisticRegression)

raise TODO("Predict with model and get accuracy + confusion matrix")

print(f"Accuracy: {accuracy}")

with open('scores.json', 'w') as f:
    raise TODO("Dump score as avg_accuracy in json")

raise TODO("Dump clf & confusionmatrix")

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