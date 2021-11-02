# %%
import joblib
import yaml
import json

from fastai import *
from fastai.vision.all import *
from dvclive.fastai import DvcLiveCallback

<<<<<<< HEAD
from fastai.vision.all import *
from dvclive.fastai import DvcLiveCallback

from datasets import load_from_disk
=======
from datasets import load_from_disk, concatenate_datasets
from datasets.arrow_dataset import Dataset
>>>>>>> f23917f (Added DVCLive & fastai ResNet CNN)

import matplotlib.pyplot as plt

# %%
def get_params() -> str:
    with open("params.yml") as f:
        return yaml.safe_load(f)

<<<<<<< HEAD
def reshape_data(example):
    example['image'] = np.array(example['image']).reshape(28*28)
    return example
=======
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

# %%
>>>>>>> f23917f (Added DVCLive & fastai ResNet CNN)

# %%
dataset = load_from_disk("data/")
dataset.set_format('pytorch')
all_data = concatenate_datasets([dataset['train'], dataset['test']])

params = get_params()


# %%
<<<<<<< HEAD
train = dataset['train'].map(reshape_data)
test = dataset['test'].map(reshape_data)

clf = LogisticRegression(random_state=42)
clf.fit(train['image'], train['label'])

y_pred = clf.predict(test['image'])
=======
datablock = DataBlock(
    blocks=(ImageBlock(cls=PILImage), CategoryBlock),
    get_items=get_data,
    splitter=EndSplitter(len(dataset['test']) / len(all_data)),
    get_y=(lambda item: item['y']),
    get_x=(lambda item: item['x'])
)
dls = datablock.dataloaders(all_data)

dls.show_batch()

# %%

learn = cnn_learner(dls, arch = models.resnet18, metrics=accuracy)
learn.fine_tune(1, cbs=[DvcLiveCallback()])

interpret = ClassificationInterpretation.from_learner(learn)
>>>>>>> f23917f (Added DVCLive & fastai ResNet CNN)

accuracy = learn.final_record[1]
interpret.plot_confusion_matrix()

print(f"Accuracy: {accuracy}")

with open('scores.json', 'w') as f:
    json.dump({'avg_accuracy': accuracy, }, f)

plt.savefig('confusion_matrix.png')
<<<<<<< HEAD
joblib.dump(clf, 'model.joblib')
=======

learn.save(os.path.abspath('./model'))
>>>>>>> f23917f (Added DVCLive & fastai ResNet CNN)
