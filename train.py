# %%
import joblib
import yaml
import json

from fastai.vision.all import *
from dvclive.fastai import DvcLiveCallback

from datasets import load_from_disk, concatenate_datasets
from datasets.arrow_dataset import Dataset

import matplotlib.pyplot as plt
# %%
def get_params() -> str:
    with open("params.yml") as f:
        return yaml.safe_load(f)

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
# %%
dataset = load_from_disk("data/")
dataset.set_format('pytorch')
all_data = concatenate_datasets([dataset['train'], dataset['test']])

params = get_params()


# %%
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

accuracy = learn.final_record[1]
interpret.plot_confusion_matrix()

print(f"Accuracy: {accuracy}")

with open('scores.json', 'w') as f:
    json.dump({'avg_accuracy': accuracy, }, f)

plt.savefig('confusion_matrix.png')
learn.save(os.path.abspath('./model'))