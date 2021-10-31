import yaml
from sklearn.linear_model import LogisticRegression
from datasets import load_from_disk

# %%
def get_params() -> str:
    with open("params.yml") as f:
        return yaml.safe_load(f)

# %%
dataset = load_from_disk("data/")
params = get_params()

print(params)
print(dataset)

# %%



# VGG16, LinearRegression, Vanilla CNN

# CI/CD
# data versioning
# versioning
# metric tracking