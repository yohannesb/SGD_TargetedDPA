import torch

model_path = "C://Users//yohan//DataPoisoning_FL//3000_models"
state_dict = torch.load(model_path)
print(state_dict.keys())