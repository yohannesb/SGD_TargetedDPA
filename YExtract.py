import os
import torch
import pandas as pd

# Load the .model file
model_path = "D://default_models"  # Replace with your actual .model file path
state_dict = torch.load(model_path)
print(state_dict.keys())

# # Convert the state_dict to a tabular format
# data = []
# for param_name, param_value in state_dict.items():
#     # Ensure the values are tensors and convert them to NumPy for compatibility
#     if isinstance(param_value, torch.Tensor):
#         param_data = param_value.numpy()
#         flattened_data = param_data.flatten()  # Flatten multi-dimensional arrays
#         for idx, value in enumerate(flattened_data):
#             data.append({"Parameter": param_name, "Index": idx, "Value": value})

# # Create a DataFrame
# df = pd.DataFrame(data)

# # Save the DataFrame to an Excel file
# output_file = "parameters_from_model.csv"
# df.to_csv(output_file, index=False)

# print(f"Parameters saved to {output_file}")
