import os
import torch
#import openpyxl
#from openpyxl import Workbook
import pandas as pd

# Directory containing the model files
#model_dir = "C://Users//yohan//DataPoisoning_FL//default_models"  # Replace with your actual directory path
model_path = "D://default_models"

# Initialize a list to store the data
all_data = []

# Iterate over all files in the directory
for file_name in os.listdir(model_dir):
    # Check if the file name matches the model naming pattern
    if file_name.startswith("model_") and file_name.endswith(("_end", "_start")):
        model_path = os.path.join(model_dir, file_name)
        try:
            # Load the model's state_dict
            state_dict = torch.load(model_path)
            print(state_dict.keys())

            # # Extract parameters
            # for param_name, param_value in state_dict.items():
            #     # Ensure the values are tensors and convert them to NumPy for compatibility
            #     if isinstance(param_value, torch.Tensor):
            #         param_data = param_value.numpy()
            #         flattened_data = param_data.flatten()  # Flatten multi-dimensional arrays
            #         for idx, value in enumerate(flattened_data):
            #             all_data.append({
            #                 "Model": file_name,  # Include model file name
            #                 "Parameter": param_name,
            #                 "Index": idx,
            #                 "Value": value
            #             })

            # print(f"Processed {file_name}")

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

# # Check if all_data is populated
# if not all_data:
#     print("No data was collected. Check if models contain state_dict parameters.")


# Create a DataFrame with all the data
# df = pd.DataFrame(all_data)

# if not df.empty:
#     # Save the DataFrame to a CSV file
#     output_file = "parameters_from_all_models.csv"
#     df.to_csv(output_file, index=False)  # Don't include the DataFrame index in the CSV
#     print(f"All parameters saved to {output_file}")
# else:
#     print("The DataFrame is empty. No data to save.")



# Save the DataFrame to an Excel file
# output_file = "parameters_from_all_models.csv"
# df.to_csv(output_file, index=True)

# print(f"All parameters saved to {output_file}")
