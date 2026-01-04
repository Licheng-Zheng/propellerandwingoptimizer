import os
import shutil

save_location = r"C:\Users\liche\OneDrive\Desktop\Video Items\Propeller Project\4. January 2026\Diagnostics January 3rd"

expected_number = len(os.listdir(r"usingasb\diagnostics\Modal_Best"))

wanted_examples = 5 

assert expected_number >= wanted_examples, f"Not enough examples found. Expected at least {wanted_examples}, but found {expected_number}."

folders_to_check = [r"usingasb\diagnostics\Modal_Best", r"usingasb\diagnostics\Modal_Display", r"usingasb\diagnostics\Modal_Sigma"]

# Verify that all folders have the expected number of items
for folder in folders_to_check: 
    length = len(os.listdir(folder))
    assert length == expected_number, f"Folder {folder} has {length} items, but expected {expected_number}."

# Create save location if it doesn't exist 
if not os.path.isdir(save_location):
    os.mkdir(save_location)

for folder in folders_to_check: 
    for index in range(0, expected_number, expected_number // wanted_examples):
        items = os.listdir(folder)
        item = items[index]
        full_path = os.path.join(folder, item)

        shutil.copy(full_path, save_location)
    #         destination_path = os.path.join(save_location, item)



        