import os
import pandas as pd
import shutil


# Define the paths
data_dir = r'C:\Users\pasca\Doggo-Classifier\data\DOGGO'  # Adjust your path as necessary
train_dir = os.path.join(data_dir, 'test')
labels_file = os.path.join(data_dir, 'labels.csv')

counter = 0

# Step 1: Read the labels CSV file
labels_df = pd.read_csv(labels_file)

# Check the first few rows of the dataframe
print(labels_df.head())

# Step 2: Create a directory for each breed
for breed in labels_df['breed'].unique():
    breed_dir = os.path.join(train_dir, breed)
    if not os.path.exists(breed_dir):
        os.makedirs(breed_dir)

# Step 3: Move images into the corresponding breed directories
for index, row in labels_df.iterrows():
    image_id = row['id']  # Replace 'id' with the actual column name if it's different
    breed = row['breed']  # Replace 'breed' with the actual column name if it's different
    
    # Construct the source and destination paths
    src_path = os.path.join(data_dir, 'test', f'{image_id}.jpg')  # Adjust the extension as necessary
    dest_path = os.path.join(train_dir, breed, f'{image_id}.jpg')
    
    # Move the file if it exists
    if os.path.exists(src_path):
        shutil.move(src_path, dest_path)
        print(f'Moved: {src_path} -> {dest_path}')
    else:
        print(f'File not found: {src_path}')
    counter += 1  

print("Directory restructuring complete.")
print(counter)