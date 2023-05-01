import pandas as pd
import os
<<<<<<< HEAD
import pandas as pd
=======
>>>>>>> a526836f88b561deb0d98adf05c3c4cb5b7349aa

# Set directory where CSV files are located
csv_dir = './_data/ai_factory/social/TRAIN'

# Create empty list to hold dataframes
df_list = []

# Loop through all CSV files in directory
for file in os.listdir(csv_dir):
    if file.endswith('.csv'):
        # Read CSV file into dataframe
        df = pd.read_csv(os.path.join(csv_dir, file))
        
        # Do any necessary data cleaning or manipulation here
        # ...
        
        # Append dataframe to list
        df_list.append(df)

# Concatenate all dataframes in list into one dataframe
combined_df = pd.concat(df_list)

# Save combined dataframe to CSV file
<<<<<<< HEAD
combined_df.to_csv('./_data/ai_factory/social/train_all.csv', index=False)
=======
combined_df.to_csv('./_data/ai_factory/social/train_all.csv', index=False)
>>>>>>> a526836f88b561deb0d98adf05c3c4cb5b7349aa
