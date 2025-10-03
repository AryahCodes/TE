# %%
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt


def merge_csv_files(directory_path):
    # Get all CSV files in the specified directory
    csv_files = glob.glob(os.path.join(directory_path, 'init_methods_*.csv'))
    
    # Sort the files to ensure consistent ordering
    csv_files.sort()
    
    # Initialize an empty list to store DataFrames
    dfs = []
    
    # Initialize the run counter
    run_counter = 0
    
    # Process each CSV file
    for file in csv_files:
        # Read the CSV file
        df = pd.read_csv(file)
        
        # Update the 'run' column
        df['run'] += run_counter
        
        # Append the DataFrame to the list
        dfs.append(df)
        
        # Increment the run counter
        run_counter += df['run'].max() + 1
    
    # Concatenate all DataFrames
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Save the merged DataFrame to a new CSV file
    output_file = os.path.join(directory_path, 'merged_output.csv')
    merged_df.to_csv(output_file, index=False)
    
    print(f"Merged CSV saved as: {output_file}")

    return merged_df

# Usage
directory_path = '.'
df = merge_csv_files(directory_path)


# %%
color = ["red", "blue", "green", "orange", "purple"]
linestyle = ["-", "--", ":", "-.", "-"]
methods = df['method'].unique()
n_runs = df['run'].max() + 1

tmp = []

fig, axis = plt.subplots(len(df['qubits'].unique()), len(df['layers'].unique()), figsize=(10, 10), dpi=300)

index = 0
for i, q in enumerate(df['qubits'].unique()):
    for j, l in enumerate(df['layers'].unique()):
        for m, method in enumerate(methods):
            method_df = df[(df['qubits'] == q) & (df['layers'] == l) & (df['method'] == method)]
            # for r in range(n_runs):
            #     run_df = method_df[method_df['run'] == r]
            #     axis[i, j].plot(run_df['iteration'], run_df['loss'], color=f'C{m}', linestyle='-', alpha=0.1)

            # Plot median over all runs:
            axis[i, j].plot(method_df.groupby('iteration')['loss'].mean(), color=f'C{m}', linestyle='-', label=method)
            tmp.append({"method": method, "qubits":q, "layers":l, "value":method_df.groupby('iteration')['loss'].median().values})
            
        axis[i, j].set_title(f"{q} qubits, {l} layers")
        
        # axis[i, j].set_yscale("log")
        axis[i, j].set_xlabel("Iteration")
        axis[i, j].set_ylabel("Loss")
        axis[i, j].grid()

# Add a single legend for the entire figure
handles, labels = axis[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=len(methods))

plt.tight_layout()
plt.show()


# %%
# Print the median loss for each method
tmp

# %%
# Create a DataFrame

df_1 = pd.DataFrame()

for item in tmp:
    method = item['method']
    qubits = item['qubits']
    layers = item['layers']
    column_name = f"{method}_{qubits}_{layers}"
    df_1[column_name] = item['value']

# Add an index column
df_1['idx'] = range(1, len(df_1) + 1)

# Print the table
print(df_1.to_string(index=False))
# %%
# save the table
df_1.to_csv("mean_values.csv", index=False)

# %%
