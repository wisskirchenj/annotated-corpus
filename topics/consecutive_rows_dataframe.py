import pandas as pd

df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'B': ['x', 'y', 'y', 'y', 'x', 'y', 'y', 'y', 'y']
})
query = df['B'] == 'y'
df['group'] = (df[query].index.to_series().diff() != 1).cumsum()

# Step 2: Group by 'group' column
groups = df.groupby('group')

# Step 3: Filter groups with more than one row
filtered_groups = groups.filter(lambda x: len(x) > 1)

# Step 4: Count the number of groups
num_groups = filtered_groups['group'].nunique()

print(num_groups)