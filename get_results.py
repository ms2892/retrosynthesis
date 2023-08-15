import pandas as pd 
import numpy as np



df = pd.read_csv('results_no_tan.csv')

names = df.NAME.unique()

# print(names)

for value_fn in names:
    new_df = df.loc[df['NAME']==value_fn]
    min_sol_time = new_df.iloc[:,3].values
    print(f"Value function: {value_fn}")
    for q in [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]:
        print(f"Quantile {q}: {np.quantile(min_sol_time, q):.2f}")
    print()


for value_fn in names:
    new_df = df.loc[df['NAME']==value_fn]
    solns = new_df.iloc[:,3].values
    s=0
    for i in solns:
        if i!=float('inf'):
            s+=1

    print(f"Value function: {value_fn}")

    print(f"Number of solutions found: {s}, {s/250 * 100}")
    print()