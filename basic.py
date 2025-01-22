import pandas as pd
import numpy as np

# Machine Defect Dataset
np.random.seed(0)
machine_ids = ['M1', 'M2', 'M3', 'M4', 'M5']
data = {
    'Machine_ID': np.random.choice(machine_ids, 100),
    'Temperature': np.random.uniform(70, 120, 100),
    'Run_Time': np.random.randint(50, 200, 100),
    'Downtime_Flag': np.random.choice([0, 1], 100)
}

df = pd.DataFrame(data)
df.to_csv('defect_dataset.csv', index=False)
