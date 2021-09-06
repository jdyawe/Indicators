import numpy as np
import pandas as pd

a = np.random.random(size=(4,))
a = pd.Series(a)
print(a)
print(a.diff(1))