import numpy as np
import pandas as pd
import time

# add sleep of one minute to test if the job is still running
time.sleep(60)

print("hello world, I'm alex lol")
print(f"numpy version: {np.__version__}")
print(f"pandas version: {pd.__version__}")
print(f"dot product of [1, 2] and [3, 4]: {np.dot([1, 2], [3, 4])}")