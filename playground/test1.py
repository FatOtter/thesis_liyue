import numpy as np
import pandas as pd
import participant

zero1 = np.genfromtxt("./records/Confined_parameters2021_09_28_09MR_1.csv", delimiter=",")
zero2 = np.genfromtxt("./records/Confined_parameters2021_09_28_10MR_2.csv", delimiter=",")
zero3 = np.genfromtxt("./records/Confined_parameters2021_09_28_11MR_3.csv", delimiter=",")
zero4 = np.genfromtxt("./records/Confined_parameters2021_09_28_12MR_4.csv", delimiter=",")

print(zero1.shape)
combined = np.hstack([zero1[1:, 1::3], zero2[1:, 1::3], zero3[1:, 1::3], zero4[1:, 1::3]])
print(combined.shape)
# avg = np.average(combined, axis=1)
# avg = avg.reshape(16970, 1)
# print(avg.shape)
# combined = np.hstack([combined, avg])
df = pd.DataFrame(combined, index=None)
df.to_csv("./records/combined_random_init_mnist.csv")