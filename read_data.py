import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_excel(open("C:/Users/huntisan/Desktop/project/std_glove_data.xlsx", "rb"))
relevant_data = data.loc[:,["method", "size", "avg_solve_time"]]
big_sizes = relevant_data.loc[:, "size"] > 90
big_sizes = relevant_data[big_sizes]
print(big_sizes)

x = big_sizes.loc[:,"size"]
y = big_sizes.loc[:,"avg_solve_time"]
plt.scatter(x,y)
plt.show()
