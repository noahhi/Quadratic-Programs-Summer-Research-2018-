import pandas as pd
import matplotlib.pyplot as plt

# data=pd.read_excel(open("C:/Users/huntisan/Desktop/summer2018/std_glove_data.xlsx", "rb"))
# relevant_data = data.loc[:,["method", "size", "avg_solve_time"]]
# big_sizes = relevant_data.loc[:, "size"] > 90
# big_sizes = relevant_data[big_sizes]
# print(big_sizes)
#
# x = big_sizes.loc[:,"size"]
# y = big_sizes.loc[:,"avg_solve_time"]
# plt.scatter(x,y)
# plt.show()

df = pd.read_pickle("dataframes/glove_bounds.pkl")
print(df)
#retrieve desired rows using boolean/mask indexing
# org_bounds = df[df["glover_bounds"]=="original"]
# tight_bounds = df[df["glover_bounds"]=="tight"]
#
# org_bounds_times = org_bounds["avg_total_time"]
# sorted_org = org_bounds_times.sort_values()
# tight_bounds_times = tight_bounds["avg_solve_time"]
# sorted_tight = tight_bounds_times.sort_values()
#
# option0 = df[df["options"]==0]
# option1 = df[df["options"]==1]
# option2 = df[df["options"]==2]
# option3 = df[df["options"]==3]
# x = ['option0', 'option1', 'option2', 'option3']
#
# plt.plot(sorted_org.values)
# plt.plot(sorted_tight.values)
# plt.show()
