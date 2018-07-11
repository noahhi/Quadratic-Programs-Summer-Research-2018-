import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os


def write_report(df, save_loc):

    #time_stamp = time.strftime("%Y_%m_%d-%H_%M_%S")
    #excel_filename = time_stamp+'-report.xlsx'
    #writer = pd.ExcelWriter("reports/"+excel_filename, engine='xlsxwriter')

    #save dataframe to excel file
    writer = pd.ExcelWriter(save_loc+"report.xlsx", engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    worksheet.conditional_format('P2:P1000',{'type':'3_color_scale', 'min_color':'green', 'max_color':'red'})
    writer.save()

    #open the excel file for viewing
    #os.chdir("reports")
    #os.system(excel_filename)

def performance_profile(df, save_loc):
    """
    convert run_trials report into form for performnace profile generation
    """
    variable = 'solver'
    formulations = ['cplex', 'xpress', 'gurobi']
    data = {}
    for form in formulations:
        form_rows = df[df[variable]==form]
        form_data = form_rows["instance_solve_time"].tolist()
        data[form] = form_data
    df = pd.DataFrame(data)

    #retrieve # of problem instances (ni), and # of formulations (nf)
    [ni, nf] = df.shape
    T = df.values

    #best solve time for each problem instance
    minperf = np.zeros(ni)
    for i in range(ni):
        minperf[i] = np.nanmin(T[i])

    #compute ratios
    r = np.zeros((ni, nf))
    for p in range(ni):
        r[p,:] = T[p,:]/minperf[p]
    max_ratio = np.nanmax(r)

    #replace nan vals with 2*maxratio
    nan_indices = np.isnan(r)
    r[nan_indices] = 2*max_ratio
    r.sort(axis=0)

    yf = np.zeros(ni)
    for i in range(ni):
        yf[i] = i/ni

    linestyles = ["solid", "dashed", "dotted", "dashdot"]
    for f in range(nf):
        xf = r[:,f]
        plt.plot(xf,yf, drawstyle="steps", linestyle=linestyles[f])
    plt.title("Performance Profile For {}".format(variable))
    plt.xlabel('Performance Profile on [0,10]')
    plt.ylabel('P(r <= t: 1 <= s <= n)')
    plt.legend([form for form in formulations])

    plt.savefig(save_loc+"performance_profile")
    plt.show()

#read in dataframe
df = pd.read_pickle("dataframes/uqp_std_cons.pkl")

save_as = 'test'
mypath = "data/{}/".format(save_as)
if save_as=="test" or not os.path.isdir(mypath):
    if not save_as=="test":
        os.makedirs(mypath)
    df.to_pickle(mypath+'/dataframef.pkl')
    write_report(df, save_loc=mypath)
    performance_profile(df, save_loc=mypath)
else:
    raise Exception("Save path folder {} already exists. ".format(mypath))


#pre converted DF
# #get number of rows and number of problems
# shape = df.shape
# n = shape[0]
# p = int(n/2)
#
# #retrieve desired rows using boolean/mask indexing
# all4cons = df[df["options"]==0]
# all4con_times = all4cons["instance_solve_time"]
# signdep = df[df["options"]==1]
# signdep_times = signdep["instance_solve_time"]
#
# #compute min times for each problem
# mintimes = np.zeros(p)
# for i, times in enumerate(zip(all4con_times, signdep_times)):
#     mintimes[i] = min(times[0], times[1])
#
# #compute performance ratios
# r4 = np.zeros(p)
# for i,t in enumerate(all4con_times):
#     r4[i] = t/mintimes[i]
# r2 = np.zeros(p)
# for i,t in enumerate(signdep_times):
#     r2[i] = t/mintimes[i]
#
# #compute rmax, and replace nan values with 2*rmax
# rmax = 2*max(np.nanmax(r2),np.nanmax(r4))
# r4[np.isnan(r4)] = rmax*2
# r2[np.isnan(r2)] = rmax*2
#
# #sort
# r4 = np.sort(r4)
# r2 = np.sort(r2)




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
