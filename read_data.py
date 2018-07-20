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

def performance_profile(df, save_loc, variable, formulations):
    """
    convert run_trials report into form for performnace profile generation
    """
    data = {}
    for form in formulations:
        form_rows = df[df[variable]==form]
        #chose from "instance_total_time", "instance_solve_time", "instance_setup_time", "instance_gap"
        form_data = form_rows["instance_solve_time"].tolist()
        data[form] = form_data
    df = pd.DataFrame(data)

    #save converted report as excel file
    writer = pd.ExcelWriter(save_loc+"perf_profile_report.xlsx", engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.save()

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
        #print(xf)
        plt.plot(xf,yf, drawstyle="steps", linestyle=linestyles[f % len(linestyles)])
    plt.title("Performance Profile For {}".format(variable))
    plt.xlabel('Factor of Best Ratio')
    plt.ylabel('Probability')
    plt.legend([form for form in formulations])
    plt.axis(xmin=1) #can set xmax to zoom in/out

    plt.savefig(save_loc+"performance_profile")
    plt.show()

def make_bar_graph(df, save_loc, variable, formulations):
    data = {}
    y = []
    stds = []
    x = []
    for form in formulations:
        x.append(form)
        form_rows = df[df[variable]==form]
        #chose from "instance_total_time", "instance_solve_time", "instance_setup_time", "instance_gap"
        #TODO make sure getting total_time not just solve time. (usually we want total_time)
        form_data = form_rows["instance_solve_time"].tolist()
        #print(sum(form_data)) ---this is the total solve time for formulation
        #plt.bar(form, sum(form_data)/len(form_data))
        #plt.bar(form, sum(form_data), yerr=np.std(form_data))
        #TODO how to handle nan values here?? currently ignoring them..
        y.append(np.nanmean(form_data))
        #stds.append(np.nanstd(form_data))
    plt.bar(x,y, color=["C0","C1", "C2", "C3", "C4"], tick_label=formulations) #can set std with yerr=[]
    plt.title("Solve Time Comparison for {}".format(variable))
    plt.xlabel("Formulation")
    plt.ylabel("Average Solve Time")
    plt.legend([form for form in formulations])
    plt.savefig(save_loc+"bar_graph") #format="pdf" to save as pdf (default is png)
    plt.show()

def analyze(df_name, new_folder_name, test_variable, formulations, specifications=None):
    """
    param df_name: name of dataframe from dataframes folder to load in
    param new_folder_name: will create a new folder with this name containing new graphs and excel files
    param test_variable: the variable to be analyzed. (can use any column name from dataframe) (ie. "solver", "glover_bounds", etc..)
    param formulations: the set of options for that variable (ie. ["tight", "tighter", "original"] for "glover_bounds" as test_variable)
    """
    #read in dataframe from dataframes folder
    df = pd.read_pickle("dataframes/{}.pkl".format(df_name))
    #df = df[:-5]   #--use this to cut off (5) rows from end if uneven length
    #df = df[df["density"]==50]

    #Create a new data folder, while making sure to not overwrite existing data
    mypath = "data/{}/".format(new_folder_name)
    if new_folder_name=="test" or not os.path.isdir(mypath):
        if not new_folder_name=="test":
            os.makedirs(mypath)
        #save a copy of the dataframe being analyzed
        df.to_pickle(mypath+'/dataframe.pkl')
        #generate an excel report
        write_report(df, save_loc=mypath)
        #generate a performance profile graph
        performance_profile(df, save_loc=mypath+"aggregate_", variable=test_variable, formulations=formulations)
        #generate a bar graph showing solve times
        make_bar_graph(df, save_loc=mypath+"aggregate_", variable=test_variable, formulations=formulations)


        #specify which rows to consider if desired. (ie. can only look at data for "QKP"))
        if specifications==None:
            return
        for key,values in specifications.items():
            for value in values:
                sub_df = df[df[key]==value]
                #generate a performance profile graph
                performance_profile(sub_df, save_loc=mypath+key+"_"+str(value)+"_", variable=test_variable, formulations=formulations)
                #generate a bar graph showing solve times
                make_bar_graph(sub_df, save_loc=mypath+key+"_"+str(value)+"_", variable=test_variable, formulations=formulations)
    else:
        raise Exception("Save path folder {} already exists. ".format(mypath))


"""
variable options: 'solver', 'method', 'glover_bounds', 'options' (ie. any column headers from report)
corresponding formulation options:
    solver : ['cplex', 'gurobi', 'xpress']
    method : ['glover', 'std']
    glover_bounds : ['original', 'tight', 'tighter']
    glover_cons : ['original', 'sub1', 'sub2']
"""

specs = {"density":[25,50,75,100], "type":["QKP", "KQKP", "HSP", "UQP", "QSAP"]}
analyze(df_name='batch1', new_folder_name='test4', specifications=specs,
            test_variable="symmetric", formulations=[0,1])











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
