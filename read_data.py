import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os


def write_report(df, save_loc):
    """
    Generates an excel spreadsheet containing all info from a given dataframe.

    param df: the dataframe to be converted
    param save_loc: location of the folder in which the excel file will be saved
    """

    writer = pd.ExcelWriter(save_loc+"report.xlsx", engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')

    #custom color coding makes good and bad solve times stick out
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    worksheet.conditional_format('P2:P5000',{'type':'3_color_scale', 'min_color':'green', 'max_color':'red'})

    writer.save()

def performance_profile(df, save_loc, variable, formulations):
    """
    Creates a performance profile graph

    param df:
    param save_loc:
    param variable: The variable being looked at (ie. "glover_bounds")
    param formulations: list of options for given variable (ie. ["original", "tight", "tighter"])
    """
    #create sub-dataframe containing a column for each formulation, with rows for each solve time
    data = {}
    for form in formulations:
        form_rows = df[df[variable]==form]
        #chose from "instance_total_time", "instance_solve_time", "instance_setup_time", "instance_gap"
        form_data = form_rows["instance_total_time"].tolist()
        data[form] = form_data
    df = pd.DataFrame(data)

    #save converted report as excel file called "perf_profile_report"
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
        plt.plot(xf,yf, drawstyle="steps", linestyle=linestyles[f % len(linestyles)])
    plt.title("Performance Profile For {}".format(variable))
    plt.xlabel('Factor of Best Ratio')
    plt.ylabel('Probability')
    plt.legend([form for form in formulations])

    plt.axis(xmin=1, xmax=5) #can set xmax to zoom in/out #TODO make xmax a variable. or base automatically on ratio?

    plt.savefig(save_loc+"performance_profile_"+variable)
    plt.close()

def make_bar_graph(df, save_loc, variable, formulations):
    """
    Creates a bar graph showing average solve times for each formulation. good complement to performance profile
    """
    for form in formulations:
        #retrieve times for formulation
        form_rows = df[df[variable]==form]
        form_data = form_rows["instance_total_time"].tolist()

        #plot formulation's average solve time
        bar = plt.bar(form, np.nanmean(form_data))[0]

        #count how many solvetimes are NaN (ie. how many didn't solve within time limit), and mark this on graph
        num_nans = (str(np.isnan(form_data).sum())+" tl")
        height = bar.get_height()
        plt.text(bar.get_x()+bar.get_width()/2, height, num_nans, ha='center', va='bottom')

    plt.title("Solve Time Comparison for {}".format(variable))
    plt.xlabel("Formulation")
    plt.ylabel("Average Solve Time")
    plt.legend([form for form in formulations])
    plt.savefig(save_loc+"bar_graph_"+variable) #format="pdf" to save as pdf (default is png)
    plt.close()

def make_line_graph(dfs, save_loc, variable, formulations):
    """
    Creates a line graph showing average solve times for each formulation across problem size. good complement to performance profile
    """
    for form in formulations:
        avg_times = []
        num_nans = []
        for df in dfs:
            #retrieve times for formulation
            form_rows = df[df[variable]==form]
            form_rows = form_rows[form_rows["type"]!="HSP"]
            form_data = form_rows["instance_total_time"].tolist()
            avg_times.append(np.nanmean(form_data))
            num_nans.append(np.isnan(form_data).sum())
        plt.plot(range(len(dfs)), avg_times, label=str(form)+" : "+str(sum(num_nans))+" time limit")
    plt.xticks(np.arange(len(dfs)),['small','medium','big'])
    plt.title("Solve Time Comparison for {}".format(variable))
    plt.xlabel("Sizes")
    plt.ylabel("Average Solve Time")
    plt.legend()
    plt.show()

def make_line_graph(dfs, save_loc, variable, formulations):
    """
    Creates a line graph showing average solve times for each formulation across problem size. good complement to performance profile
    """
    for form in formulations:
        avg_times = []
        labels = []
        for df in dfs:
            #retrieve times for formulation
            form_rows = df[df[variable]==form]
            form_rows = form_rows[form_rows["type"]=="QKP"]
            unique_sizes = form_rows["size"].unique()
            unique_sizes.sort()
            for size in unique_sizes:
                corresponding_density = form_rows[form_rows["size"]==size]["density"].unique()[0]
                labels.append((size,corresponding_density))
                size_data = form_rows[form_rows["size"]==size]
                size_time = size_data["instance_total_time"].tolist()
                avg_times.append(np.nanmean(size_time))
        plt.plot(range(4), avg_times)
    plt.xticks(np.arange(4),labels)
    plt.title("Solve Time Comparison for {}".format(variable))
    plt.xlabel("(Size,Density)")
    plt.ylabel("Average Solve Time (s)")
    plt.show()

def analyze(df_name, new_folder_name, test_variable, formulations, specifications=None):
    """
    Generates a suite of performance profiles, graphs, and excel spreadsheets for analysis of a dataset. This is the only
    method in this module which needs to be called

    param df_name: name of dataframe from dataframes folder to load in
    param new_folder_name: will create a new folder with this name containing new graphs and excel files
    param test_variable: the variable to be analyzed. (can use any column name from dataframe) (ie. "solver", "glover_bounds", etc..)
    param formulations: the set of options for that variable (ie. ["tight", "tighter", "original"] for "glover_bounds" as test_variable)
    param specifications:
    """
    #read in dataframe from dataframes folder
    df = pd.read_pickle("dataframes/{}.pkl".format(df_name))

    #uncomment this to replace values that should be NaN
    mask = df["instance_total_time"] > 600 #~where 600 was timelimit
    df.loc[mask, "instance_total_time"] = np.nan

    #df = df[:-5]   #--use this to cut off (5) rows from end if uneven length
    #df = df[df["density"]==50]

    #Create a new data folder, while making sure to not overwrite existing data
    mypath = "data/{}/".format(new_folder_name)
    if new_folder_name=="test" or not os.path.isdir(mypath):
        if not new_folder_name=="test":
            os.makedirs(mypath)
        os.makedirs(mypath+"/aggregate/")
        #save a copy of the dataframe being analyzed
        df.to_pickle(mypath+'/aggregate/dataframe.pkl')
        #generate an excel report
        write_report(df, save_loc=mypath+"/aggregate/")
        #generate a performance profile graph for the aggregated data
        performance_profile(df, save_loc=mypath+"/aggregate/aggregate_", variable=test_variable, formulations=formulations)
        #generate a bar graph showing solve times for the aggregated data
        make_bar_graph(df, save_loc=mypath+"/aggregate/aggregate_", variable=test_variable, formulations=formulations)

        if specifications==None:
            return
        #create graphs for subsets of data as desired (specified in specificaitons)
        for key,values in specifications.items():
            os.makedirs(mypath+"/by_"+key+"/")
            for value in values:
                sub_df = df[df[key]==value]
                #generate a performance profile graph
                performance_profile(sub_df, save_loc=mypath+"/by_"+key+"/"+key+"_"+str(value)+"_", variable=test_variable, formulations=formulations)
                #generate a bar graph showing solve times
                make_bar_graph(sub_df, save_loc=mypath+"/by_"+key+"/"+key+"_"+str(value)+"_", variable=test_variable, formulations=formulations)
    else:
        raise Exception("Save path folder {} already exists. ".format(mypath))

    #open up new data folder for viewing
    os.chdir(mypath)
    os.system("start .")

def helpful_snippets():
    """
    Do not call this function. Simply contains potentially useful code snippets
    """
    #read in 2 dataframes and combine them into one
    df1 = pd.read_pickle("dataframes/{}.pkl".format("batch2_original_cons"))
    df2 = pd.read_pickle("dataframes/{}.pkl".format("batch2"))
    df = df2.append(df1)
    df.to_pickle('dataframes/batch2_combined.pkl')

    #~~if you want the time saved in a filename
    time_stamp = time.strftime("%Y_%m_%d-%H_%M_%S")

    #~~to automatically open an excel file for viewing
    os.chdir("reports")
    os.system(excel_filename)

    #read in a dataframe from an excel file. choose desired rows and columns and create a simple graph
    data=pd.read_excel(open("C:/Users/huntisan/Desktop/summer2018/std_glove_data.xlsx", "rb"))
    relevant_data = data.loc[:,["method", "size", "avg_solve_time"]]
    big_sizes = relevant_data.loc[:, "size"] > 90
    big_sizes = relevant_data[big_sizes]
    x = big_sizes.loc[:,"size"]
    y = big_sizes.loc[:,"avg_solve_time"]
    plt.scatter(x,y)
    plt.show()

    #FOR BATCH FILE
    if(len(sys.argv)==5): #batch file will go through here
        run_trials(trials=5,type=sys.argv[1],method=sys.argv[2],den=int(sys.argv[3]),size=int(sys.argv[4]))


"""
variable options: 'solver', 'method', 'glover_bounds', 'options' (ie. any column headers from report)
corresponding formulation options:
    solver : ['cplex', 'gurobi', 'xpress']
    method : ['glover', 'std']
    glover_bounds : ['original', 'tight', 'tighter']
    glover_cons : ['original', 'sub1', 'sub2']
"""

# specs = {"density":[25,50,75,100], "type":["QKP", "KQKP", "HSP"]}
# analyze(df_name='batch3_reorder_smaller', new_folder_name='testing', specifications=specs,
#    test_variable="options", formulations=[0,1,2,3,4])

df = pd.read_pickle("dataframes/{}.pkl".format("batch3_reorder_smaller"))
df2 = pd.read_pickle("dataframes/{}.pkl".format("batch3_reorder"))
df3 = pd.read_pickle("dataframes/{}.pkl".format("batch3_reorder_bigger"))
make_line_graph([df2],save_loc="",variable="options",formulations=[0,1,2,3,4])
