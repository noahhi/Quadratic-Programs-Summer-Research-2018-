from quadratics import *
import cplex_solve as cplex
import gurobi_solve as gurobi
import sys
import time
import pandas as pd
from timeit import default_timer as timer


def run_trials(trials=5,solver="cplex",type="QKP",method="std",size=5,den=100, options=0):
	"""
	Runs the same problem type thru given solver with given method repeatedly to get
	an average solve time
	"""
	#keep track of total run time across all trials to compute avg later
	total_time, total_gap, total_obj = 0, 0, 0
	#need individual run time to compute standard deviation
	run_times = []

	#write data to log file with descriptive name
	description = solver+"-"+type+"-"+str(method)+"-"+str(size)+"-"+str(den)
	filename = "log/"+description+".txt"
	seperator = "=============================================\n"

	with open(filename, "w") as f:
		#header information
		f.write(seperator)
		f.write("Solver: " + str(solver) +"\n")
		f.write("Problem Type: " + str(type) +"\n")
		f.write("Method: " + method+"\n")
		f.write("nSize: " + str(size) +"\n")
		f.write("Density: " + str(den) +"\n")
		f.write("Trials: " + str(trials) +"\n")
		f.write(seperator+"\n\n")

		for i in range(trials):
			f.write("Iteration "+str(i+1)+"\n")

			#generate problem instance
			if type=="QKP":
				quad = Knapsack(seed=i, n=size, density=den)
			elif type=="KQKP":
				quad = Knapsack(seed=i, n=size, k_item=True, density=den)
			elif type=="HSP":
				quad = HSP(seed=i, n=size, density=den)
			elif type=="UQP":
				quad = UQP(seed=i, n=size, density=den)
			else:
				raise Exception(str(type) + " is not a valid problem type")

			#model problem with given solver/method
			if(solver=="cplex"):
				if method=="std":
					if options==0:
						m = cplex.standard_linearization(quad)
					elif options==1:
						m = cplex.standard_linearization(quad, con3=False, con4=False)
					elif options==2:
						m = cplex.standard_linearization(quad, con1=False, con2=False)
				elif method=="glover":
					if options==0:
						m = cplex.glovers_linearization(quad, bounds="tight")
					elif options==1:
						m = cplex.glovers_linearization(quad, bounds="original")
				elif method=="glover_rlt":
					m = cplex.glovers_linearization_rlt(quad)
				elif method=="glover_prlt":
					m = cplex.glovers_linearization_prlt(quad)
				else:
					raise Exception(str(method) + " is not a valid method type")
				results = cplex.solve_model(m[0])
			elif(solver=="gurobi"):
				if method=="std":
					if options==0:
						m = gurobi.standard_linearization(quad)
					elif options==1:
						m = gurobi.standard_linearization(quad, con3=False, con4=False)
					elif options==2:
						m = gurobi.standard_linearization(quad, con1=False, con2=False)
				elif method=="glover":
					if options==0:
						m = gurobi.glovers_linearization(quad, bounds="tight")
					elif options==1:
						m = gurobi.glovers_linearization(quad, bounds="original")
				elif method=="glover_rlt":
					m = gurobi.glovers_linearization_rlt(quad)
				elif method=="glover_prlt":
					m = gurobi.glovers_linearization_prlt(quad)
				else:
					raise Exception(str(method) + " is not a valid method type")
				results = gurobi.solve_model(m[0])
			else:
				raise Exception(str(solver) + "is not a valid solver type")

			#retrieve setup time from modeling process and results from solve
			setup_time = m[1]
			solve_time = results.get("solve_time")
			obj_val = results.get("objective_value")
			int_gap = results.get("integrality_gap")
			relax = results.get("relaxed_solution")
			instance_time = setup_time+solve_time

			#running totals across trials
			run_times.append(instance_time)
			total_time += instance_time
			total_obj += obj_val
			total_gap += int_gap

			f.write("Integer Solution: " + str(obj_val)+"\n")
			f.write("Continuous Solution: " + str(relax)+"\n")
			f.write("Integrality Gap: " + str(int_gap)+"\n")
			f.write("Setup Time: " + str(setup_time)+"\n")
			f.write("Solve Time: " + str(solve_time)+"\n")
			f.write("Instance Total Time (Setup+Solve): " + str(instance_time)+"\n")
			f.write("=============================================\n")

		results = {"solver":solver, "type":type, "method":method, "options":options, "size":size, "density":den, "avg_gap":total_gap/trials,
					"avg_solve_time":total_time/trials, "std_dev":np.std(run_times), "avg_obj_val":total_obj/trials}

		#print summary by iterating thru results dict
		f.write("\n\nSummary Statistics\n")
		f.write("=============================================\n")
		f.write("Total solve time: " + str(total_time)+"\n")
		f.write("Average Solve Time: " + str(total_time/trials)+"\n")
		f.write("Solve Time Standard Deviation: " + str(np.std(run_times))+"\n")
		f.write("Average Integrality Gap: " + str(total_gap/trials)+"\n")

		return results

if __name__=="__main__":
	start = timer()
	num_trials = 2
	sizes = [20]
	densities = [100]
	data = []
	for i in sizes:
		for j in densities:
			"""
			solver = solver to use ("cplex", "gurobi")
			type = problem type ("QKP", "KQKP", "UQP", "HSP")
			method = linearization technique ("std", "glover", "glover_rlt", "glover_prlt")
			options = specify alternative/optional constraints specific to each linearization
			"""
			print("current(size,density) = ("+str(i)+","+str(j)+")")
			dict = run_trials(trials=num_trials, solver="cplex", type="QKP", method="std", size=i, den=j)
			data.append(dict)
			dict = run_trials(trials=num_trials, solver="cplex", type="QKP", method="glover", size=i, den=j)
			data.append(dict)
			dict = run_trials(trials=num_trials, solver="cplex", type="QKP", method="glover_rlt", size=i, den=j)
			data.append(dict)
			dict = run_trials(trials=num_trials, solver="cplex", type="QKP", method="glover_prlt", size=i, den=j)
			data.append(dict)
			#
			dict = run_trials(trials=num_trials, solver="gurobi", type="QKP", method="std", size=i, den=j)
			data.append(dict)
			dict = run_trials(trials=num_trials, solver="gurobi", type="QKP", method="glover", size=i, den=j)
			data.append(dict)
			dict = run_trials(trials=num_trials, solver="gurobi", type="QKP", method="glover_rlt", size=i, den=j)
			data.append(dict)
			dict = run_trials(trials=num_trials, solver="gurobi", type="QKP", method="glover_prlt", size=i, den=j)
			data.append(dict)


	df = pd.DataFrame(data)
	df = df[["solver", "type", "method","options", "size", "density", "avg_gap", "avg_solve_time", "std_dev", "avg_obj_val"]]  #reorder columns
	print(df)

	time_stamp = time.strftime("%Y_%m_%d-%H_%M_%S")
	excel_filename = "reports/"+time_stamp+'-report.xlsx'
	writer = pd.ExcelWriter(excel_filename, engine='xlsxwriter')
	df.to_excel(writer, index=False)
	writer.save()
	end = timer()
	print("took " + str(end-start) + " seconds to run all trials")


# FOR BATCH FILE
# if(len(sys.argv)==5): #batch file will go through here
# run_trials(trials=5,type=sys.argv[1],method=sys.argv[2],den=int(sys.argv[3]),size=int(sys.argv[4]))
