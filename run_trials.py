from quadratics import *
import cplex_solve as cplex
import gurobi_solve as gurobi
import xpress_solve as xpress
import sys
import time
import pandas as pd
from timeit import default_timer as timer


def run_trials(data_, trials=5,solver="cplex",type="QKP",reorder=False,symmetric=False,
			method="std",size=5,multiple=1,den=100, options=0,glover_bounds="tight", glover_cons="original",mixed_sign=False):
	"""
	Runs the same problem type thru given solver with given method repeatedly to get
	an average solve time
	"""
	#keep track of total run time across all trials to compute avg later
	setup_time_sum, solve_time_sum, int_gap_sum, obj_sum = 0,0,0,0
	#need individual run time to compute standard deviation
	instance_total_times = []

	#description = solver+"-"+type+"-"+str(method)+"-"+str(size)+"-"+str(den)+"-"+str(multiple)+'-'+str(glover_bounds)+"-"+glover_cons+"-"+options
	description = "{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(solver,type,symmetric,method,size,den,multiple,options,glover_bounds,glover_cons,mixed_sign)
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
					quad = Knapsack(seed=i+size+den, n=size, m=multiple, density=den, symmetric=symmetric, mixed_sign=mixed_sign)
			elif type=="KQKP":
					quad = Knapsack(seed=i+size+den, n=size, m=multiple, k_item=True, density=den, symmetric=symmetric, mixed_sign=mixed_sign)
			elif type=="HSP":
					quad = HSP(seed=i+size+den, n=size, density=den, symmetric=symmetric)
			elif type=="UQP":
					quad = UQP(seed=i+size+den, n=size, density=den, symmetric=symmetric)
			elif type=="QSAP":
					#qsap always 100% dense, so den is used instead to represent m. number of tasks
					quad = QSAP(seed=i+size+den, n=size, m=den)
			else:
				raise Exception(str(type) + " is not a valid problem type")

			if reorder==True:
				quad.reorder()

			def get_solver(solver_):
				if solver_=="cplex":
					return cplex
				elif solver=="gurobi":
					return gurobi
				elif solver=="xpress":
					return xpress

			cur_solver = get_solver(solver)

			def get_method(method_):
				if method_=="std":
					return cur_solver.standard_linearization
				elif method=="glover":
					return cur_solver.glovers_linearization
				elif method=="glover_rlt":
					return cur_solver.glovers_linearization_rlt
				elif method=="glover_prlt":
					return cur_solver.glovers_linearization_prlt
				elif method=="glover_qsap":
					return cur_solver.qsap_glovers
				elif method=="no_lin":
					return cur_solver.no_linearization
				else:
					raise Exception(str(method_) + " is not a valid method type")

			cur_method = get_method(method)
			#try:
			m = cur_method(quad, bounds=glover_bounds, constraints=glover_cons, lhs_constraints=options, use_diagonal=False)
			#except:
			# f.write("TRIAL FAILED - FAILURE DURING MODELING\n")
			# f.write("=============================================\n")
			# print("TRIAL FAILED - FAILURE DURING MODELING")
			#try:
			if method=="no_lin":
				results = cur_solver.solve_model(m[0], solve_relax=False)
			else:
				results = cur_solver.solve_model(m[0])

			#retrieve info from solving instance
			instance_setup_time = m[1]
			instance_solve_time = results.get("solve_time")
			instance_obj_val = results.get("objective_value")
			instance_int_gap = results.get("integrality_gap")
			instance_relax = results.get("relaxed_solution")
			time_limit = results.get("time_limit")
			instance_total_time = instance_setup_time + instance_solve_time
			#print instance solve info to log file
			if(time_limit):
				instance_solve_time = np.nan
				instance_int_gap = np.nan
				f.write("TIME LIMIT REACHED\n")
			f.write("Integer Solution: " + str(instance_obj_val)+"\n")
			f.write("Continuous Solution: " + str(instance_relax)+"\n")
			f.write("Integrality Gap: " + str(instance_int_gap)+"\n")
			f.write("Setup Time: " + str(instance_setup_time)+"\n")
			f.write("Solve Time: " + str(instance_solve_time)+"\n")
			f.write("Instance Total Time (Setup+Solve): " + str(instance_total_time)+"\n")
			f.write("=============================================\n")

			#update running totals across trials for computing averages
			instance_total_times.append(instance_total_time)
			setup_time_sum += instance_setup_time
			solve_time_sum += instance_solve_time
			obj_sum += instance_obj_val
			int_gap_sum += instance_int_gap
			print("trial number {:2} took {:7.2f} seconds to solve".format(i,instance_total_time))
			#except:
			#trials-=1
			#f.write("TRIAL FAILED - FAILED TO SOLVE MODEL\n")
			#f.write("=============================================\n")
			#print("TRIAL FAILED - FAILED TO SOLVE MODEL")
			results = {"trial":i, "solver":solver, "type":type, "method":method, "options":options, "size":size, "density":den, "instance_gap":instance_int_gap,
					"instance_total_time":instance_total_time, "instance_obj_val":instance_obj_val, "symmetric": symmetric,
					 "instance_setup_time": instance_setup_time,"instance_solve_time": instance_solve_time, "glover_bounds": glover_bounds,
					  "mixed_sign": mixed_sign, "reorder":reorder, "multiple":multiple, "glover_cons":glover_cons, "id":filename}
					  #TODO make id auto not include whatever variable we are looking at.
			data_.append(results)
			#TODO this list is going to get huge....

		df = pd.DataFrame(data_)
		df = df[["trial","solver", "type","reorder","mixed_sign", "symmetric", "method","glover_bounds", "glover_cons", "options","size",
					"density", "multiple", "instance_gap","instance_setup_time", "instance_solve_time", "instance_total_time", "instance_obj_val"]]  #reorder columns
		df.to_pickle('dataframes/test.pkl')
		#return results across trials
		results = {"solver":solver, "type":type, "method":method, "options":options, "size":size, "density":den, "avg_gap":int_gap_sum/trials,
					"avg_total_time":(setup_time_sum+solve_time_sum)/trials, "std_dev":np.std(instance_total_times),
					"avg_obj_val":obj_sum/trials, "symmetric": symmetric, "avg_setup_time": setup_time_sum/trials,
					"avg_solve_time": solve_time_sum/trials, "glover_bounds": glover_bounds, "mixed_sign": mixed_sign, "reorder":reorder,
					"multiple":multiple, "glover_cons":glover_cons}

		#print results summary to log file by iterating through results dictionary
		f.write("\n\nSummary Statistics\n")
		f.write("=============================================\n")
		for key,val in results.items():
			f.write(key + " : " + str(val)+"\n")

		#return results dictionary
		return results

if __name__=="__main__":
	"""
	solver = solver to use ("cplex", "gurobi")
	type = problem type ("QKP", "KQKP", "UQP", "HSP", "QSAP")
	method = linearization technique ("std", "glover", "glover_rlt", "glover_prlt", "qsap_glover", "no_lin")
	glover_bounds = simple ub, continuous program ub, or binary ub ("original", "tight", "tighter")
	glover_cons = ("original", "sub1", "sub2")
	options = specify alternative/optional constraints specific to each linearization
	"""
	start = timer()
	num_trials = 5
	sizes = [75]
	densities = [25,50,75,100]
	solvers = ["xpress", "cplex", "gurobi"]
	types = ["UQP"]
	bounds = ["-"]
	cons = ["-"]
	methods = ["std"]
	signs = ["-"]
	multiples = [1]
	data = []
	for j in densities:
		for solve_with in solvers:
			for i in sizes:
				for type in types:
					for bound in bounds:
						for con in cons:
							for method in methods:
								for sign in signs:
									for mult in multiples:
										print("running-( {} , {} , {} , {} , {} , all4cons-{} , mixed_sign-{} , {} )".format(solve_with.upper(),i,j,type,method,"FALSE",sign, mult))
										run_trials(data_=data,trials=num_trials, solver=solve_with, type=type,method=method, symmetric=False,
														glover_bounds=bound, glover_cons=con, size=i, den=j, multiple=mult, options=0, reorder=False, mixed_sign=sign)
										#data.append(dict)
										print("running-( {} , {} , {} , {} , {} , all4cons-{} , mixed_sign-{} , {} )".format(solve_with.upper(),i,j,type,method,"TRUE",sign, mult))
										run_trials(data_=data,trials=num_trials, solver=solve_with, type=type,method=method, symmetric=False,
														glover_bounds=bound, glover_cons=con, size=i, den=j, multiple=mult, options=1, reorder=False, mixed_sign=sign)
										#data.append(dict)
										#repeadetely save to DF so we don't lose any data
										#df = pd.DataFrame(data)
										#df = df[["solver", "type","reorder","mixed_sign", "symmetric", "method","glover_bounds", "glover_cons", "options","size",
										# "density", "multiple", "avg_gap","avg_setup_time", "avg_solve_time", "avg_total_time", "std_dev", "avg_obj_val"]]  #reorder columns
										#df.to_pickle('dataframes/testing.pkl')

	#save to excel file (name = timestamp)
	# time_stamp = time.strftime("%Y_%m_%d-%H_%M_%S")
	# excel_filename = "reports/"+time_stamp+'-report.xlsx'
	# writer = pd.ExcelWriter(excel_filename, engine='xlsxwriter')
	# df.to_excel(writer, index=False)
	# writer.save()
	# print(df)
	end = timer()
	print("took {:.3} seconds to run all trials".format(end-start))



# FOR BATCH FILE
# if(len(sys.argv)==5): #batch file will go through here
# run_trials(trials=5,type=sys.argv[1],method=sys.argv[2],den=int(sys.argv[3]),size=int(sys.argv[4]))
