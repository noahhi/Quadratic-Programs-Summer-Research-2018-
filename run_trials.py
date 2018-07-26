from quadratics import *
import cplex_solve as cplex
import gurobi_solve as gurobi
import xpress_solve as xpress
import sys
import time
import pandas as pd
from timeit import default_timer as timer

TIME_LIMIT = 3600
DATAFRAME_NAME = "test"
NUM_TRIALS = 10

#TODO reorder should be false by default
def run_trials(data_, solver="cplex",type="QKP",reorder=True,symmetric=False,
			method="std",size=5,multiple=1,den=100, options=0,glover_bounds="tight", glover_cons="original",mixed_sign=False):
	"""
	Runs a set of trials on a given problem formulation
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
		f.write("Trials: " + str(NUM_TRIALS) +"\n")
		f.write(seperator+"\n\n")

		for i in range(NUM_TRIALS):
			#time_stamp = time.strftime("%H_%M")
			#print("starting trial number {:2} at {}".format(i,time_stamp))
			#print("starting at {}".format(time_stamp))
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
					quad = QSAP(seed=i+size+den, n=size, m=den, symmetric=symmetric)
			else:
				raise Exception(str(type) + " is not a valid problem type")

			start_reorder = timer()
			if reorder==True:
				if options==1:
					quad.reorder(take_max=False, flip_order=False)
				elif options==2:
					quad.reorder(take_max=False, flip_order=True)
				elif options==3:
					quad.reorder(take_max=True, flip_order=False)
				elif options==4:
					quad.reorder(take_max=True, flip_order=True)
				elif options==5:
					quad.reorder_refined(take_max=False, flip_order=False, k=3)
				elif options==6:
					quad.reorder_refined(take_max=True, flip_order=True, k=3)
				elif options==7:
					quad.reorder_refined(take_max=False, flip_order=False, k=5)
				elif options==8:
					quad.reorder_refined(take_max=True, flip_order=True, k=5)
			reorder_time = timer() - start_reorder

			def get_solver(solver_):
				if solver_=="cplex":
					return cplex
				elif solver=="gurobi":
					return gurobi
				elif solver=="xpress":
					return xpress

			cur_solver = get_solver(solver)

			def get_method(method_):
				if type=="QSAP":
					if method=="std" or method=="standard":
						return cur_solver.qsap_standard
					elif method=="glover" or method=="glovers":
						return cur_solver.qsap_glovers
					elif method=="glovers_rlt":
						return cur_solver.qsap_rlt
					elif method=="elf":
						return cur_solver.qsap_elf
					elif method=="sslf":
						return cur_solver.qsap_ss
					else:
						raise Exception(str(method_) + " is not a valid method type")
				else:
					if method=="std" or method=="standard":
						return cur_solver.standard_linearization
					elif method=="glover" or method=="glovers":
						return cur_solver.glovers_linearization
					elif method=="glovers_rlt" or method=="rlt":
						return cur_solver.glovers_linearization_rlt
					elif method=="elf":
						return cur_solver.extended_linear_formulation
					elif method=="sslf":
						return cur_solver.ss_linear_formulation
					elif method=="no_lin":
						return cur_solver.no_linearization
					else:
						raise Exception(str(method_) + " is not a valid method type")

			cur_method = get_method(method)
			m = cur_method(quad, bounds=glover_bounds, constraints=glover_cons, lhs_constraints=False, use_diagonal=False)

			if method=="no_lin":
				results = cur_solver.solve_model(m[0], solve_relax=False, time_limit=TIME_LIMIT)
			else:
				results = cur_solver.solve_model(m[0], time_limit=TIME_LIMIT)

			#note that this setup time is set to be 0 unless significant time operations are performed (ie. computing tight bounds)
			instance_setup_time = m[1]
			if reorder:
				instance_setup_time += reorder_time # add time to reorder to setuptime
			instance_solve_time = results.get("solve_time")
			instance_obj_val = results.get("objective_value")
			instance_int_gap = results.get("integrality_gap")
			instance_relax = results.get("relaxed_solution")
			time_limit = results.get("time_limit")
			instance_total_time = instance_setup_time + instance_solve_time
			#print instance solve info to log file
			if(time_limit):
				#if problem didn't solve within time limit, enter solve time as NaN
				instance_solve_time = np.nan
				instance_setup_time = np.nan
				instance_total_time = np.nan
				instance_int_gap = np.nan
				f.write("TIME LIMIT REACHED\n")
				print("TIME LIMIT REACHED")
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

			results = {"trial":i, "solver":solver, "type":type, "method":method, "options":options, "size":size, "density":den, "instance_gap":instance_int_gap,
					"instance_total_time":instance_total_time, "instance_obj_val":instance_obj_val, "symmetric": symmetric,
					 "instance_setup_time": instance_setup_time,"instance_solve_time": instance_solve_time, "glover_bounds": glover_bounds,
					  "mixed_sign": mixed_sign, "reorder":reorder, "multiple":multiple, "glover_cons":glover_cons}
			data_.append(results)
			#TODO this list is going to get huge....

		df = pd.DataFrame(data_)
		#reorder columns since dict is ordered randomly by default
		df = df[["trial","solver", "type","reorder","mixed_sign", "symmetric", "method","glover_bounds", "glover_cons", "options","size",
					"density", "multiple", "instance_gap","instance_setup_time", "instance_solve_time", "instance_total_time", "instance_obj_val"]]
		#save the dataframe in pickle format
		df.to_pickle('dataframes/{}.pkl'.format(DATAFRAME_NAME))
		#return results across trials
		results = {"solver":solver, "type":type, "method":method, "options":options, "size":size, "density":den, "avg_gap":int_gap_sum/NUM_TRIALS,
					"avg_total_time":(setup_time_sum+solve_time_sum)/NUM_TRIALS, "std_dev":np.std(instance_total_times),
					"avg_obj_val":obj_sum/NUM_TRIALS, "symmetric": symmetric, "avg_setup_time": setup_time_sum/NUM_TRIALS,
					"avg_solve_time": solve_time_sum/NUM_TRIALS, "glover_bounds": glover_bounds, "mixed_sign": mixed_sign, "reorder":reorder,
					"multiple":multiple, "glover_cons":glover_cons}

		#print results summary to log file by iterating through results dictionary
		f.write("\n\nSummary Statistics\n")
		f.write("=============================================\n")
		for key,val in results.items():
			f.write(key + " : " + str(val)+"\n")

		#return results dictionary
		return results

if __name__=="__main__":
	data = [] # this list will be filled with dictionaries containing all run_stats; used to make dataframes

	# run trials constants
	NUM_TRIALS = 5 # number of instances of each formulation to generate and solve
	TIME_LIMIT = 3600 # max amount of time (seconds) to spend on each problem
	DATAFRAME_NAME = "test" # dataframe will save to \dataframes\DATAFRAME_NAME

	# variables. (each corresponds to a column of resulting dataframe)
	SOLVERS = ["cplex", "gurobi", "xpress"] # 'solver' = {"cplex", "gurobi", "xpress"}
	TYPES = ["QKP", "KQKP", "UQP", "HSP", "QSAP"] # 'type' = {"QKP", "KQKP", "UQP", "HSP", "QSAP"}
	REORDER = [True, False] # 'reorder' = {True, False}
	MIXED_SIGN = [True, False] # 'mixed_sign' = {True, False} ~true->mixed, false->nonnegative
	SYMMETRIC = [True, False] # 'symmetric' = {True, False} ~true->symmetric, false->upper triangular
	METHODS = ["standard", "glovers", "glovers_rlt", "elf", "sslf"] # 'method' = {"standard", "glovers", "glovers_rlt", "elf", "sslf"}
	BOUNDS = ["original", "tight", "tighter"] # 'glover_bounds' = {"original", "tight", "tighter"}
	CONSTRAINTS = ["original", "sub1", "sub2"] # 'glovers_cons' = {"original", "sub1", "sub2"}
	OPTIONS = [0,1,2,3,4] # 'options' = {0-4} ~Used for reordering; or other misc things
	MULTIPLE = [1, 5, 10] # 'multiple' = {1-10} ~only does anything for QKP; adds multiple Knapsack Constraints
	SIZES = []
	DENSITIES = []

	for solver in SOLVERS:
		for size in SIZES:
			for density in DENSITIES:
				for type in TYPES:
					for order in REORDER:
						for sign in MIXED_SIGN:
							for method in METHODS:
								for bound in BOUNDS:
									for constraint in CONSTRAINTS:
										for opt in OPTIONS:
											for mult in MULTIPLE:
												print("running-(solver-{},size-{},density-{},type-{},reorder-{},sign-{},method-{},bound-{},con-{},opt-{},m-{})"
													.format(solver,size,density,type,order,sign,method,bound,constraint, ))
												run_trials()

	start = timer() # record total time taken by running trials

	symmetry = [False]
	cons = ["sub1"]
	bounds = ["tight"]
	options = (0,5,6,7,8)

	qkp_set = ((25,70),(50,60),(75,50),(100,40))
	multiples = (1,5,10)
	signs = (False)
	for density,size in qkp_set:
		for opt in options:
			for mult in multiples:
				for con in cons:
					for bound in bounds:
						print("running-(size-{}, density-{}, type-{}, bound-{}, cons-{}, multiple-{}, symmetric-{}, options-{})".format(
								size,density,"QKP","tight",0,mult,False,opt))
						run_trials(data_=data, solver="cplex", type="QKP",symmetric=False, method="glover",
							size=size+5,den=density, multiple=mult, options=opt, glover_bounds=bound, glover_cons=con, mixed_sign=False)


	kqkp_set = ((25,70),(50,65),(75,60),(100,55))
	signs = (True, False)
	for density,size in kqkp_set:
		for opt in options:
			for con in cons:
				for bound in bounds:
					print("running-(size-{}, density-{}, type-{}, bound-{}, cons-{},symmetric-{},options-{})".format(
							size,density,"KQKP","tight",0,False,opt))
					run_trials(data_=data, solver="cplex", type="KQKP",symmetric=False, method="glover",
						size=size+5,den=density, multiple=1, options=opt, glover_bounds=bound, glover_cons=con, mixed_sign=False)


	uqp_set = ((25,55),(50,50),(75,45),(100,40))
	for density,size in uqp_set:
		for opt in options:
			print("running-(size-{}, density-{}, type-{}, bound-{}, cons-{},symmetric-{},options-{})".format(
					size,density,"UQP","tight",0,False,opt))
			run_trials(data_=data, solver="cplex", type="UQP",symmetric=False, method="glover",
				size=size+5,den=density, multiple=1, options=opt, glover_bounds="org", glover_cons=con)


	hsp_set = ((25,45),(50,35),(75,30),(100,25))
	for density,size in hsp_set:
		for opt in options:
			for bound in bounds:
				print("running-(size-{}, density-{}, type-{}, bound-{}, cons-{}, symmetric-{}, options-{})".format(
						size,density,"HSP","tight",0,False,opt))
				run_trials(data_=data, solver="cplex", type="HSP",symmetric=False, method="glover",
					size=size+5,den=density, multiple=1, options=opt, glover_bounds=bound, glover_cons="sub1")

	# qsap_set = ((12,6),(15,5),(18,4),(22,3))
	# for density,size in qsap_set:
	# 	for con in cons:
	# 		for bound in bounds:
	# 			print("running-(size-{}, density-{}, type-{}, bound-{}, cons-{}, symmetric-{})".format(
	# 					size,density,"QSAP","tight",0,False))
	# 			run_trials(data_=data, trials=num_trials, solver="cplex", type="QSAP",symmetric=False, method="qsap_glover",
	# 				size=size,den=density, multiple=1, options=0, glover_bounds=bound, glover_cons=con)


	end = timer()
	print("\n\nDONE. TOOK {:.5} SECONDS TO RUN ALL TRIALS".format(end-start))
