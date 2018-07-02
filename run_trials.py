from quadratics import *
import cplex_solve as cplex
import gurobi_solve as gurobi
import sys
import time
import pandas as pd
from timeit import default_timer as timer

def run_trials(trials=5,solver="cplex",type="QKP",reorder=False,symmetric=False,
			method="std",size=5,multiple=1,den=100, options=0,glover_bounds="tight", glover_cons="original",mixed_sign=False):
	"""
	Runs the same problem type thru given solver with given method repeatedly to get
	an average solve time
	"""
	#keep track of total run time across all trials to compute avg later
	setup_time_sum, solve_time_sum, int_gap_sum, obj_sum = 0,0,0,0
	#need individual run time to compute standard deviation
	instance_total_times = []

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
						m = cplex.glovers_linearization(quad, use_diagonal=False, lhs_constraints=True, bounds=glover_bounds, constraints=glover_cons)
					elif options==1:
						m = cplex.glovers_linearization(quad, use_diagonal=True, lhs_constraints=True, bounds=glover_bounds, constraints=glover_cons)
					elif options==2:
						m = cplex.glovers_linearization(quad, use_diagonal=False, lhs_constraints=False, bounds=glover_bounds, constraints=glover_cons)
					elif options==3:
						m = cplex.glovers_linearization(quad, use_diagonal=True, lhs_constraints=False, bounds=glover_bounds, constraints=glover_cons)
				elif method=="glover_rlt":
					m = cplex.glovers_linearization_rlt(quad)
				elif method=="glover_prlt":
					m = cplex.glovers_linearization_prlt(quad)
				elif method=="glover_qsap":
					m = cplex.qsap_glovers(quad, bounds=glover_bounds, constraints=glover_cons)
				else:
					raise Exception(str(method) + " is not a valid method type")
				#m[0].log_output = True
				m[0].set_time_limit(3600) #10800=3 hours #TODO need to output warning here
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
						#TODO should have constraints param in here like in cplex
						m = gurobi.glovers_linearization(quad, use_diagonal=False, lhs_constraints=True, bounds=glover_bounds)
					elif options==1:
						m = gurobi.glovers_linearization(quad, use_diagonal=True, lhs_constraints=True, bounds=glover_bounds)
					elif options==2:
						m = gurobi.glovers_linearization(quad, use_diagonal=False, lhs_constraints=False, bounds=glover_bounds)
					elif options==3:
						m = gurobi.glovers_linearization(quad, use_diagonal=True, lhs_constraints=False, bounds=glover_bounds)
				elif method=="glover_rlt":
					m = gurobi.glovers_linearization_rlt(quad)
				elif method=="glover_prlt":
					m = gurobi.glovers_linearization_prlt(quad)
				elif method=="glover_qsap":
					m = gurobi.qsap_glovers(quad, bounds=glover_bounds)
				else:
					raise Exception(str(method) + " is not a valid method type")
				results = gurobi.solve_model(m[0])
			else:
				raise Exception(str(solver) + "is not a valid solver type")

			#retrieve info from solving instance
			instance_setup_time = m[1]
			instance_solve_time = results.get("solve_time")
			instace_obj_val = results.get("objective_value")
			instace_int_gap = results.get("integrality_gap")
			instance_relax = results.get("relaxed_solution")
			instance_total_time = instance_setup_time + instance_solve_time
			#print instance solve info to log file
			f.write("Integer Solution: " + str(instace_obj_val)+"\n")
			f.write("Continuous Solution: " + str(instance_relax)+"\n")
			f.write("Integrality Gap: " + str(instace_int_gap)+"\n")
			f.write("Setup Time: " + str(instance_setup_time)+"\n")
			f.write("Solve Time: " + str(instance_solve_time)+"\n")
			f.write("Instance Total Time (Setup+Solve): " + str(instance_total_time)+"\n")
			f.write("=============================================\n")

			#update running totals across trials for computing averages
			instance_total_times.append(instance_total_time)
			setup_time_sum += instance_setup_time
			solve_time_sum += instance_solve_time
			obj_sum += instace_obj_val
			int_gap_sum += instace_int_gap
			print("trial number " + str(i) + " took " + str(instance_total_time) + " seconds to solve")

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
	type = problem type ("QKP", "KQKP", "UQP", "HSP")
	method = linearization technique ("std", "glover", "glover_rlt", "glover_prlt")
	glover_bounds = simple ub, continuous program ub, or binary ub ("original", "tight", "tighter")
	options = specify alternative/optional constraints specific to each linearization
	"""
	start = timer()
	num_trials = 10
	sizes = [3,4,5,6]
	densities = [10,15,18,20]
	solvers = ["cplex"]
	bounds = ["original","tight","tighter"]
	cons = ["original", "sub1", "sub2"]
	types = ["QSAP"]
	data = []
	for i in sizes:
		for j in densities:
			for solve_with in solvers:
				for type in types:
					for bound in bounds:
						print("current(size,den,bound) = ("+str(i)+","+str(j)+","+str(bound)+")...")
						dict = run_trials(trials=num_trials, solver=solve_with, type=type,method="glover_qsap", symmetric=False,
										glover_bounds=bound, glover_cons="original", size=i, den=j, multiple=1, options=2, reorder=False)
						data.append(dict)

						df = pd.DataFrame(data)
						df = df[["solver", "type","reorder","mixed_sign", "symmetric", "method","glover_bounds", "glover_cons", "options","size",
						 "density", "multiple", "avg_gap","avg_setup_time", "avg_solve_time", "avg_total_time", "std_dev", "avg_obj_val"]]  #reorder columns
						df.to_pickle('dataframes/glover_bounds_qsap.pkl')


	#To add everything to DF once at the end
	#df = pd.DataFrame(data)
	#df = df[["solver", "type", "symmetric", "method","glover_bounds","options", "size", "density", "avg_gap",
	#		"avg_setup_time", "avg_solve_time", "avg_total_time", "std_dev", "avg_obj_val"]]  #reorder columns
	#df.to_pickle('glove_bounds.pkl')
	print(df)

	time_stamp = time.strftime("%Y_%m_%d-%H_%M_%S")
	excel_filename = "reports/"+time_stamp+'-report.xlsx'
	writer = pd.ExcelWriter(excel_filename, engine='xlsxwriter')
	df.to_excel(writer, index=False)
	writer.save()
	end = timer()
	print("took " + str(end-start) + " seconds to run all trials")

	#testing..


# FOR BATCH FILE
# if(len(sys.argv)==5): #batch file will go through here
# run_trials(trials=5,type=sys.argv[1],method=sys.argv[2],den=int(sys.argv[3]),size=int(sys.argv[4]))
