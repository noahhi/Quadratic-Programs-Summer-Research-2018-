from quadratics import *
import cplex_solve as cplex
import gurobi_solve as gurobi
import sys
import time
import pandas as pd
from timeit import default_timer as timer

if(len(sys.argv)==5): #batch file will go through here
	run_trials(trials=5,type=sys.argv[1],method=sys.argv[2],den=int(sys.argv[3]),size=int(sys.argv[4]))
elif __name__=="__main__":
	if True:
		start = timer()
		num_trials = 2
		data = []
		for i in range(4,5): #i*10 gives sizes
			for j in range(1): #100-(j*10) gives densities
				dict = cplex.run_trials(trials=num_trials, type="QKP", method="std", size=10*i, den=100-(j*10))
				data.append(dict)
				dict = cplex.run_trials(trials=num_trials, type="QKP", method="glover", size=10*i, den=100-(j*10))
				data.append(dict)
				dict = cplex.run_trials(trials=num_trials, type="QKP", method="glover_ext", size=10*i, den=100-(j*10))
				data.append(dict)
				dict = gurobi.run_trials(trials=num_trials, type="QKP", method="std", size=10*i, den=100-(j*10))
				data.append(dict)
				dict = gurobi.run_trials(trials=num_trials, type="QKP", method="glover", size=10*i, den=100-(j*10))
				data.append(dict)
				dict = gurobi.run_trials(trials=num_trials, type="QKP", method="glover_ext", size=10*i, den=100-(j*10))
				data.append(dict)
				dict = gurobi.run_trials(trials=num_trials, type="QKP", method="prlt", size=10*i, den=100-(j*10))
				data.append(dict)
				print("(i,j) = ("+str(i)+","+str(j)+")")

		df = pd.DataFrame(data)
		df = df[["solver", "type", "method", "size", "density", "avg_gap", "avg_solve_time", "std_dev", "avg_obj_val"]]  #reorder columns
		print(df)

		time_stamp = time.strftime("%Y_%m_%d-%H_%M_%S")
		excel_filename = "reports/"+time_stamp+'-report.xlsx'
		writer = pd.ExcelWriter(excel_filename, engine='xlsxwriter')
		df.to_excel(writer, index=False)
		#df_footer.to_excel(writer, startrow=6, index=False) - to put multiple dataframes to same excel file
		writer.save()

		end = timer()
		print("took " + str(end-start) + " seconds")

if False:
	knap = Knapsack()
	knap.print_info()

	#CPLEX TESTS
	# m = cplex.standard_linearization(knap)[0]
	# m.solve()
	# print(m.objective_value)
	#
	# m = cplex.glovers_linearization(knap)[0]
	# m.solve()
	# print(m.objective_value)
	#
	# m = cplex.reformulate_glover(knap)[0]
	# m.solve()
	# print(m.objective_value)
	#
	# m = cplex.glovers_linearization_ext(knap)[0]
	# m.solve()
	# print(m.objective_value)

	#GUROBI TESTS
	m = gurobi.standard_linearization(knap)[0]
	m.setParam('OutputFlag',0)
	m.optimize();
	print(m.objVal)

	m = gurobi.glovers_linearization(knap)[0]
	m.setParam('OutputFlag',0)
	m.optimize()
	print(m.objVal)

	m = gurobi.reformulate_glover(knap)[0]
	m.setParam('OutputFlag',0)
	m.optimize()
	print(m.objVal)

	m = gurobi.glovers_linearization_ext(knap)[0]
	m.setParam('OutputFlag',0)
	m.optimize()
	print(m.objVal)
