from quadratics import *
import cplex_solve as cplex
import gurobi_solve as gurobi
import sys
from timeit import default_timer as timer

if(len(sys.argv)==5): #batch file will go through here
	run_trials(trials=5,type=sys.argv[1],method=sys.argv[2],den=int(sys.argv[3]),size=int(sys.argv[4]))
elif __name__=="__main__":
	if False:
		start = timer()

		data = []
		for i in range(1,3):
			for j in range(3):
				dict = cplex.run_trials(trials=2, type="QKP", method="std", size=10*i, den=100-(j*10))
				data.append(dict)
				dict = cplex.run_trials(trials=2, type="QKP", method="glover", size=10*i, den=100-(j*10))
				data.append(dict)
				dict = cplex.run_trials(trials=2, type="QKP", method="glover_ext", size=10*i, den=100-(j*10))
				data.append(dict)
				dict = gurobi.run_trials(trials=2, type="QKP", method="std", size=10*i, den=100-(j*10))
				data.append(dict)
				dict = gurobi.run_trials(trials=2, type="QKP", method="glover", size=10*i, den=100-(j*10))
				data.append(dict)
				dict = gurobi.run_trials(trials=2, type="QKP", method="glover_ext", size=10*i, den=100-(j*10))
				data.append(dict)
				print("(i,j) = ("+str(i)+","+str(j)+")")

		df = pd.DataFrame(data)
		df = df[["solver", "type", "method", "size", "density", "avg_gap", "avg_solve_time", "std_dev"]]  #reorder columns
		print(df)
		#TODO put date in title so not overwritting
		writer = pd.ExcelWriter('report.xlsx', engine='xlsxwriter')
		df.to_excel(writer, index=False)
		#df_footer.to_excel(writer, startrow=6, index=False) - to put multiple dataframes to same excel file
		writer.save()

		end = timer()
		print("took " + str(end-start) + " seconds")

knap = Knapsack()
knap.print_info()

#CPLEX TESTS
m = cplex.standard_linearization(knap)[0]
m.solve()
print(m.objective_value)

m = cplex.glovers_linearization(knap)[0]
m.solve()
print(m.objective_value)

m = cplex.reformulate_glover(knap)[0]
m.solve()
print(m.objective_value)

m = cplex.glovers_linearization_ext(knap)[0]
m.solve()
print(m.objective_value)

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
