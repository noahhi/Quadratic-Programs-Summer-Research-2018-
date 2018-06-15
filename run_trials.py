from quadratics import *
import cplex_solve as cplex
import gurobi_solve as gurobi
import sys

if(len(sys.argv)==5): #batch file will go through here
	run_trials(trials=5,type=sys.argv[1],method=sys.argv[2],den=int(sys.argv[3]),size=int(sys.argv[4]))
elif __name__=="__main__":
	if False:
		start = timer()

		data = []
		for i in range(1,3):
			for j in range(3):
				dict = run_trials(trials=2, type="QKP", method="std", size=10*i, den=100-(j*10))
				data.append(dict)
				dict = run_trials(trials=2, type="QKP", method="glover", size=10*i, den=100-(j*10))
				data.append(dict)
				dict = run_trials(trials=2, type="QKP", method="glover_ext", size=10*i, den=100-(j*10))
				data.append(dict)
				print("(i,j) = ("+str(i)+","+str(j)+")")

		df = pd.DataFrame(data)
		df = df[["type", "method", "size", "density", "avg_gap", "avg_solve_time", "std_dev"]]  #reorder columns
		print(df)

		writer = pd.ExcelWriter('simple-report.xlsx', engine='xlsxwriter')
		df.to_excel(writer, index=False)
		#df_footer.to_excel(writer, startrow=6, index=False)
		writer.save()

		end = timer()
		print("took " + str(end-start) + " seconds")

knap = Knapsack()
#knap.print_info()
# m = gurobi.standard_linearization(knap)[0]
# m.optimize();
# print(m.objVal)
# m = cplex.standard_linearization(knap)[0]
# m.solve()
# print(m.objective_value)
m = gurobi.glovers_linearization(knap)[0]
m.setParam('OutputFlag',0)
m.optimize()
print(m.objVal)
