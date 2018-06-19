from quadratics import *
import numpy as np
import pandas as pd
from timeit import default_timer as timer
from gurobipy import *

def standard_linearization(quad, con1=True, con2=True, con3=True, con4=True):
	start = timer()
	n = quad.n
	c = quad.c
	C = quad.C
	a = quad.a
	b = quad.b

	#create model and add variables
	m = Model(name='standard_linearization')
	x = m.addVars(n, name="binary_var", vtype=GRB.BINARY)
	w = m.addVars(n,n)

	if type(quad) is Knapsack: #HSP and UQP don't have cap constraint
		#add capacity constraint(s)
		for k in range(quad.m):
			m.addConstr(sum(x[i]*a[k][i] for i in range(n)) <= b[k])
		#k_item constraint(s) if necessary (if KQKP)
		for k in range(len(quad.num_items)):
			m.addConstr(sum(x[i] for i in range(n)) == quad.num_items[k])
	elif type(quad) is HSP:
		m.addConstr(sum(x[i] for i in range(n)) == quad.num_items)

	#add auxiliary constraints
	for i in range(n):
		for j in range(i+1,n):
			if(con1):
				m.addConstr(w[i,j] <= x[i])
			if(con2):
				m.addConstr(w[i,j] <= x[j])
			if(con3):
				m.addConstr(x[i]+x[j]-1 <= w[i,j])
			if(con4):
				m.addConstr(w[i,j] >= 0)

	#add objective function
	if type(quad)==HSP: #only HSP has different objective function
		quadratic_values = 0
		for i in range(n):
			for j in range(i+1,n):
				quadratic_values = quadratic_values + (w[i,j]*(C[i,j]+C[j,i]))
		m.setObjective(quadratic_values, GRB.MAXIMIZE)
	else:
		linear_values = sum(x[i]*c[i] for i in range(n))
		quadratic_values = 0
		for i in range(n):
			for j in range(i+1,n):
				quadratic_values = quadratic_values + (w[i,j]*(C[i,j]+C[j,i]))
		m.setObjective(linear_values + quadratic_values, GRB.MAXIMIZE)

	end = timer()
	setup_time = end-start
	#return model + setup time
	return [m, setup_time]

def glovers_linearization(quad, bounds="tight", constraints="original"):
	start = timer()
	n = quad.n
	c = quad.c
	C = quad.C
	a = quad.a
	b = quad.b

	#create model and add variables
	m = Model(name='glovers_linearization_'+bounds+'_'+constraints)
	#print(m.ModelSense) can set this to -1 to always max model (default is 1 = min)
	x = m.addVars(n, name="binary_var", vtype=GRB.BINARY)


	if type(quad) is Knapsack: #HSP and UQP don't have cap constraint
		#add capacity constraint(s)
		for k in range(quad.m):
			m.addConstr(sum(x[i]*a[k][i] for i in range(n)) <= b[k])
		#k_item constraint(s) if necessary
		for k in range(len(quad.num_items)):
			m.addConstr(sum(x[i] for i in range(n)) == quad.num_items[k])
	elif type(quad) is HSP:
		m.addConstr(sum(x[i] for i in range(n)) == quad.num_items)

	#determine bounds for each column of C
	U = np.zeros(n)
	L = np.zeros(n)
	if(bounds=="original"):
		for j in range(n):
			col = C[:,j]
			U[j] = np.sum(col[col>0])
			L[j] = np.sum(col[col<0])
	elif(bounds=="tight"):
		u_bound_m = Model(name='upper_bound_model')
		l_bound_m = Model(name='lower_bound_model')
		u_bound_m.setParam('OutputFlag',0)
		l_bound_m.setParam('OutputFlag',0)
		u_bound_x = u_bound_m.addVars(n, ub=1)
		l_bound_x = l_bound_m.addVars(n, ub=1)
		for k in range(quad.m):
			u_bound_m.addConstr(sum(u_bound_x[i]*a[k][i] for i in range(n)) <= b[k])
			l_bound_m.addConstr(sum(l_bound_x[i]*a[k][i] for i in range(n)) <= b[k])
		for j in range(n):
			u_bound_m.setObjective(sum(C[i,j]*u_bound_x[i] for i in range(n)), GRB.MAXIMIZE)
			l_bound_m.setObjective(sum(C[i,j]*l_bound_x[i] for i in range(n)), GRB.MINIMIZE)
			u_con = u_bound_m.addConstr(u_bound_x[j]==1)
			l_con = l_bound_m.addConstr(l_bound_x[j]==0)
			u_bound_m.optimize()
			l_bound_m.optimize()
			u_bound_m.remove(u_con)
			l_bound_m.remove(l_con)
			U[j] = u_bound_m.objVal
			L[j] = l_bound_m.objVal
	else:
		raise Exception(bounds + " is not a valid bound type for glovers")

	#add auxiliary constrains
	if(constraints=="original"):
		z = m.addVars(n,lb=-GRB.INFINITY)
		m.addConstrs(z[j] <= U[j]*x[j] for j in range(n))
		for j in range(n):
			tempsum = sum(C[i,j]*x[i] for i in range(n))
			m.addConstr(z[j] <= tempsum - L[j]*(1-x[j]))
		if type(quad) is HSP:
			m.setObjective(sum(z[j] for j in range(n)), GRB.MAXIMIZE)
		else:
			m.setObjective(sum(c[j]*x[j] + z[j] for j in range(n)), GRB.MAXIMIZE)
	elif(constraints=="sub1"):
		s = m.addVars(n)
		for j in range(n):
			tempsum = sum(C[i,j]*x[i] for i in range(n))
			m.addConstr(s[j] >= U[j]*x[j] - tempsum + L[j]*(1-x[j]))
		m.setObjective(sum(c[i]*x[i] + (U[i]*x[i]-s[i]) for i in range(n)), GRB.MAXIMIZE)
	elif(constraints=="sub2"):
		s = m.addVars(n)
		for j in range(n):
			tempsum = sum(C[i,j]*x[i] for i in range(n))
			m.addConstr(s[j] >= -U[j]*x[j] + tempsum - L[j]*(1-x[j]))
		m.setObjective(sum(c[i]*x[i] + sum(C[i,j]*x[j] for j in range(n))-L[i]*(1-x[i])-s[i] for i in range(n)), GRB.MAXIMIZE)
	else:
		raise Exception(constraints + " is not a valid constraint type for glovers")

	end = timer()
	setup_time = end-start

	#return model
	return [m,setup_time]

def glovers_linearization_prlt(quad):
	def prlt1_linearization(quad): #only called from within reformulate_glover (make inner func?)
		n = quad.n
		c = quad.c
		C = quad.C
		a = quad.a
		b = quad.b

		#create model and add variables
		m = Model(name='PRLT-1_linearization')
		x = m.addVars(n, lb=0, ub=1)
		w = m.addVars(n,n)

		#add capacity constraint
		for k in range(quad.m):
			m.addConstr(sum(x[i]*a[k][i] for i in range(n)) <= b[k])

		#add auxiliary constraints
		for i in range(n):
			for j in range(i+1,n):
				m.addConstr(w[i,j]==w[j,i], name='con16'+str(i)+str(j))

		for k in range(quad.m):
			for j in range(n):
				m.addConstr(sum(a[k][i]*w[i,j] for i in range(n) if i!=j)<=(b[k]-a[k][j])*x[j])
				for i in range(n):
					m.addConstr(w[i,j] <= x[j])

		#add objective function
		linear_values = sum(x[j]*c[j] for j in range(n))
		quadratic_values = 0
		for j in range(n):
			for i in range(n):
				if(i==j):
					continue
				quadratic_values = quadratic_values + (C[i,j]*w[i,j])
		m.setObjective(linear_values + quadratic_values, GRB.MAXIMIZE)

		#return model
		return m
	n=quad.n
	start = timer()
	m = prlt1_linearization(quad)
	m.setParam('OutputFlag',0)
	m.optimize()
	duals16 = np.zeros((n,n))
	for i in range(n):
		for j in range(i+1,n):
			con_name = 'con16'+str(i)+str(j)
			duals16[i][j]=(m.getConstrByName(con_name).getAttr("Pi"))
	C = quad.C
	for i in range(quad.n):
		for j in range(i+1,quad.n):
			duals16[j,i]=C[j,i]+duals16[i,j]
			duals16[i,j]=C[i,j]-duals16[i,j]
	quad.C = duals16
	new_m = glovers_linearization(quad, bounds="tight", constraints="original")[0]
	end = timer()
	setup_time = end-start
	return [new_m, setup_time]

def glovers_linearization_rlt(quad, bounds="tight", constraints="original"):
	def rlt1_linearization(quad):
		n = quad.n
		c = quad.c
		C = quad.C
		a = quad.a
		b = quad.b

		#create model and add variables
		m = Model(name='RLT-1_linearization')
		#default var type is continuous in gurobi
		x = m.addVars(n,name='binary_var', lb=0, ub=1, vtype=GRB.CONTINUOUS) #named binary_var so can easily switch for debug
		w = m.addVars(n,n)
		y = m.addVars(n,n)

		#add capacity constraint(s)
		for k in range(quad.m):
			m.addConstr(sum(x[i]*a[k][i] for i in range(n)) <= b[k])

		#add auxiliary constraints
		for i in range(n):
			for j in range(i+1,n):
				#con 16
				m.addConstr(w[i,j]==w[j,i], name='con16'+str(i)+str(j))

		for k in range(quad.m):
			for j in range(n):
				#con 12
				m.addConstr(sum(a[k][i]*w[i,j] for i in range(n) if i!=j)<=(b[k]-a[k][j])*x[j])
				#con 14
				m.addConstr(sum(a[k][i]*y[i,j] for i in range(n) if i!=j)<=b[k]*(1-x[j]))

		for j in range(n):
			for i in range(n):
				if(i==j):
					continue
				#con 13 (w>=0 implied) - imp to add anyways?
				m.addConstr(w[i,j] <= x[j])
				#con 15 (y>=0 implied)
				m.addConstr(y[i,j] <= 1-x[j])
				#con 17
				m.addConstr(y[i,j] == x[i]-w[i,j], name='con17'+str(i)+str(j))

		#add objective function
		linear_values = sum(x[j]*c[j] for j in range(n))
		quadratic_values = 0
		for j in range(n):
			for i in range(n):
				if(i==j):
					continue
				quadratic_values = quadratic_values + (C[i,j]*w[i,j])
		m.setObjective(linear_values + quadratic_values, GRB.MAXIMIZE)

		#return model
		return m

	start = timer()
	n = quad.n
	c = quad.c
	C = quad.C
	a = quad.a
	b = quad.b

	#model with continuous relaxed rlt1 and get duals to constraints 16,17
	m = rlt1_linearization(quad)
	m.setParam('OutputFlag',0)
	m.optimize()
	#print(m.objVal)     #this should be continuous relax solution to glover_ext.
	#retrieve dual variables
	duals16 = np.zeros((n,n))
	duals17 = np.zeros((n,n))
	for i in range(n):
		for j in range(i+1,n):
			con_name = 'con16'+str(i)+str(j)
			duals16[i][j]=(m.getConstrByName(con_name).getAttr("Pi"))
		for j in range(n):
			if i==j:
				continue
			con_name = 'con17'+str(i)+str(j)
			duals17[i][j]=(m.getConstrByName(con_name).getAttr("Pi"))
	D = np.zeros((n,n))
	E = np.zeros((n,n))
	#optimal split, found using dual vars from rlt1 continuous relaxation
	for i in range(n):
		for j in range(n):
			if i==j:
				continue
			if i<j:
				D[i,j] = C[i,j]-duals16[i,j]-duals17[i,j]
			if i>j:
				D[i,j] = C[i,j]+duals16[j,i]-duals17[i,j]
	E = -duals17

	#update linear values as well
	for j in range(n):
		c[j] = c[j] + sum(duals17[j,i] for i in range(n))

	#simple split (this works but is not optimal)
	# for i in range(n):
		# for j in range(i+1,n):
			# D[i,j] = C[i,j]/4
			# D[j,i] = C[i,j]/4
			# E[i,j] = C[i,j]/4
			# E[j,i] = C[i,j]/4

	#create model and add variables
	m = Model(name='glovers_linearization_ext_'+bounds+'_'+constraints)
	x = m.addVars(n, name="binary_var", vtype=GRB.BINARY)

	#add capacity constraint(s)
	for k in range(quad.m):
		m.addConstr(sum(x[i]*a[k][i] for i in range(n)) <= b[k])

	#determine bounds for each column of C
	U1 = np.zeros(n)
	L1 = np.zeros(n)
	U2 = np.zeros(n)
	L2 = np.zeros(n)
	if(bounds=="original"):
		for j in range(n):
			col1 = D[:,j]
			col2 = E[:,j]
			U1[j] = np.sum(col1[col1>0])
			L1[j] = np.sum(col1[col1<0])
			U2[j] = np.sum(col2[col2>0])
			L2[j] = np.sum(col2[col2<0])

	elif(bounds=="tight"):
		u_bound_m1 = Model(name='upper_bound_model1')
		l_bound_m1 = Model(name='lower_bound_model1')
		u_bound_m1.setParam('OutputFlag',0)
		l_bound_m1.setParam('OutputFlag',0)
		u_bound_x1 = u_bound_m1.addVars(n, ub=1)
		l_bound_x1 = l_bound_m1.addVars(n, ub=1)
		for k in range(quad.m):
			u_bound_m1.addConstr(sum(u_bound_x1[i]*a[k][i] for i in range(n)) <= b[k])
			l_bound_m1.addConstr(sum(l_bound_x1[i]*a[k][i] for i in range(n)) <= b[k])
		u_bound_m2 = Model(name='upper_bound_model2')
		l_bound_m2 = Model(name='lower_bound_model2')
		u_bound_m2.setParam('OutputFlag',0)
		l_bound_m2.setParam('OutputFlag',0)
		u_bound_x2 = u_bound_m2.addVars(n, ub=1)
		l_bound_x2 = l_bound_m2.addVars(n, ub=1)
		for k in range(quad.m):
			u_bound_m2.addConstr(sum(u_bound_x2[i]*a[k][i] for i in range(n)) <= b[k])
			l_bound_m2.addConstr(sum(l_bound_x2[i]*a[k][i] for i in range(n)) <= b[k])

		for j in range(n):
			u_bound_m1.setObjective(sum(D[i,j]*u_bound_x1[i] for i in range(n) if i!=j), GRB.MAXIMIZE)
			l_bound_m1.setObjective(sum(D[i,j]*l_bound_x1[i] for i in range(n) if i!=j), GRB.MINIMIZE)
			u_con1 = u_bound_m1.addConstr(u_bound_x1[j]==1)
			l_con1 = l_bound_m1.addConstr(l_bound_x1[j]==0)
			u_bound_m1.optimize()
			l_bound_m1.optimize()
			u_bound_m1.remove(u_con1)
			l_bound_m1.remove(l_con1)
			U1[j] = u_bound_m1.objVal
			L1[j] = l_bound_m1.objVal

			u_bound_m2.setObjective(sum(E[i,j]*u_bound_x2[i] for i in range(n) if i!=j), GRB.MAXIMIZE)
			l_bound_m2.setObjective(sum(E[i,j]*l_bound_x2[i] for i in range(n) if i!=j), GRB.MINIMIZE)
			u_con2 = u_bound_m2.addConstr(u_bound_x2[j]==0)
			l_con2 = l_bound_m2.addConstr(l_bound_x2[j]==1)
			u_bound_m2.optimize()
			l_bound_m2.optimize()
			u_bound_m2.remove(u_con2)
			l_bound_m2.remove(l_con2)
			U2[j] = u_bound_m2.objVal
			L2[j] = l_bound_m2.objVal
	else:
		raise Exception(bounds + " is not a valid bound type for glovers")

	#add auxiliary constrains
	if(constraints=="original"):
		z1 = m.addVars(n,lb=-GRB.INFINITY)
		z2 = m.addVars(n,lb=-GRB.INFINITY)
		for j in range(n):
			m.addConstr(z1[j] <= U1[j]*x[j])
			m.addConstr(z2[j] <= U2[j]*(1-x[j]))
		for j in range(n):
			tempsum1 = sum(D[i,j]*x[i] for i in range(n) if i!=j)
			m.addConstr(z1[j] <= tempsum1 - L1[j]*(1-x[j]))
			tempsum2 = sum(E[i,j]*x[i] for i in range(n) if i!=j)
			m.addConstr(z2[j] <= tempsum2 - (L2[j]*x[j]))
		m.setObjective(sum(c[j]*x[j] + z1[j] + z2[j] for j in range(n)), GRB.MAXIMIZE)
	#substituted constraints not yet implemented here
	# elif(constraints=="sub1"):
		# s = m.addVars(n)
		# for j in range(n):
			# tempsum = sum(C[i,j]*x[i] for i in range(n))
			# m.addConstr(s[j] >= U[j]*x[j] - tempsum + L[j]*(1-x[j]))
		# m.setObjective(sum(c[i]*x[i] + (U[i]*x[i]-s[i]) for i in range(n)), GRB.MAXIMIZE)
	# elif(constraints=="sub2"):
		# s = m.addVars(n)
		# for j in range(n):
			# tempsum = sum(C[i,j]*x[i] for i in range(n))
			# m.addConstr(s[j] >= -U[j]*x[j] + tempsum - L[j]*(1-x[j]))
		# m.setObjective(sum(c[i]*x[i] + sum(C[i,j]*x[j] for j in range(n))-L[i]*(1-x[i])-s[i] for i in range(n)), GRB.MAXIMIZE)
	else:
		raise Exception(constraints + " is not a valid constraint type for glovers")

	end = timer()
	setup_time = end-start

	#return model
	return [m,setup_time]

def solve_model(m):
	"""
	Takes in an unsolved gurobi model of a MIP. Solves it as well as continuous
	relaxation and returns a dictionary containing relevant solve details
	"""
	#turn off model output. otherwise prints bunch of info, clogs console
	m.setParam('OutputFlag',0)

	#start timer and solve model
	start = timer()
	m.optimize()
	end = timer()
	solve_time = end-start
	objective_value = m.objVal

	#relax and solve to get continuous relaxation and integrality_gap
	r = m.relax()
	r.optimize()
	continuous_obj_value = r.objVal
	integrality_gap=((continuous_obj_value-objective_value)/objective_value)*100
	#TODO double check this relax is same as cplex relax
	#terminate model so not allocating resources
	m.terminate()

	#create and return results dictionary
	results = {"solve_time":solve_time,
				"objective_value":objective_value,
				"relaxed_solution":continuous_obj_value,
				"integrality_gap":integrality_gap}
	return results

# knap = Knapsack(n=40)
# m = glovers_linearization_ext(knap)[0]
# r = solve_model(m)
# print(r.get("objective_value"))
# print(r.get("relaxed_solution"))
# print(r.get("integrality_gap"))
