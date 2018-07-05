from quadratics import *
import numpy as np
from timeit import default_timer as timer
from gurobipy import *

# turn off model output. otherwise prints bunch of info, clogs console
setParam('OutputFlag',0)

def standard_linearization(quad, con1=True, con2=True, con3=True, con4=True):
	start = timer()
	n = quad.n
	c = quad.c
	C = quad.C
	a = quad.a
	b = quad.b

	# create model and add variables
	m = Model(name='standard_linearization')
	x = m.addVars(n, name="binary_var", vtype=GRB.BINARY)
	w = m.addVars(n, n)

	if type(quad) is Knapsack:  # HSP and UQP don't have cap constraint
		# add capacity constraint(s)
		for k in range(quad.m):
			m.addConstr(quicksum(x[i]*a[k][i] for i in range(n)) <= b[k])

	#k_item constraint if necessary (if KQKP or HSP)
	if quad.num_items > 0:
		m.addConstr(quicksum(x[i] for i in range(n)) == quad.num_items)

	# add auxiliary constraints
	for i in range(n):
		for j in range(i+1, n):
			if(con1):
				m.addConstr(w[i, j] <= x[i])
			if(con2):
				m.addConstr(w[i, j] <= x[j])
			if(con3):
				m.addConstr(x[i]+x[j]-1 <= w[i, j])
			if(con4):
				m.addConstr(w[i, j] >= 0)

	#compute quadratic values contirbution to obj
	quadratic_values = 0
	for i in range(n):
		for j in range(i+1, n):
			quadratic_values = quadratic_values + (w[i, j]*(C[i, j]+C[j, i]))
	# set objective function
	if type(quad) == HSP:
		#HSP doesn't habe any linear terms
		m.setObjective(quadratic_values, GRB.MAXIMIZE)
	else:
		linear_values = sum(x[i]*c[i] for i in range(n))
		m.setObjective(linear_values + quadratic_values, GRB.MAXIMIZE)

	end = timer()
	setup_time = end-start
	# return model + setup time
	return [m, setup_time]

def glovers_linearization(quad, bounds="tight", constraints="original", lhs_constraints=False, use_diagonal=False):
	start = timer()
	n = quad.n
	c = quad.c
	C = quad.C
	a = quad.a
	b = quad.b

	#put linear terms along diagonal of quadratic matrix. set linear terms to zero
	if use_diagonal:
		for i in range(n):
			C[i,i] = c[i]
			c[i] = 0


	# create model and add variables
	m = Model(name='glovers_linearization_'+bounds+'_'+constraints)
	# print(m.ModelSense) can set this to -1 to always max model (default is 1 = min)
	x = m.addVars(n, name="binary_var", vtype=GRB.BINARY)

	if type(quad) is Knapsack:  # HSP and UQP don't have cap constraint
		# add capacity constraint(s)
		for k in range(quad.m):
			m.addConstr(quicksum(x[i]*a[k][i] for i in range(n)) <= b[k])

	#k_item constraint if necessary (if KQKP or HSP)
	if quad.num_items > 0:
		m.addConstr(quicksum(x[i] for i in range(n)) == quad.num_items)

	# determine bounds for each column of C
	#U1,L1 must take item at index j, U0,L0 must not take
	U1 = np.zeros(n)
	L0 = np.zeros(n)
	U0 = np.zeros(n)
	L1 = np.zeros(n)
	if(bounds == "original"):
		for j in range(n):
			col = C[:, j]
			pos_take_vals = col > 0
			pos_take_vals[j] = True
			U1[j] = np.sum(col[pos_take_vals])
			neg_take_vals = col < 0
			neg_take_vals[j] = False
			L0[j] = np.sum(col[neg_take_vals])
			if lhs_constraints:
				U0[j] = U1[j] - col[j]
				L1[j] = L0[j] + col[j]
	elif(bounds == "tight" or bounds=="tighter"):
		u_bound_m = Model(name='upper_bound_model')
		l_bound_m = Model(name='lower_bound_model')
		if bounds=="tight":
			u_bound_x = u_bound_m.addVars(n, ub=1, lb=0, vtype=GRB.CONTINUOUS)
			l_bound_x = l_bound_m.addVars(n, ub=1, lb=0, vtype=GRB.CONTINUOUS)
		elif bounds=="tighter":
			u_bound_x = u_bound_m.addVars(n, ub=1, lb=0, vtype=GRB.BINARY)
			l_bound_x = l_bound_m.addVars(n, ub=1, lb=0, vtype=GRB.BINARY)
		if type(quad) is Knapsack:
			for k in range(quad.m):
				#add capacity constraints
				u_bound_m.addConstr(quicksum(u_bound_x[i]*a[k][i] for i in range(n)) <= b[k])
				l_bound_m.addConstr(quicksum(l_bound_x[i]*a[k][i] for i in range(n)) <= b[k])
		if quad.num_items > 0:
			u_bound_m.addConstr(quicksum(u_bound_x[i] for i in range(n)) == quad.num_items)
			l_bound_m.addConstr(quicksum(l_bound_x[i] for i in range(n)) == quad.num_items)
		for j in range(n):
			#for each col, solve model to find upper/lower bound
			u_bound_m.setObjective(quicksum(C[i, j]*u_bound_x[i] for i in range(n)), GRB.MAXIMIZE)
			l_bound_m.setObjective(quicksum(C[i, j]*l_bound_x[i] for i in range(n)), GRB.MINIMIZE)
			u_con = u_bound_m.addConstr(u_bound_x[j] == 1)
			l_con = l_bound_m.addConstr(l_bound_x[j] == 0)
			u_bound_m.optimize()
			#if GRB.OPTIMAL!=2: #2 means optimal, 3 infeasible. see gurobi docs for status_codes
			#	print(GRB.OPTIMAL)
			if u_bound_m.status != GRB.Status.OPTIMAL:
				print("non-optimal solve status: " + str(u_bound_m.status) + " when solving for upper bound (U1)")
				u_bound_m.remove(u_con)
				u_bound_m.addConstr(u_bound_x[j]==0)
				m.addConstr(x[j]==0)
				u_bound_m.optimize()
			l_bound_m.optimize()
			if l_bound_m.status != GRB.Status.OPTIMAL:
				print("non-optimal solve status: " + str(l_bound_m.status) + " when solving for lower bound (L0)")
				l_bound_m.remove(l_con)
				l_bound_m.addConstr(l_bound_x[j]==1)
				m.addConstr(x[j]==1)
				l_bound_m.optimize()
			U1[j] = u_bound_m.objVal
			L0[j] = l_bound_m.objVal
			u_bound_m.remove(u_con)
			l_bound_m.remove(l_con)
			if lhs_constraints:
				u_con = u_bound_m.addConstr(u_bound_x[j] == 0)
				l_con = l_bound_m.addConstr(l_bound_x[j] == 1)
				u_bound_m.optimize()
				if u_bound_m.status != GRB.Status.OPTIMAL:
					print("non-optimal solve status: " + str(u_bound_m.status) + " when solving for upper bound (U0)")
					u_bound_m.remove(u_con)
					u_bound_m.addConstr(u_bound_x[j]==1)
					m.addConstr(x[j]==1)
					u_bound_m.optimize()
				l_bound_m.optimize()
				if l_bound_m.status != GRB.Status.OPTIMAL:
					print("non-optimal solve status: " + str(l_bound_m.status) + " when solving for lower bound (L1)")
					l_bound_m.remove(l_con)
					l_bound_m.addConstr(l_bound_x[j]==0)
					m.addConstr(x[j]==0)
					l_bound_m.optimize()
				U0[j] = u_bound_m.objVal
				L1[j] = l_bound_m.objVal
				u_bound_m.remove(u_con)
				l_bound_m.remove(l_con)
	else:
		raise Exception(bounds + " is not a valid bound type for glovers")

	# add auxiliary constrains
	if(constraints == "original"):
		#original glovers constraints
		z = m.addVars(n, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS)
		m.addConstrs(z[j] <= U1[j]*x[j] for j in range(n))
		if lhs_constraints:
			m.addConstrs(z[j] >= L1[j]*x[j] for j in range(n))
		for j in range(n):
			tempsum = quicksum(C[i, j]*x[i] for i in range(n))
			m.addConstr(z[j] <= tempsum - L0[j]*(1-x[j]))
			if lhs_constraints:
				m.addConstr(z[j] >= tempsum - U0[j]*(1-x[j]))
		if type(quad) is HSP:
			m.setObjective(quicksum(z[j] for j in range(n)), GRB.MAXIMIZE)
		else:
			m.setObjective(quicksum(c[j]*x[j] + z[j] for j in range(n)), GRB.MAXIMIZE)


	elif(constraints=="sub1" or constraints=="sub2"):
		#can make one of 2 substitutions using slack variables to further reduce # of constraints
		s = m.addVars(n, vtype=GRB.CONTINUOUS)
		for j in range(n):
			tempsum = quicksum(C[i, j]*x[i] for i in range(n))
			if constraints=="sub1":
				m.addConstr(s[j] >= U[j]*x[j] - tempsum + L[j]*(1-x[j]))
			else:
				m.addConstr(s[j] >= -U[j]*x[j] + tempsum - L[j]*(1-x[j]))
		if constraints=="sub1":
			m.setObjective(quicksum(c[i]*x[i] + (U[i]*x[i]-s[i]) for i in range(n)), GRB.MAXIMIZE)
		else:
			m.setObjective(quicksum(c[i]*x[i] + quicksum(C[i, j]*x[j] for j in range(n))-L[i]*(1-x[i])-s[i] for i in range(n)), GRB.MAXIMIZE)
	else:
		raise Exception(constraints + " is not a valid constraint type for glovers")

	end = timer()
	setup_time = end-start
	# return model
	return [m, setup_time]

def glovers_linearization_prlt(quad):
	def prlt1_linearization(quad):  # only called from within reformulate_glover (make inner func?)
		n = quad.n
		c = quad.c
		C = quad.C
		a = quad.a
		b = quad.b

		# create model and add variables
		m = Model(name='PRLT-1_linearization')
		x = m.addVars(n, lb=0, ub=1, vtype=GRB.CONTINUOUS)
		w = m.addVars(n, n, vtype=GRB.CONTINUOUS)

		if type(quad) is Knapsack:  # HSP and UQP don't have cap constraint
			# add capacity constraint(s)
			for k in range(quad.m):
				m.addConstr(quicksum(x[i]*a[k][i] for i in range(n)) <= b[k])

		#k_item constraint if necessary (if KQKP or HSP)
		if quad.num_items > 0:
			m.addConstr(quicksum(x[i] for i in range(n)) == quad.num_items)

		# add auxiliary constraints
		for i in range(n):
			for j in range(i+1, n):
				m.addConstr(w[i, j] == w[j, i], name='con16'+str(i)+str(j))

		for k in range(quad.m):
			for j in range(n):
				m.addConstr(quicksum(a[k][i]*w[i, j] for i in range(n) if i != j) <= (b[k]-a[k][j])*x[j])
				for i in range(n):
					m.addConstr(w[i, j] <= x[j])

		# add objective function

		quadratic_values = 0
		for j in range(n):
			for i in range(n):
				if(i == j):
					continue
				quadratic_values = quadratic_values + (C[i, j]*w[i, j])
		if type(quad) is HSP:
			m.setObjective(quadratic_values, GRB.MAXIMIZE)
		else:
			m.setObjective(linear_values + quadratic_values, GRB.MAXIMIZE)
			linear_values = quicksum(x[j]*c[j] for j in range(n))

		# return model
		return m

	start = timer()
	m = prlt1_linearization(quad)
	m.optimize()

	n = quad.n
	C = quad.C
	duals16 = np.zeros((n, n))
	for i in range(n):
		for j in range(i+1, n):
			con_name = 'con16'+str(i)+str(j)
			dual = (m.getConstrByName(con_name).getAttr("Pi"))
			#makes sure there are no negative values in resulting quadratic matrix
			if dual > C[i][j]:
				dual = C[i][j]
			elif dual < -C[j][i]:
				dual = -C[j][i]
			duals16[i][j] = dual
	for i in range(quad.n):
		for j in range(i+1, quad.n):
			#modify quadratic coefficient matrix using duals
			duals16[j, i] = C[j, i]+duals16[i, j]
			duals16[i, j] = C[i, j]-duals16[i, j]
	quad.C = duals16
	new_m = glovers_linearization(quad, bounds="tight", constraints="original")[0]
	end = timer()
	setup_time = end-start
	return [new_m, setup_time]

#TODO glovers_rlt currently works only for knapsack w/ original constraints
def glovers_linearization_rlt(quad, bounds="tight", constraints="original"):
	def rlt1_linearization(quad):
		n = quad.n
		c = quad.c
		C = quad.C
		a = quad.a
		b = quad.b

		# create model and add variables
		m = Model(name='RLT-1_linearization')
		x = m.addVars(n, name='binary_var', lb=0, ub=1, vtype=GRB.CONTINUOUS) # named binary_var so can easily switch for debug
		w = m.addVars(n, n, vtype=GRB.CONTINUOUS)
		y = m.addVars(n, n, vtype=GRB.CONTINUOUS)

		if type(quad) is Knapsack:  # HSP and UQP don't have cap constraint
			# add capacity constraint(s)
			for k in range(quad.m):
				m.addConstr(quicksum(x[i]*a[k][i] for i in range(n)) <= b[k])

		#k_item constraint if necessary (if KQKP or HSP)
		if quad.num_items > 0:
			m.addConstr(quicksum(x[i] for i in range(n)) == quad.num_items)

		# add auxiliary constraints
		for i in range(n):
			for j in range(i+1, n):
				# con 16
				m.addConstr(w[i, j] == w[j, i], name='con16'+str(i)+str(j))

		for k in range(quad.m):
			for j in range(n):
				# con 12
				m.addConstr(quicksum(a[k][i]*w[i, j] for i in range(n) if i != j) <= (b[k]-a[k][j])*x[j])
				# con 14
				m.addConstr(quicksum(a[k][i]*y[i, j] for i in range(n) if i != j) <= b[k]*(1-x[j]))

		for j in range(n):
			for i in range(n):
				if(i == j):
					continue
				# con 13 (w>=0 implied) - imp to add anyways?
				m.addConstr(w[i, j] <= x[j])
				# con 15 (y>=0 implied)
				m.addConstr(y[i, j] <= 1-x[j])
				# con 17
				m.addConstr(y[i, j] == x[i]-w[i, j], name='con17'+str(i)+str(j))

		quadratic_values = 0
		for j in range(n):
			for i in range(n):
				if(i == j):
					continue
				quadratic_values = quadratic_values + (C[i, j]*w[i, j])
		if type(quad) is HSP:
			m.setObjective(quadratic_values, GRB.MAXIMIZE)
		else:
			m.setObjective(linear_values + quadratic_values, GRB.MAXIMIZE)
			linear_values = quicksum(x[j]*c[j] for j in range(n))

		# return model
		return m

	start = timer()
	n = quad.n
	c = quad.c
	C = quad.C
	a = quad.a
	b = quad.b

	# model with continuous relaxed rlt1 and get duals to constraints 16,17
	m = rlt1_linearization(quad)
	m.optimize()
	#print(m.objVal)     #this should be continuous relax solution to glover_ext.
	# retrieve dual variables
	duals16 = np.zeros((n, n))
	duals17 = np.zeros((n, n))
	for i in range(n):
		for j in range(i+1, n):
			con_name = 'con16'+str(i)+str(j)
			duals16[i][j] = (m.getConstrByName(con_name).getAttr(GRB.attr.Pi))
		for j in range(n):
			if i == j:
				continue
			con_name = 'con17'+str(i)+str(j)
			duals17[i][j] = (m.getConstrByName(con_name).getAttr(GRB.attr.Pi))

	D = np.zeros((n, n))
	E = np.zeros((n, n))
	# optimal split, found using dual vars from rlt1 continuous relaxation
	for i in range(n):
		for j in range(n):
			if i == j:
				continue
			if i < j:
				D[i, j] = C[i, j]-duals16[i, j]-duals17[i, j]
			if i > j:
				D[i, j] = C[i, j]+duals16[j, i]-duals17[i, j]
	E = -duals17

	# update linear values as well
	for j in range(n):
		c[j] = c[j] + quicksum(duals17[j, i] for i in range(n))

	# simple split (this works but is not optimal)
	# for i in range(n):
		# for j in range(i+1,n):
		# D[i,j] = C[i,j]/4
		# D[j,i] = C[i,j]/4
		# E[i,j] = C[i,j]/4
		# E[j,i] = C[i,j]/4

	# create model and add variables
	m = Model(name='glovers_linearization_ext_'+bounds+'_'+constraints)
	x = m.addVars(n, name="binary_var", vtype=GRB.BINARY)

	if type(quad) is Knapsack:  # HSP and UQP don't have cap constraint
		# add capacity constraint(s)
		for k in range(quad.m):
			m.addConstr(quicksum(x[i]*a[k][i] for i in range(n)) <= b[k])

	#k_item constraint if necessary (if KQKP or HSP)
	if quad.num_items > 0:
		m.addConstr(quicksum(x[i] for i in range(n)) == quad.num_items)

	# determine bounds for each column of C
	U1 = np.zeros(n)
	L1 = np.zeros(n)
	U2 = np.zeros(n)
	L2 = np.zeros(n)
	if(bounds == "original"):
		for j in range(n):
			col1 = D[:, j]
			col2 = E[:, j]
			U1[j] = np.sum(col1[col1 > 0])
			L1[j] = np.sum(col1[col1 < 0])
			U2[j] = np.sum(col2[col2 > 0])
			L2[j] = np.sum(col2[col2 < 0])

	elif(bounds == "tight"):
		u_bound_m1 = Model(name='upper_bound_model1')
		l_bound_m1 = Model(name='lower_bound_model1')
		u_bound_x1 = u_bound_m1.addVars(n, ub=1, lb=0.0)
		l_bound_x1 = l_bound_m1.addVars(n, ub=1, lb=0.0)
		for k in range(quad.m):
			u_bound_m1.addConstr(quicksum(u_bound_x1[i]*a[k][i] for i in range(n)) <= b[k])
			l_bound_m1.addConstr(quicksum(l_bound_x1[i]*a[k][i] for i in range(n)) <= b[k])
		u_bound_m2 = Model(name='upper_bound_model2')
		l_bound_m2 = Model(name='lower_bound_model2')
		u_bound_x2 = u_bound_m2.addVars(n, ub=1, lb=0)
		l_bound_x2 = l_bound_m2.addVars(n, ub=1, lb=0)
		for k in range(quad.m):
			u_bound_m2.addConstr(quicksum(u_bound_x2[i]*a[k][i] for i in range(n)) <= b[k])
			l_bound_m2.addConstr(quicksum(l_bound_x2[i]*a[k][i] for i in range(n)) <= b[k])

		for j in range(n):
			u_bound_m1.setObjective(quicksum(D[i, j]*u_bound_x1[i] for i in range(n) if i != j), GRB.MAXIMIZE)
			l_bound_m1.setObjective(quicksum(D[i, j]*l_bound_x1[i] for i in range(n) if i != j), GRB.MINIMIZE)
			u_con1 = u_bound_m1.addConstr(u_bound_x1[j] == 1)
			l_con1 = l_bound_m1.addConstr(l_bound_x1[j] == 0)
			u_bound_m1.optimize()
			l_bound_m1.optimize()
			u_bound_m1.remove(u_con1)
			l_bound_m1.remove(l_con1)
			U1[j] = u_bound_m1.objVal
			L1[j] = l_bound_m1.objVal

			u_bound_m2.setObjective(quicksum(E[i, j]*u_bound_x2[i] for i in range(n) if i != j), GRB.MAXIMIZE)
			l_bound_m2.setObjective(quicksum(E[i, j]*l_bound_x2[i] for i in range(n) if i != j), GRB.MINIMIZE)
			u_con2 = u_bound_m2.addConstr(u_bound_x2[j] == 0)
			l_con2 = l_bound_m2.addConstr(l_bound_x2[j] == 1)
			u_bound_m2.optimize()
			l_bound_m2.optimize()
			u_bound_m2.remove(u_con2)
			l_bound_m2.remove(l_con2)
			U2[j] = u_bound_m2.objVal
			L2[j] = l_bound_m2.objVal
	else:
		raise Exception(bounds + " is not a valid bound type for glovers")

	# add auxiliary constrains
	if(constraints == "original"):
		z1 = m.addVars(n, lb=-GRB.INFINITY)
		z2 = m.addVars(n, lb=-GRB.INFINITY)
		for j in range(n):
			m.addConstr(z1[j] <= U1[j]*x[j])
			m.addConstr(z2[j] <= U2[j]*(1-x[j]))
		for j in range(n):
			tempsum1 = quicksum(D[i, j]*x[i] for i in range(n) if i != j)
			m.addConstr(z1[j] <= tempsum1 - L1[j]*(1-x[j]))
			tempsum2 = quicksum(E[i, j]*x[i] for i in range(n) if i != j)
			m.addConstr(z2[j] <= tempsum2 - (L2[j]*x[j]))
		m.setObjective(quicksum(c[j]*x[j] + z1[j] + z2[j] for j in range(n)), GRB.MAXIMIZE)
	else:
		raise Exception(constraints + " is not a valid constraint type for glovers")

	end = timer()
	setup_time = end-start

	# return model
	return [m, setup_time]

def solve_model(m, solve_relax=True):
	"""
	Takes in an unsolved gurobi model of a MIP. Solves it as well as continuous
	relaxation and returns a dictionary containing relevant solve details
	"""
	# turn off model output. otherwise prints bunch of info, clogs console
	#m.setParam('OutputFlag', 0)

	# start timer and solve model
	start = timer()
	m.optimize()
	end = timer()
	solve_time = end-start
	objective_value = m.objVal

	if solve_relax:
		# relax and solve to get continuous relaxation and integrality_gap
		vars = m.getVars()
		for var in vars:
			var.VType = GRB.CONTINUOUS
		# TODO could just use r = m.relax()?? - probobly more efficient

		m.optimize()
		continuous_obj_value = m.objVal
		integrality_gap = ((continuous_obj_value-objective_value)/objective_value)*100
	else:
		continuous_obj_value = -1
		integrality_gap = -1
	# terminate model so not allocating resources
	m.terminate()


	# create and return results dictionary
	results = {"solve_time": solve_time,
			   "objective_value": objective_value,
			   "relaxed_solution": continuous_obj_value,
			   "integrality_gap": integrality_gap}
	return results

def no_linearization(quad, **kwargs):
	start = timer()
	n = quad.n
	c = quad.c
	m = Model(name='no_linearization')
	x = m.addVars(n, name="binary_var", vtype=GRB.BINARY)
	if type(quad) is Knapsack: #HSP and UQP don't have cap constraint
		#add capacity constraint(s)
		for k in range(quad.m):
			m.addConstr(quicksum(x[i]*quad.a[k][i] for i in range(n)) <= quad.b[k])

	#k_item constraint if necessary (if KQKP or HSP)
	if quad.num_items > 0:
		m.addConstr(quicksum(x[i] for i in range(n)) == quad.num_items)

	#compute quadratic values contirbution to obj
	quadratic_values = 0
	for i in range(n):
		for j in range(i+1,n):
			quadratic_values = quadratic_values + (x[i]*x[j]*quad.C[i,j])
	#set objective function
	if type(quad)==HSP:
		#HSP doesn't habe any linear terms
		m.setObjective(quadratic_values, GRB.MAXIMIZE)
	else:
		linear_values = quicksum(x[i]*c[i] for i in range(n))
		m.setObjective(linear_values + quadratic_values, GRB.MAXIMIZE)
	end = timer()
	setup_time = end-start
	return [m, setup_time]

def qsap_glovers(qsap, bounds="original", constraints="original", lhs_constraints=False, **kwargs):
	start = timer()
	n = qsap.n
	m = qsap.m
	e = qsap.e
	c = qsap.c
	mdl = Model(name='qsap_glovers')
	x = mdl.addVars(m,n,name="binary_var", vtype=GRB.BINARY)
	mdl.addConstrs((sum(x[i,k] for k in range(n)) == 1) for i in range(m))

	#let gurobi solve w/ quadratic objective function
	# mdl.setObjective(sum(sum(e[i,k]*x[i,k] for k in range(n))for i in range(m))
	# 			+ sum(sum(sum(sum(c[i,k,j,l]*x[i,k]*x[j,l] for l in range(n))for k in range(n))
	# 			for j in range(1+i,m)) for i in range(m-1)), GRB.MAXIMIZE)
	# mdl.optimize()
	# print(mdl.objVal)


	U1 = np.zeros((m,n))
	L0 = np.zeros((m,n))
	U0 = np.zeros((m,n))
	L1 = np.zeros((m,n))
	if bounds=="original":
		for i in range(m-1):
			for k in range(n):
				col = c[i,k,:,:]
				pos_take_vals = col > 0
				pos_take_vals[i,k] = True
				U1[i,k] = np.sum(col[pos_take_vals])
				neg_take_vals = col < 0
				neg_take_vals[i,k] = False
				L0[i,k] = np.sum(col[neg_take_vals])
				if lhs_constraints:
					# pos_take_vals[i,k] = False
					# U0[i,k] = np.sum(col[pos_take_vals])
					# neg_take_vals[i,k] = True
					# L1[i,k] = np.sum(col[neg_take_vals])
					# This should be equivalent but more efficient
					U0[i,k] = U1[i,k] - col[i,k]
					L1[i,k] = L0[i,k] + col[i,k]
	elif bounds=="tight" or bounds=="tighter":
		u_bound_mdl = Model(name="u_bound_m")
		l_bound_mdl = Model(name="l_bound_m")
		if bounds=="tight":
			u_bound_x = u_bound_mdl.addVars(keys1=m,keys2=n,ub=1,lb=0)
			l_bound_x = l_bound_mdl.addVars(keys1=m,keys2=n,ub=1,lb=0)
		elif bounds == "tighter":
			u_bound_x = u_bound_mdl.addVars(keys1=m,keys2=n, vtype=GRB.BINARY)
			l_bound_x = l_bound_mdl.addVars(keys1=m,keys2=n, vtype=GRB.BINARY)
		u_bound_mdl.addConstrs((quicksum(u_bound_x[i,k] for k in range(n)) == 1) for i in range(m))
		l_bound_mdl.addConstrs((quicksum(l_bound_x[i,k] for k in range(n)) == 1) for i in range(m))
		for i in range(m-1):
			for k in range(n):
				u_bound_mdl.setObjective(quicksum(quicksum(c[i,k,j,l]*u_bound_x[j,l] for l in range(n)) for j in range(i+1,m)), GRB.MAXIMIZE)
				l_bound_mdl.setObjective(quicksum(quicksum(c[i,k,j,l]*l_bound_x[j,l] for l in range(n)) for j in range(i+1,m)), GRB.MAXIMIZE)
				u_con = u_bound_mdl.addConstr(u_bound_x[i,k]==1)
				l_con = l_bound_mdl.addConstr(l_bound_x[i,k]==0)
				u_bound_mdl.optimize()
				l_bound_mdl.optimize()
				U1[i,k] = u_bound_mdl.objVal
				L0[i,k] = l_bound_mdl.objVal
				u_bound_mdl.remove(u_con)
				l_bound_mdl.remove(l_con)
				if lhs_constraints:
					u_con = u_bound_mdl.addConstr(u_bound_x[i,k] == 0)
					l_con = l_bound_mdl.addConstr(l_bound_x[i,k] == 1)
					u_bound_mdl.optimize()
					l_bound_mdl.optimize()
					U0[i,k] = u_bound_mdl.objVal
					L1[i,k] = l_bound_mdl.objVal
					u_bound_mdl.remove(u_con)
					l_bound_mdl.remove(l_con)
	else:
		raise Exception(bounds + " is not a valid bound type for glovers")

	#add auxiliary constrains
	if constraints=="original":
		z = mdl.addVars(m,n,lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS)
		mdl.addConstrs(z[i,k] <= x[i,k]*U1[i,k] for i in range(m-1) for k in range(n))
		mdl.addConstrs(z[i,k] <= quicksum(quicksum(c[i,k,j,l]*x[j,l] for l in range(n)) for j in range(i+1,m))
										-L0[i,k]*(1-x[i,k]) for i in range(m-1) for k in range(n))
		if lhs_constraints:
			mdl.addConstrs(z[i,k] >= x[i,k]*L1[i,k] for i in range(m-1) for k in range(n))
			mdl.addConstrs(z[i,k] >= quicksum(quicksum(c[i,k,j,l]*x[j,l] for l in range(n)) for j in range(i+1,m))
										-U0[i,k]*(1-x[i,k]) for i in range(m-1) for k in range(n))
		mdl.setObjective(quicksum(quicksum(e[i,k]*x[i,k] for k in range(n))for i in range(m))
					+ quicksum(quicksum(z[i,k] for k in range(n)) for i in range(m-1)), GRB.MAXIMIZE)
	elif constraints=="sub1":
		s = mdl.addVars(m,n,lb=0)
		mdl.addConstrs(s[i,k] >= U1[i,k]*x[i,k]+L0[i,k]*(1-x[i,k])-quicksum(quicksum(c[i,k,j,l]*x[j,l] for l in range(n)) for j in range(i+1,m))
						for k in range(n) for i in range(m-1))
		mdl.setObjective(quicksum(quicksum(e[i,k]*x[i,k] for k in range(n))for i in range(m))
					+ quicksum(quicksum(U1[i,k]*x[i,k]-s[i,k] for k in range(n)) for i in range(m-1)), GRB.MAXIMIZE)
	elif constraints=="sub2":
		s = mdl.addVars(m,n,lb=0)
		mdl.addConstrs(s[i,k] >= -L0[i,k]*(1-x[i,k])-(x[i,k]*U1[i,k])+quicksum(quicksum(c[i,k,j,l]*x[j,l] for l in range(n)) for j in range(i+1,m))
		 				for k in range(n) for i in range(m-1))
		mdl.setObjective(quicksum(quicksum(e[i,k]*x[i,k] for k in range(n))for i in range(m))
					+ quicksum(quicksum(-s[i,k]-(L0[i,k]*(1-x[i,k])) + quicksum(quicksum(c[i,k,j,l]*x[j,l] for l in range(n))
					for j in range(i+1,m)) for k in range(n)) for i in range(m-1)), GRB.MAXIMIZE)
	else:
		raise Exception(constraints + " is not a valid constraint type for glovers")

	end = timer()
	setup_time = end-start

	#return model
	return [mdl,setup_time]

# p = UQP()
# m = no_linearization(p)[0]
# print(solve_model(m, solve_relax=False))
# m = standard_linearization(p)[0]
# print(solve_model(m))
