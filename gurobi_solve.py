from quadratics import *
import numpy as np
from timeit import default_timer as timer
from gurobipy import *

# turn off model output. otherwise prints bunch of info, clogs console
setParam('OutputFlag',0)
setParam('LogFile',"")

def standard_linearization(quad, lhs_constraints=True, **kwargs):
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
			if lhs_constraints:
				if C[i,j] > 0:
					m.addConstr(w[i,j] <= x[i])
					m.addConstr(w[i,j] <= x[j])
				else:
					m.addConstr(x[i]+x[j]-1 <= w[i,j])
					m.addConstr(w[i,j] >= 0)
			else:
				m.addConstr(w[i,j] <= x[i])
				m.addConstr(w[i,j] <= x[j])
				m.addConstr(x[i]+x[j]-1 <= w[i,j])
				m.addConstr(w[i,j] >= 0)

	#compute quadratic values contirbution to obj
	quadratic_values = 0
	for i in range(n):
		for j in range(i+1, n):
			quadratic_values = quadratic_values + (w[i, j]*(C[i, j]+C[j, i]))
	# set objective function
	linear_values = sum(x[i]*c[i] for i in range(n))
	m.setObjective(linear_values + quadratic_values, GRB.MAXIMIZE)

	# return model + setup time
	return [m, 0]

def glovers_linearization(quad, bounds="tight", constraints="original", lhs_constraints=False, use_diagonal=False, **kwargs):
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
	start = timer()
	if(bounds == "original" or type(quad)==UQP):
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
		bound_m = Model(name='bound_model')
		if bounds=="tight":
			bound_x = bound_m.addVars(n, ub=1, lb=0, vtype=GRB.CONTINUOUS)
		elif bounds=="tighter":
			bound_x = u_bound_m.addVars(n, ub=1, lb=0, vtype=GRB.BINARY)
		if type(quad) is Knapsack:
			for k in range(quad.m):
				#add capacity constraints
				bound_m.addConstr(quicksum(bound_x[i]*a[k][i] for i in range(n)) <= b[k])
		if quad.num_items > 0:
			bound_m.addConstr(quicksum(bound_x[i] for i in range(n)) == quad.num_items)
		for j in range(n):
			# solve for upper bound U1
			bound_m.setObjective(quicksum(C[i, j]*bound_x[i] for i in range(n)), GRB.MAXIMIZE)
			u_con = bound_m.addConstr(bound_x[j] == 1)
			bound_m.optimize()
			#if GRB.OPTIMAL!=2: #2 means optimal, 3 infeasible. see gurobi docs for status_codes
			if bound_m.status != GRB.Status.OPTIMAL:
				#print("non-optimal solve status: " + str(u_bound_m.status) + " when solving for upper bound (U1)")
				bound_m.remove(u_con)
				bound_m.addConstr(bound_x[j]==0)
				m.addConstr(x[j]==0)
				bound_m.optimize()
			else:
				bound_m.remove(u_con)
			U1[j] = bound_m.objVal

			# solve for lower bound L0
			bound_m.setObjective(quicksum(C[i, j]*bound_x[i] for i in range(n)), GRB.MINIMIZE)
			l_con = bound_m.addConstr(bound_x[j] == 0)
			bound_m.optimize()
			if bound_m.status != GRB.Status.OPTIMAL:
				#print("non-optimal solve status: " + str(l_bound_m.status) + " when solving for lower bound (L0)")
				bound_m.remove(l_con)
				bound_m.addConstr(bound_x[j]==1)
				m.addConstr(x[j]==1)
				bound_m.optimize()
			else:
				bound_m.remove(l_con)
			L0[j] = bound_m.objVal

			if lhs_constraints:
				# solve for upper bound U0
				bound_m.setObjective(quicksum(C[i, j]*bound_x[i] for i in range(n)), GRB.MAXIMIZE)
				u_con = bound_m.addConstr(bound_x[j] == 0)
				bound_m.optimize()
				#if GRB.OPTIMAL!=2: #2 means optimal, 3 infeasible. see gurobi docs for status_codes
				if bound_m.status != GRB.Status.OPTIMAL:
					#print("non-optimal solve status: " + str(u_bound_m.status) + " when solving for upper bound (U1)")
					bound_m.remove(u_con)
					bound_m.addConstr(bound_x[j]==1)
					m.addConstr(x[j]==1)
					bound_m.optimize()
				else:
					bound_m.remove(u_con)
				U0[j] = bound_m.objVal

				# solve for lower bound L1
				bound_m.setObjective(quicksum(C[i, j]*bound_x[i] for i in range(n)), GRB.MINIMIZE)
				l_con = bound_m.addConstr(bound_x[j] == 1)
				bound_m.optimize()
				if bound_m.status != GRB.Status.OPTIMAL:
					#print("non-optimal solve status: " + str(l_bound_m.status) + " when solving for lower bound (L0)")
					bound_m.remove(l_con)
					bound_m.addConstr(bound_x[j]==0)
					m.addConstr(x[j]==0)
					bound_m.optimize()
				else:
					bound_m.remove(l_con)
				L1[j] = bound_m.objVal
		# terminate bound model
		bound_m.terminate()
	else:
		raise Exception(bounds + " is not a valid bound type for glovers")
	end = timer()
	setup_time = end-start

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

		m.setObjective(quicksum(c[j]*x[j] + z[j] for j in range(n)), GRB.MAXIMIZE)


	elif(constraints=="sub1" or constraints=="sub2"):
		#can make one of 2 substitutions using slack variables to further reduce # of constraints
		s = m.addVars(n, vtype=GRB.CONTINUOUS)
		for j in range(n):
			tempsum = quicksum(C[i, j]*x[i] for i in range(n))
			if constraints=="sub1":
				m.addConstr(s[j] >= U1[j]*x[j] - tempsum + L0[j]*(1-x[j]))
			else:
				m.addConstr(s[j] >= -U1[j]*x[j] + tempsum - L0[j]*(1-x[j]))
		if constraints=="sub1":
			m.setObjective(quicksum(c[i]*x[i] + (U1[i]*x[i]-s[i]) for i in range(n)), GRB.MAXIMIZE)
		else:
			#m.setObjective(quicksum(c[i]*x[i] + quicksum(C[i, j]*x[j] for j in range(n))-L0[i]*(1-x[i])-s[i] for i in range(n)), GRB.MAXIMIZE)
			m.setObjective(quicksum(c[j]*x[j] for j in range(n)) + quicksum(quicksum(C[i,j]*x[i] for i in range(n))-L0[j]*(1-x[j])-s[j] for j in range(n)), GRB.MAXIMIZE)
	else:
		raise Exception(constraints + " is not a valid constraint type for glovers")

	# return model
	return [m, setup_time]

def glovers_linearization_prlt(quad, **kwargs):
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
		linear_values = quicksum(x[j]*c[j] for j in range(n))
		m.setObjective(linear_values + quadratic_values, GRB.MAXIMIZE)


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

def glovers_linearization_rlt(quad, bounds="tight", constraints="original", **kwargs):
	def rlt1_linearization(quad):
		n = quad.n
		c = quad.c
		C = quad.C
		a = quad.a
		b = quad.b

		#create model and add variables
		m = Model(name='RLT-1_linearization')
		x = m.addVars(n,name='binary_var', lb=0, ub=1, vtype=GRB.CONTINUOUS)
		w = m.addVars(n, n, lb=0, ub=1, vtype=GRB.CONTINUOUS)
		y = m.addVars(n, n, lb=0, ub=1, vtype=GRB.CONTINUOUS)

		if type(quad) is Knapsack:
			# Multiply knapsack constraints by each x_j and (1-x_j)
			# Note: no need to include original knapsack constraints
			for k in range(quad.m):
				for j in range(n):
					#con 12
					m.addConstr(quicksum(a[k][i]*w[i,j] for i in range(n) if i!=j)<=(b[k]-a[k][j])*x[j])
					#con 14
					m.addConstr(quicksum(a[k][i]*y[i,j] for i in range(n) if i!=j)<=b[k]*(1-x[j]))

		#k_item constraint if necessary (if KQKP or HSP)
		if quad.num_items > 0:
			# Multiply partition constraint by each x_j
			# Note: There is no need to multiple by each (1-x_j), but must include original constraints
			m.addConstr(quicksum(x[i] for i in range(n)) == quad.num_items)
			for j in range(n):
				m.addConstr(quicksum(w[i,j] for i in range(n) if i!=j)== (quad.num_items-1)*x[j])

		# Add in symmetry constraints
		for i in range(n):
			for j in range(i+1,n):
				#con 16
				m.addConstr(w[i,j]==w[j,i], name='con16_'+str(i)+"_"+str(j))

		for j in range(n):
			for i in range(n):
				if(i==j):
					continue
				m.addConstr(w[i,j] <= x[j])
				m.addConstr(y[i,j] <= 1-x[j])
				m.addConstr(y[i,j] == x[i]-w[i,j], name='con17_'+str(i)+"_"+str(j))

		#add objective function
		quadratic_values = 0
		for j in range(n):
			for i in range(n):
				if(i==j):
					continue
				quadratic_values = quadratic_values + (C[i,j]*w[i,j])

		linear_values = sum(x[j]*c[j] for j in range(n))
		m.setObjective(linear_values + quadratic_values, GRB.MAXIMIZE)

		#return model
		return m

	n = quad.n
	c = quad.c
	C = quad.C
	a = quad.a
	b = quad.b

	#model with rlt1, solve continuous relax and get duals to constraints 16,17
	m = rlt1_linearization(quad)
	start = timer()
	m.optimize()

	print()
	print("RLT objective value = " + str(m.objVal))
	print()

	# Obtain the duals to the symmetry constraints
	duals16 = np.zeros((n,n))
	duals17 = np.zeros((n,n))
	for i in range(n):
		for j in range(i+1,n):
			con_name = 'con16_'+str(i)+"_"+str(j)
			duals16[i][j] = (m.getConstrByName(con_name).getAttr(GRB.attr.Pi))
		for j in range(n):
			if i==j:
				continue
			con_name = 'con17_'+str(i)+"_"+str(j)
			duals17[i][j] = (m.getConstrByName(con_name).getAttr(GRB.attr.Pi))

	# Delete RLT model
	m.terminate()

	Cbar = np.zeros((n,n))
	Chat = np.zeros((n,n))
	#optimal split, found using dual vars from rlt1 continuous relaxation
	for i in range(n):
		for j in range(n):
			if i<j:
				Cbar[i,j] = C[i,j]-duals16[i,j]-duals17[i,j]
				Chat[i,j] = -duals17[i,j]
			elif i>j:
				Cbar[i,j] = C[i,j]+duals16[j,i]-duals17[i,j]
				Chat[i,j] = -duals17[i,j]

	#update linear values as well
	for j in range(n):
		c[j] = c[j] + sum(duals17[j,i] for i in range(n) if i!=j)

	#create model and add variables
	m = Model(name='glovers_linearization_rlt_'+bounds+'_'+constraints)
	x = m.addVars(n, name="binary_var", vtype=GRB.BINARY)

	if type(quad) is Knapsack: #HSP and UQP don't have cap constraint
		#add capacity constraint(s)
		for k in range(quad.m):
			m.addConstr(quicksum(a[k][i]*x[i] for i in range(n)) <= b[k])

	#k_item constraint if necessary (if KQKP or HSP)
	if quad.num_items > 0:
		m.addConstr(quicksum(x[i] for i in range(n)) == quad.num_items)

	#determine bounds for each column of C
	Ubar1 = np.zeros(n)
	Lbar0 = np.zeros(n)
	Uhat0 = np.zeros(n)
	Lhat1 = np.zeros(n)

	if(bounds=="original"):
		for j in range(n):
			col1 = Cbar[:,j]
			col2 = Chat[:,j]
			Ubar1[j] = np.sum(col1[col1>0])
			Lbar0[j] = np.sum(col1[col1<0])
			Uhat0[j] = np.sum(col2[col2>0])
			Lhat1[j] = np.sum(col2[col2<0])
	elif(bounds=="tight"):
		bound_m = Model(name='bound_model')
		x_bound = bound_m.addVars(n, ub=1, lb=0, vtype=GRB.CONTINUOUS)

		# Add in original structural constraints
		if type(quad) is Knapsack: #HSP and UQP don't have cap constraint
			# Include knapsack constraints
			for k in range(quad.m):
				bound_m.addConstr(quicksum(a[k][i]*x_bound[i] for i in range(n)) <= b[k])

		#k_item constraint if necessary (if KQKP or HSP)
		if quad.num_items > 0:
			bound_m.addConstr(quicksum(x_bound[i] for i in range(n)) == quad.num_items)

		for j in range(n):
			# Solve for Ubar1
			bound_m.setObjective(quicksum(Cbar[i,j]*x_bound[i] for i in range(n) if i!=j), GRB.MAXIMIZE)
			xEquals1 = bound_m.addConstr(x_bound[j]==1)
			bound_m.optimize()
			if bound_m.status != GRB.Status.OPTIMAL:
				# in case the xEquals1 constraint makes problem infeasible. (happens with kitem sometimes)
				bound_m.remove(xEquals1)
				bound_m.addConstr(x_bound[j]==0)
				m.addConstr(x[j]==0)
				bound_m.optimize()
			else:
				bound_m.remove(xEquals1)
			Ubar1[j] = bound_m.objVal

			# Solve for Lbar0
			xEquals0 = bound_m.addConstr(x_bound[j]==0)
			bound_m.setObjective(quicksum(Cbar[i,j]*x_bound[i] for i in range(n) if i!=j), GRB.MINIMIZE)
			bound_m.optimize()
			if bound_m.status != GRB.Status.OPTIMAL:
				# in case the xEquals1 constraint makes problem infeasible. (happens with kitem sometimes)
				bound_m.remove(xEquals0)
				bound_m.addConstr(x_bound[j]==1)
				m.addConstr(x[j]==1)
				bound_m.optimize()
			else:
				bound_m.remove(xEquals0)
			Lbar0[j] = bound_m.objVal

			# Solve for Uhat0
			bound_m.setObjective(quicksum(Chat[i,j]*x_bound[i] for i in range(n) if i!=j), GRB.MAXIMIZE)
			xEquals0 = bound_m.addConstr(x_bound[j]==0)
			bound_m.optimize()
			if bound_m.status != GRB.Status.OPTIMAL:
				# in case the xEquals1 constraint makes problem infeasible. (happens with kitem sometimes)
				bound_m.remove(xEquals0)
				bound_m.addConstr(x_bound[j]==1)
				m.addConstr(x[j]==1)
				bound_m.optimize()
			else:
				bound_m.remove(xEquals0)
			Uhat0[j] = bound_m.objVal

			# Solve for Lhat1
			bound_m.setObjective(quicksum(Chat[i,j]*x_bound[i] for i in range(n) if i!=j), GRB.MINIMIZE)
			xEquals1 = bound_m.addConstr(x_bound[j]==1)
			bound_m.optimize()
			if bound_m.status != GRB.Status.OPTIMAL:
				# in case the xEquals1 constraint makes problem infeasible. (happens with kitem sometimes)
				bound_m.remove(xEquals1)
				bound_m.addConstr(x_bound[j]==0)
				m.addConstr(x[j]==0)
				bound_m.optimize()
			else:
				bound_m.remove(xEquals1)
			Lhat1[j] = bound_m.objVal

		# Delete bound model
		bound_m.terminate()
	else:
		raise Exception(bounds + " is not a valid bound type for glovers")

	end = timer()
	setup_time = end-start

	if(constraints=="original"):
		z1 = m.addVars(n,lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS)
		z2 = m.addVars(n,lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS)

		for j in range(n):
			m.addConstr(z1[j] <= Ubar1[j]*x[j])
			tempsum1 = sum(Cbar[i,j]*x[i] for i in range(n) if i!=j)
			m.addConstr(z1[j] <= tempsum1 - Lbar0[j]*(1-x[j]))

			m.addConstr(z2[j] <= Uhat0[j]*(1-x[j]))
			tempsum2 = sum(Chat[i,j]*x[i] for i in range(n) if i!=j)
			m.addConstr(z2[j] <= tempsum2 - (Lhat1[j]*x[j]))

		# Set up the objective function
		m.setObjective(quicksum(c[j]*x[j] + z1[j] + z2[j] for j in range(n)), GRB.MAXIMIZE)
	else:
		raise Exception(constraints + " is not a valid constraint type for glovers")

	#return model
	return [m,setup_time]

def extended_linear_formulation(quad, **kwargs):
	n = quad.n
	c = quad.c
	C = quad.C
	a = quad.a
	b = quad.b

	#create model and add variables
	m = Model(name='extended_linear_formulation')
	x = m.addVars(n, name="binary_var", vtype=GRB.BINARY)
	z = m.addVars(n,n,n, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS)

	if type(quad) is Knapsack: #HSP and UQP don't have cap constraint
		#add capacity constraint(s)
		for k in range(quad.m):
			m.addConstr(quicksum(x[i]*a[k][i] for i in range(n)) <= b[k])

	#k_item constraint if necessary (if KQKP or HSP)
	if quad.num_items > 0:
		m.addConstr(quicksum(x[i] for i in range(n)) == quad.num_items)

	#add auxiliary constraints
	for i in range(n):
		for j in range(i+1,n):
			m.addConstr(z[i,i,j]+z[j,i,j] <= 1)
			if C[i,j] < 0:
				m.addConstr(x[i] + z[i,i,j] <= 1)
				m.addConstr(x[j] + z[j,i,j] <= 1)
			elif C[i,j] > 0:
				m.addConstr(x[i] + z[i,i,j] + z[j,i,j] >= 1)
				m.addConstr(x[j] + z[i,i,j] + z[j,i,j] >= 1)

	#compute quadratic values contirbution to obj
	constant = 0
	quadratic_values = 0
	for i in range(n):
		for j in range(i+1,n):
			constant = constant + C[i,j]
			quadratic_values = quadratic_values + (C[i,j]*(z[i,i,j]+z[j,i,j]))
	#set objective function
	linear_values = quicksum(x[i]*c[i] for i in range(n))
	m.setObjective(linear_values + constant - quadratic_values, GRB.MAXIMIZE)

	#return model + setup time
	return [m, 0]

def ss_linear_formulation(quad, **kwargs):
	"""
	Sherali-Smith Linear Formulation
	"""
	n = quad.n
	c = quad.c
	C = quad.C
	a = quad.a
	b = quad.b

	#create model and add variables
	m = Model(name='ss_linear_formulation')
	x = m.addVars(n, name="binary_var", vtype=GRB.BINARY)
	s = m.addVars(n, vtype=GRB.CONTINUOUS)
	y = m.addVars(n, vtype=GRB.CONTINUOUS)

	if type(quad) is Knapsack: #HSP and UQP don't have cap constraint
		#add capacity constraint(s)
		for k in range(quad.m):
			m.addConstr(quicksum(x[i]*a[k][i] for i in range(n)) <= b[k])

	#k_item constraint if necessary (if KQKP or HSP)
	if quad.num_items > 0:
		m.addConstr(quicksum(x[i] for i in range(n)) == quad.num_items)

	start = timer()
	U = np.zeros(n)
	L = np.zeros(n)
	u_bound_m = Model(name='upper_bound_model')
	l_bound_m = Model(name='lower_bound_model')
	bound_x = u_bound_m.addVars(n, ub=1, lb=0, vtype=GRB.CONTINUOUS)
	bound_x = l_bound_m.addVars(n, ub=1, lb=0, vtype=GRB.CONTINUOUS)
	if type(quad) is Knapsack:
		for k in range(quad.m):
			#add capacity constraints
			u_bound_m.addConstr(quicksum(bound_x[i]*a[k][i] for i in range(n)) <= b[k])
			l_bound_m.addConstr(quicksum(bound_x[i]*a[k][i] for i in range(n)) <= b[k])
	if quad.num_items > 0:
		u_bound_m.addConstr(quicksum(bound_x[i] for i in range(n)) == quad.num_items)
		l_bound_m.addConstr(quicksum(bound_x[i] for i in range(n)) == quad.num_items)
	for i in range(n):
		#for each col, solve model to find upper/lower bound
		u_bound_m.setObjective(sense=GRB.MAXIMIZE, expr=quicksum(C[i,j]*bound_x[j] for j in range(n)))
		l_bound_m.setObjective(sense=GRB.MINIMIZE, expr=quicksum(C[i,j]*bound_x[j] for j in range(n)))
		u_bound_m.optimize()
		l_bound_m.optimize()
		U[i] = u_bound_m.objVal
		L[i] = l_bound_m.objVal
	bound_m.terminate()
	end = timer()
	setup_time = end-start

	#add auxiliary constraints
	for i in range(n):
		m.addConstr(sum(C[i,j]*x[j] for j in range(n))-s[i]-L[i]==y[i])
		m.addConstr(y[i] <= (U[i]-L[i])*(1-x[i]))
		m.addConstr(s[i] <= (U[i]-L[i])*x[i])
		m.addConstr(y[i] >= 0)
		m.addConstr(s[i] >= 0)

	#set objective function
	m.setObjective(sum(s[i]+x[i]*(c[i]+L[i])for i in range(n)), sense=GRB.MAXIMIZE)

	#return model + setup time
	return [m, setup_time]

def qsap_standard(qsap, **kwargs):
	n = qsap.n
	m = qsap.m
	e = qsap.e
	c = qsap.c

	#create model and add variables
	mdl = Model(name='qsap_standard_linearization')
	x = mdl.addVars(m,n,name="binary_var", vtype=GRB.BINARY)
	w = mdl.addVars(m,n,m,n, vtype=GRB.CONTINUOUS)

	mdl.addConstrs((sum(x[i,k] for k in range(n)) == 1) for i in range(m))

	#add auxiliary constraints
	#TODO implement lhs here?
	for i in range(m-1):
		for k in range(n):
			for j in range(i+1,m):
				for l in range(n):
					mdl.addConstr(w[i,k,j,l] <= x[i,k])
					mdl.addConstr(w[i,k,j,l] <= x[j,l])
					mdl.addConstr(x[i,k] + x[j,l] - 1 <= w[i,k,j,l])
					mdl.addConstr(w[i,k,j,l] >= 0)

	#compute quadratic values contirbution to obj
	quadratic_values = 0
	for i in range(m-1):
		for j in range(i+1,m):
			for k in range(n):
				for l in range(n):
					quadratic_values = quadratic_values + (c[i,k,j,l]*(w[i,k,j,l]))

	linear_values = quicksum(x[i,k]*e[i,k] for k in range(n) for i in range(m))
	mdl.setObjective(linear_values + quadratic_values, GRB.MAXIMIZE)

	#return model. no setup time for std
	return [mdl, 0]

def qsap_glovers(qsap, bounds="original", constraints="original", lhs_constraints=False, **kwargs):
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
	start = timer()
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
		bound_mdl = Model(name="bound_m")
		if bounds=="tight":
			bound_x = bound_mdl.addVars(m,n,ub=1,lb=0)
		elif bounds == "tighter":
			bound_x = bound_mdl.addVars(m,n, vtype=GRB.BINARY)
		bound_mdl.addConstrs((quicksum(bound_x[i,k] for k in range(n)) == 1) for i in range(m))
		for i in range(m-1):
			for k in range(n):
				# solve for upper bound U1
				bound_mdl.setObjective(quicksum(quicksum(c[i,k,j,l]*bound_x[j,l] for l in range(n)) for j in range(i+1,m)), GRB.MAXIMIZE)
				u_con = bound_mdl.addConstr(bound_x[i,k]==1)
				bound_mdl.optimize()
				bound_mdl.remove(u_con)
				U1[i,k] = bound_mdl.objVal

				# solve for lower bound L0
				bound_mdl.setObjective(quicksum(quicksum(c[i,k,j,l]*bound_x[j,l] for l in range(n)) for j in range(i+1,m)), GRB.MINIMIZE)
				l_con = bound_mdl.addConstr(bound_x[i,k]==0)
				bound_mdl.optimize()
				bound_mdl.remove(l_con)
				L0[i,k] = bound_mdl.objVal

				if lhs_constraints:
					# solve for upper bound U0
					bound_mdl.setObjective(quicksum(quicksum(c[i,k,j,l]*bound_x[j,l] for l in range(n)) for j in range(i+1,m)), GRB.MAXIMIZE)
					u_con = bound_mdl.addConstr(bound_x[i,k] == 0)
					bound_mdl.optimize()
					bound_mdl.remove(u_con)
					U0[i,k] = bound_mdl.objVal

					# solve for lower bound L1
					bound_mdl.setObjective(quicksum(quicksum(c[i,k,j,l]*bound_x[j,l] for l in range(n)) for j in range(i+1,m)), GRB.MINIMIZE)
					l_con = bound_mdl.addConstr(bound_x[i,k] == 1)
					bound_mdl.optimize()
					L1[i,k] = bound_mdl.objVal
					bound_mdl.remove(l_con)
		# end bound model
		bound_mdl.terminate()
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

def qsap_elf(qsap, **kwargs):
	n = qsap.n
	m = qsap.m
	e = qsap.e
	c = qsap.c
	mdl = Model(name='qsap_elf')
	x = mdl.addVars(m,n,name="binary_var", vtype=GRB.BINARY)
	z = mdl.addVars(m,n,m,n,m,n, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS)

	mdl.addConstrs((sum(x[i,k] for k in range(n)) == 1) for i in range(m))

	#add auxiliary constraints
	for i in range(m-1):
		for j in range(i+1,m):
			for k in range(n):
				for l in range(n):
					mdl.addConstr(z[i,k,i,k,j,l] + z[j,l,i,k,j,l] <= 1)
					mdl.addConstr(x[i,k] + z[i,k,i,k,j,l] <= 1)
					mdl.addConstr(x[j,l] + z[j,l,i,k,j,l] <= 1)
					mdl.addConstr(x[i,k] + z[i,k,i,k,j,l] + z[j,l,i,k,j,l] >= 1)
					mdl.addConstr(x[j,l] + z[i,k,i,k,j,l] + z[j,l,i,k,j,l] >= 1)

	#compute quadratic values contirbution to obj
	constant = 0
	quadratic_values = 0
	for i in range(m-1):
		for j in range(i+1,m):
			for k in range(n):
				for l in range(n):
					constant = constant + c[i,k,j,l]
					quadratic_values = quadratic_values + (c[i,k,j,l]*(z[i,k,i,k,j,l]+z[j,l,i,k,j,l]))

	linear_values = quicksum(x[i,k]*e[i,k] for k in range(n) for i in range(m))
	mdl.setObjective(linear_values + constant - quadratic_values, GRB.MAXIMIZE)

	#return model + setup time
	return [mdl, 0]

def qsap_ss(qsap, **kwargs):
	"""
	Sherali-Smith Linear Formulation
	"""
	n = qsap.n
	m = qsap.m
	e = qsap.e
	c = qsap.c

	#create model and add variables
	mdl = Model(name='qsap_ss')
	x = mdl.addVars(m,n,name="binary_var", vtype=GRB.BINARY)
	s = mdl.addVars(m,n, vtype=GRB.CONTINUOUS)
	y = mdl.addVars(m,n, vtype=GRB.CONTINUOUS)

	mdl.addConstrs((sum(x[i,k] for k in range(n)) == 1) for i in range(m))

	start = timer()
	U = np.zeros((m,n))
	L = np.zeros((m,n))
	bound_mdl = Model(name='upper_bound_model')
	bound_x = bound_mdl.addVars(m,n, ub=1, lb=0)
	bound_mdl.addConstrs((sum(bound_x[i,k] for k in range(n)) == 1) for i in range(m))
	for i in range(m-1):
		for k in range(n):
			bound_mdl.setObjective(sum(sum(c[i,k,j,l]*bound_x[j,l] for l in range(n)) for j in range(i+1,m)), GRB.MAXIMIZE)
			bound_mdl.optimize()
			U[i,k] = bound_mdl.objVal

			bound_mdl.setObjective(sum(sum(c[i,k,j,l]*bound_x[j,l] for l in range(n)) for j in range(i+1,m)), GRB.MINIMIZE)
			bound_mdl.optimize()
			L[i,k] = bound_mdl.objVal
	bound_mdl.terminate()
	end = timer()
	setup_time = end-start

	#add auxiliary constraints
	for i in range(m-1):
		for k in range(n):
			mdl.addConstr(sum(sum(c[i,k,j,l]*x[j,l] for j in range(i+1,m)) for l in range(n))-s[i,k]-L[i,k]==y[i,k])
			mdl.addConstr(y[i,k] <= (U[i,k]-L[i,k])*(1-x[i,k]))
			mdl.addConstr(s[i,k] <= (U[i,k]-L[i,k])*x[i,k])
			mdl.addConstr(y[i,k] >= 0)
			mdl.addConstr(s[i,k] >= 0)

	#set objective function
	linear_values = sum(sum(e[i,k]*x[i,k] for i in range(m)) for k in range(n))
	mdl.setObjective(linear_values + sum(sum(s[i,k]+x[i,k]*(L[i,k]) for i in range(m-1)) for k in range(n)), GRB.MAXIMIZE)


	#return model + setup time
	return [mdl, setup_time]

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
	linear_values = quicksum(x[i]*c[i] for i in range(n))
	m.setObjective(linear_values + quadratic_values, GRB.MAXIMIZE)

	end = timer()
	setup_time = end-start
	return [m, setup_time]

def solve_model(m, solve_relax=True, **kwargs):
	"""
	Takes in an unsolved gurobi model of a MIP. Solves it as well as continuous
	relaxation and returns a dictionary containing relevant solve details
	"""
	# turn off model output. otherwise prints bunch of info, clogs console
	#m.setParam('OutputFlag', 0)
	time_limit = False
	m.setParam('TimeLimit',600)
	# start timer and solve model
	start = timer()
	m.optimize()
	end = timer()
	if(m.status == 9):
		print('time limit exceeded')
		time_limit=True
	solve_time = end-start
	objective_value = m.objVal

	if solve_relax and not time_limit:
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
			   "integrality_gap": integrality_gap,
				"time_limit":time_limit}
	return results
