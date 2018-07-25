from quadratics import *
import numpy as np
from timeit import default_timer as timer
import sys
sys.path.insert(0,"c:/xpressmp/bin")
import xpress as xp

#xp.addcbmsghandler(None,"xpress.log",0)

def standard_linearization(quad, lhs_constraints=True, **kwargs):
	n = quad.n
	c = quad.c
	C = quad.C
	a = quad.a
	b = quad.b

	# create model and add variables
	m = xp.problem(name='standard_linearization')
	x = np.array([xp.var(vartype=xp.binary) for i in range(n)])
	w = np.array([[xp.var(vartype=xp.continuous) for i in range(n)] for i in range(n)])
	m.addVariable(x,w)

	if type(quad) is Knapsack:  # HSP and UQP don't have cap constraint
		# add capacity constraint(s)
		for k in range(quad.m):
			m.addConstraint(xp.Sum(x[i]*a[k][i] for i in range(n)) <= b[k])

	#k_item constraint if necessary (if KQKP or HSP)
	if quad.num_items > 0:
		m.addConstraint(xp.Sum(x[i] for i in range(n)) == quad.num_items)

	# add auxiliary constraints
	for i in range(n):
		for j in range(i+1, n):
			if lhs_constraints:
				if C[i,j] > 0:
					m.addConstraint(w[i,j] <= x[i])
					m.addConstraint(w[i,j] <= x[j])
				else:
					m.addConstraint(x[i]+x[j]-1 <= w[i,j])
					m.addConstraint(w[i,j] >= 0)
			else:
				m.addConstraint(w[i,j] <= x[i])
				m.addConstraint(w[i,j] <= x[j])
				m.addConstraint(x[i]+x[j]-1 <= w[i,j])
				m.addConstraint(w[i,j] >= 0)

	#compute quadratic values contirbution to obj
	quadratic_values = 0
	for i in range(n):
		for j in range(i+1, n):
			quadratic_values = quadratic_values + (w[i, j]*(C[i, j]+C[j, i]))
	# set objective function
	linear_values = sum(x[i]*c[i] for i in range(n))
	m.setObjective(linear_values + quadratic_values, sense=xp.maximize)

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
	m = xp.problem(name='glovers_linearization_'+bounds+'_'+constraints)
	x = np.array([xp.var(vartype=xp.binary) for i in range(n)])
	m.addVariable(x)

	if type(quad) is Knapsack:  # HSP and UQP don't have cap constraint
		# add capacity constraint(s)
		for k in range(quad.m):
			m.addConstraint(xp.Sum(x[i]*a[k][i] for i in range(n)) <= b[k])

	#k_item constraint if necessary (if KQKP or HSP)
	if quad.num_items > 0:
		m.addConstraint(xp.Sum(x[i] for i in range(n)) == quad.num_items)

	# determine bounds for each column of C
	#U1,L1 must take item at index j, U0,L0 must not take
	U1 = np.zeros(n)
	L0 = np.zeros(n)
	U0 = np.zeros(n)
	L1 = np.zeros(n)
	start = timer()
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
		bound_m = xp.problem(name='bound_model')
		bound_m.setlogfile("xpress.log")
		if bounds=="tight":
			bound_x = np.array([xp.var(vartype=xp.continuous, lb=0, ub=1) for i in range(n)])
		elif bounds=="tighter":
			bound_x = np.array([xp.var(vartype=xp.binary) for i in range(n)])
		bound_m.addVariable(bound_x)
		if type(quad) is Knapsack:
			for k in range(quad.m):
				#add capacity constraints
				bound_m.addConstraint(xp.Sum(bound_x[i]*a[k][i] for i in range(n)) <= b[k])
		if quad.num_items > 0:
			bound_m.addConstraint(xp.Sum(bound_x[i] for i in range(n)) == quad.num_items)
		for j in range(n):
			#for each col, solve model to find upper/lower bound
			bound_m.setObjective(xp.Sum(C[i, j]*bound_x[i] for i in range(n)), sense=xp.maximize)
			u_con = bound_x[j] == 1
			bound_m.addConstraint(u_con)
			bound_m.solve()
			if "optimal" not in str(bound_m.getProbStatusString()):
				#print("non-optimal solve status: " + str(u_bound_m.getProbStatusString()) + " when solving for upper bound (U1)")
				bound_m.delConstraint(u_con)
				bound_m.addConstraint(bound_x[j]==0)
				m.addConstraint(x[j]==0)
				bound_m.solve()
			else:
				bound_m.delConstraint(u_con)
			U1[j] = bound_m.getObjVal()

			bound_m.setObjective(xp.Sum(C[i, j]*bound_x[i] for i in range(n)), sense=xp.minimize)
			l_con = bound_x[j] == 0
			bound_m.addConstraint(l_con)
			bound_m.solve()
			if "optimal" not in str(bound_m.getProbStatusString()):
				#print("non-optimal solve status: " + str(l_bound_m.getProbStatusString()) + " when solving for lower bound (L0)")
				bound_m.delConstraint(l_con)
				bound_m.addConstraint(bound_x[j]==1)
				m.addConstraint(x[j]==1)
				bound_m.solve()
			else:
				bound_m.delConstraint(l_con)
			L0[j] = bound_m.getObjVal()

			if lhs_constraints:
				bound_m.setObjective(xp.Sum(C[i, j]*bound_x[i] for i in range(n)), sense=xp.maximize)
				u_con = bound_x[j] == 0
				bound_m.addConstraint(u_con)
				bound_m.solve()
				if "optimal" not in str(bound_m.getProbStatusString()):
					#print("non-optimal solve status: " + str(u_bound_m.status) + " when solving for upper bound (U0)")
					bound_m.delConstraint(u_con)
					bound_m.addConstraint(bound_x[j]==1)
					m.addConstraint(x[j]==1)
					bound_m.solve()
				else:
					bound_m.delConstraint(u_con)
				U0[j] = bound_m.getObjVal()

				bound_m.setObjective(xp.Sum(C[i, j]*bound_x[i] for i in range(n)), sense=xp.minimize)
				l_con = bound_x[j] == 1
				bound_m.addConstraint(l_con)
				bound_m.solve()
				if "optimal" not in str(bound_m.getProbStatusString()):
					#print("non-optimal solve status: " + str(l_bound_m.status) + " when solving for lower bound (L1)")
					bound_m.delConstraint(l_con)
					bound_m.addConstraint(bound_x[j]==0)
					m.addConstraint(x[j]==0)
					bound_m.solve()
				else:
					bound_m.delConstraint(l_con)
				L1[j] = bound_m.getObjVal()
	else:
		raise Exception(bounds + " is not a valid bound type for glovers")
	end = timer()
	setup_time = end-start

	# add auxiliary constrains
	if(constraints == "original"):
		#original glovers constraints
		z = np.array([xp.var(vartype=xp.continuous, lb=-xp.infinity) for i in range(n)])
		m.addVariable(z)
		m.addConstraint(z[j] <= U1[j]*x[j] for j in range(n))
		if lhs_constraints:
			m.addConstraint(z[j] >= L1[j]*x[j] for j in range(n))
		for j in range(n):
			tempsum = xp.Sum(C[i, j]*x[i] for i in range(n))
			m.addConstraint(z[j] <= tempsum - L0[j]*(1-x[j]))
			if lhs_constraints:
				m.addConstraint(z[j] >= tempsum - U0[j]*(1-x[j]))

		m.setObjective(xp.Sum(c[j]*x[j] + z[j] for j in range(n)), sense=xp.maximize)


	elif(constraints=="sub1" or constraints=="sub2"):
		#can make one of 2 substitutions using slack variables to further reduce # of constraints
		s = np.array([xp.var(vartype=xp.continuous, lb=0) for i in range(n)])
		m.addVariable(s)
		for j in range(n):
			tempsum = xp.Sum(C[i, j]*x[i] for i in range(n))
			if constraints=="sub1":
				m.addConstraint(s[j] >= U1[j]*x[j] - tempsum + L0[j]*(1-x[j]))
			else:
				m.addConstraint(s[j] >= -U1[j]*x[j] + tempsum - L0[j]*(1-x[j]))
		if constraints=="sub1":
			m.setObjective(xp.Sum(c[i]*x[i] + (U1[i]*x[i]-s[i]) for i in range(n)), sense=xp.maximize)
		else:
			m.setObjective(xp.Sum(c[j]*x[j] for j in range(n)) + xp.Sum(xp.Sum(C[i,j]*x[i] for i in range(n))-L0[j]*(1-x[j])-s[j] for j in range(n)), sense=xp.maximize)
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
		m = xp.problem(name='PRLT-1_linearization')
		x = m.addVars(n, lb=0, ub=1, vtype=GRB.CONTINUOUS)
		w = m.addVars(n, n, vtype=GRB.CONTINUOUS)

		if type(quad) is Knapsack:  # HSP and UQP don't have cap constraint
			# add capacity constraint(s)
			for k in range(quad.m):
				m.addConstraint(xp.Sum(x[i]*a[k][i] for i in range(n)) <= b[k])

		#k_item constraint if necessary (if KQKP or HSP)
		if quad.num_items > 0:
			m.addConstraint(xp.Sum(x[i] for i in range(n)) == quad.num_items)

		# add auxiliary constraints
		for i in range(n):
			for j in range(i+1, n):
				m.addConstraint(w[i, j] == w[j, i], name='con16'+str(i)+str(j))

		for k in range(quad.m):
			for j in range(n):
				m.addConstraint(xp.Sum(a[k][i]*w[i, j] for i in range(n) if i != j) <= (b[k]-a[k][j])*x[j])
				for i in range(n):
					m.addConstraint(w[i, j] <= x[j])

		# add objective function

		quadratic_values = 0
		for j in range(n):
			for i in range(n):
				if(i == j):
					continue
				quadratic_values = quadratic_values + (C[i, j]*w[i, j])

		m.setObjective(linear_values + quadratic_values, sense=xp.maximize)
		linear_values = xp.Sum(x[j]*c[j] for j in range(n))

		# return model
		return m

	start = timer()
	m = prlt1_linearization(quad)
	m.solve()

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

		# create model and add variables
		m = xp.problem(name='RLT-1_linearization')
		m.setlogfile("xpress.log")
		x = np.array([xp.var(vartype=xp.continuous, lb=0, ub=1) for i in range(n)])
		w = np.array([[xp.var(vartype=xp.continuous) for i in range(n)] for i in range(n)])
		y = np.array([[xp.var(vartype=xp.continuous) for i in range(n)] for i in range(n)])
		m.addVariable(x,w,y)

		#keep track of order constraints are added so can retrieve duals values later
		constraint_count = 0
		con16 = []
		con17 = []

		# add constraints 16 and 17, for which the duals will be needed
		for i in range(n):
			for j in range(i+1, n):
				m.addConstraint(w[i, j] == w[j, i])
				con16.append(constraint_count)
				constraint_count+=1
		for j in range(n):
			for i in range(n):
				if(i == j):
					continue
				# con 17
				m.addConstraint(y[i, j] == x[i]-w[i, j])
				con17.append(constraint_count)
				constraint_count+=1

		if type(quad) is Knapsack:
			# Multiply knapsack constraints by each x_j and (1-x_j)
			# Note: no need to include original knapsack constraints
			for k in range(quad.m):
				for j in range(n):
					#con 12
					m.addConstraint(xp.Sum(a[k][i]*w[i,j] for i in range(n) if i!=j)<=(b[k]-a[k][j])*x[j])
					#con 14
					m.addConstraint(xp.Sum(a[k][i]*y[i,j] for i in range(n) if i!=j)<=b[k]*(1-x[j]))

		#k_item constraint if necessary (if KQKP or HSP)
		if quad.num_items > 0:
			# Multiply partition constraint by each x_j
			# Note: There is no need to multiple by each (1-x_j), but must include original constraints
			m.addConstraint(xp.Sum(x[i] for i in range(n)) == quad.num_items)
			for j in range(n):
				m.addConstraint(xp.Sum(w[i,j] for i in range(n) if i!=j)== (quad.num_items-1)*x[j])

		for j in range(n):
			for i in range(n):
				if(i == j):
					continue
				# con 13 (w>=0 implied) - imp to add anyways?
				m.addConstraint(w[i, j] <= x[j])
				# con 15 (y>=0 implied)
				m.addConstraint(y[i, j] <= 1-x[j])

		quadratic_values = 0
		for j in range(n):
			for i in range(n):
				if(i == j):
					continue
				quadratic_values = quadratic_values + (C[i, j]*w[i, j])

		linear_values = xp.Sum(x[j]*c[j] for j in range(n))
		m.setObjective(linear_values + quadratic_values, sense=xp.maximize)


		# return model, and indices for constraint duals
		return [m, con16, con17]


	n = quad.n
	c = quad.c
	C = quad.C
	a = quad.a
	b = quad.b

	# model with continuous relaxed rlt1 and get duals to constraints 16,17
	m = rlt1_linearization(quad)
	start = timer()
	m[0].solve()
	d = m[0].getDual()
	con16indices = m[1]
	con17indices = m[2]
	print(m[0].getObjVal())     #this should be continuous relax solution to glover_ext.
	con16duals = d[con16indices[0]:con16indices[-1]+1]
	#con17duals = d[con17indices[0]:con17indices[-1]+1]
	con17duals = [d[index] for index in con17indices]

	duals16 = np.zeros((n, n))
	duals17 = np.zeros((n, n))
	dualcount = 0
	for i in range(n):
		for j in range(i+1, n):
			duals16[i][j] = con16duals[dualcount]
			dualcount+=1
	dualcount = 0
	for j in range(n):
		for i in range(n):
			if i == j:
				continue
			duals17[i][j] = con17duals[dualcount]
			dualcount+=1

	Cbar = np.zeros((n, n))
	Chat = np.zeros((n, n))
	# optimal split, found using dual vars from rlt1 continuous relaxation
	for i in range(n):
		for j in range(n):
			if i == j:
				continue
			if i < j:
				Cbar[i, j] = C[i, j]-duals16[i, j]-duals17[i, j]
				Chat[i, j] = -duals17[i,j]
			if i > j:
				Cbar[i, j] = C[i, j]+duals16[j, i]-duals17[i, j]
				Chat[i, j] = -duals17[i,j]
	#Chat = -duals17

	# update linear values as well
	for j in range(n):
		c[j] = c[j] + sum(duals17[j, i] for i in range(n) if i!=j)

	# simple split (this works but is not optimal)
	# for i in range(n):
		# for j in range(i+1,n):
		# D[i,j] = C[i,j]/4
		# D[j,i] = C[i,j]/4
		# E[i,j] = C[i,j]/4
		# E[j,i] = C[i,j]/4

	#create model and add variables
	m = xp.problem(name='glovers_linearization_rlt_'+bounds+'_'+constraints)
	x = np.array([xp.var(vartype=xp.binary) for i in range(n)])
	m.addVariable(x)

	if type(quad) is Knapsack: #HSP and UQP don't have cap constraint
		#add capacity constraint(s)
		for k in range(quad.m):
			m.addConstraint(xp.Sum(a[k][i]*x[i] for i in range(n)) <= b[k])

	#k_item constraint if necessary (if KQKP or HSP)
	if quad.num_items > 0:
		m.addConstraint(xp.Sum(x[i] for i in range(n)) == quad.num_items)

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
		bound_m = xp.problem(name='bound_model')
		bound_m.setlogfile("xpress.log")
		bound_x = np.array([xp.var(vartype=xp.continuous, lb=0, ub=1) for i in range(n)])
		bound_m.addVariable(bound_x)

		# Add in original structural constraints
		if type(quad) is Knapsack: #HSP and UQP don't have cap constraint
			# Include knapsack constraints
			for k in range(quad.m):
				bound_m.addConstraint(xp.Sum(a[k][i]*bound_x[i] for i in range(n)) <= b[k])

		#k_item constraint if necessary (if KQKP or HSP)
		if quad.num_items > 0:
			bound_m.addConstraint(xp.Sum(bound_x[i] for i in range(n)) == quad.num_items)

		for j in range(n):
			# Solve for Ubar1
			bound_m.setObjective(xp.Sum(Cbar[i,j]*bound_x[i] for i in range(n) if i!=j), sense=xp.maximize)
			xEquals1 = bound_x[j] == 1
			bound_m.addConstraint(xEquals1)
			bound_m.solve()
			if "optimal" not in str(bound_m.getProbStatusString()):
				# in case the xEquals1 constraint makes problem infeasible. (happens with kitem sometimes)
				bound_m.delConstraint(xEquals1)
				bound_m.addConstraint(bound_x[j]==0)
				m.addConstraint(x[j]==0)
				bound_m.solve()
			else:
				bound_m.delConstraint(xEquals1)
			Ubar1[j] = bound_m.getObjVal()

			# Solve for Lbar0
			bound_m.setObjective(xp.Sum(Cbar[i,j]*bound_x[i] for i in range(n) if i!=j), sense=xp.minimize)
			xEquals0 = bound_x[j]==0
			bound_m.addConstraint(xEquals0)
			bound_m.solve()
			if "optimal" not in str(bound_m.getProbStatusString()):
				# in case the xEquals1 constraint makes problem infeasible. (happens with kitem sometimes)
				bound_m.delConstraint(xEquals0)
				bound_m.addConstraint(bound_x[j]==1)
				m.addConstraint(x[j]==1)
				bound_m.solve()
			else:
				bound_m.delConstraint(xEquals0)
			Lbar0[j] = bound_m.getObjVal()

			# Solve for Uhat0
			bound_m.setObjective(xp.Sum(Chat[i,j]*bound_x[i] for i in range(n) if i!=j), sense=xp.maximize)
			xEquals0 = bound_x[j]==0
			bound_m.addConstraint(xEquals0)
			bound_m.solve()
			if "optimal" not in str(bound_m.getProbStatusString()):
				# in case the xEquals1 constraint makes problem infeasible. (happens with kitem sometimes)
				bound_m.delConstraint(xEquals0)
				bound_m.addConstraint(bound_x[j]==1)
				m.addConstraint(x[j]==1)
				bound_m.solve()
			else:
				bound_m.delConstraint(xEquals0)
			Uhat0[j] = bound_m.getObjVal()

			# Solve for Lhat1
			bound_m.setObjective(xp.Sum(Chat[i,j]*bound_x[i] for i in range(n) if i!=j), sense=xp.minimize)
			xEquals1 = bound_x[j]==1
			bound_m.addConstraint(xEquals1)
			bound_m.solve()
			if "optimal" not in str(bound_m.getProbStatusString()):
				# in case the xEquals1 constraint makes problem infeasible. (happens with kitem sometimes)
				bound_m.delConstraint(xEquals1)
				bound_m.addConstraint(bound_x[j]==0)
				m.addConstraint(x[j]==0)
				bound_m.solve()
			else:
				bound_m.delConstraint(xEquals1)
			Lhat1[j] = bound_m.getObjVal()
	else:
		raise Exception(bounds + " is not a valid bound type for glovers")

	end = timer()
	setup_time = end-start

	if(constraints=="original"):
		z1 = np.array([xp.var(vartype=xp.continuous, lb=-xp.infinity) for i in range(n)])
		z2 = np.array([xp.var(vartype=xp.continuous, lb=-xp.infinity) for i in range(n)])
		m.addVariable(z1,z2)

		for j in range(n):
			m.addConstraint(z1[j] <= Ubar1[j]*x[j])
			tempsum1 = sum(Cbar[i,j]*x[i] for i in range(n) if i!=j)
			m.addConstraint(z1[j] <= tempsum1 - Lbar0[j]*(1-x[j]))

			m.addConstraint(z2[j] <= Uhat0[j]*(1-x[j]))
			tempsum2 = sum(Chat[i,j]*x[i] for i in range(n) if i!=j)
			m.addConstraint(z2[j] <= tempsum2 - (Lhat1[j]*x[j]))

		# Set up the objective function
		m.setObjective(xp.Sum(c[j]*x[j] + z1[j] + z2[j] for j in range(n)), sense=xp.maximize)
	else:
		raise Exception(constraints + " is not a valid constraint type for glovers")

	#return model
	return [m,setup_time]

def extended_linear_formulation(quad, **kwargs):
	start = timer()
	n = quad.n
	c = quad.c
	C = quad.C
	a = quad.a
	b = quad.b

	#create model and add variables
	m = xp.problem(name='extended_linear_formulation')
	x = np.array([xp.var(vartype=xp.binary) for i in range(n)])
	z = np.array([[[xp.var(vartype=xp.continuous, lb=-xp.infinity) for i in range(n)] for j in range(n)] for k in range(n)])
	m.addVariable(x,z)

	if type(quad) is Knapsack: #HSP and UQP don't have cap constraint
		#add capacity constraint(s)
		for k in range(quad.m):
			m.addConstraint(xp.Sum(x[i]*a[k][i] for i in range(n)) <= b[k])

	#k_item constraint if necessary (if KQKP or HSP)
	if quad.num_items > 0:
		m.addConstraint(xp.Sum(x[i] for i in range(n)) == quad.num_items)

	#add auxiliary constraints
	for i in range(n):
		for j in range(i+1,n):
			m.addConstraint(z[i,i,j]+z[j,i,j] <= 1)
			if C[i,j] < 0:
				m.addConstraint(x[i] + z[i,i,j] <= 1)
				m.addConstraint(x[j] + z[j,i,j] <= 1)
			elif C[i,j] > 0:
				m.addConstraint(x[i] + z[i,i,j] + z[j,i,j] >= 1)
				m.addConstraint(x[j] + z[i,i,j] + z[j,i,j] >= 1)

	#compute quadratic values contirbution to obj
	constant = 0
	quadratic_values = 0
	for i in range(n):
		for j in range(i+1,n):
			constant = constant + C[i,j]
			quadratic_values = quadratic_values + (C[i,j]*(z[i,i,j]+z[j,i,j]))
	#set objective function
	linear_values = xp.Sum(x[i]*c[i] for i in range(n))
	m.setObjective(linear_values + constant - quadratic_values, sense=xp.maximize)

	end = timer()
	setup_time = end-start
	#return model + setup time
	return [m, setup_time]

def ss_linear_formulation(quad, **kwargs):
	"""
	Sherali-Smith Linear Formulation
	"""
	start = timer()
	n = quad.n
	c = quad.c
	C = quad.C
	a = quad.a
	b = quad.b

	#create model and add variables
	m = xp.problem(name='ss_linear_formulation')
	x = np.array([xp.var(vartype=xp.binary) for i in range(n)])
	s = np.array([xp.var(vartype=xp.continuous) for i in range(n)])
	y = np.array([xp.var(vartype=xp.continuous) for i in range(n)])
	m.addVariable(x,y,s)

	if type(quad) is Knapsack: #HSP and UQP don't have cap constraint
		#add capacity constraint(s)
		for k in range(quad.m):
			m.addConstraint(xp.Sum(x[i]*a[k][i] for i in range(n)) <= b[k])

	#k_item constraint if necessary (if KQKP or HSP)
	if quad.num_items > 0:
		m.addConstraint(xp.Sum(x[i] for i in range(n)) == quad.num_items)

	U = np.zeros(n)
	L = np.zeros(n)
	bound_m = xp.problem(name='bound_model')
	bound_m.setlogfile("xpress.log")
	bound_x = np.array([xp.var(vartype=xp.continuous, lb=0, ub=1) for i in range(n)])
	bound_m.addVariable(bound_x)
	if type(quad) is Knapsack:
		for k in range(quad.m):
			#add capacity constraints
			bound_m.addConstraint(xp.Sum(bound_x[i]*a[k][i] for i in range(n)) <= b[k])
	if quad.num_items > 0:
		bound_m.addConstraint(xp.Sum(bound_x[i] for i in range(n)) == quad.num_items)
	for i in range(n):
		#for each col, solve model to find upper/lower bound
		bound_m.setObjective(xp.Sum(C[i,j]*bound_x[j] for j in range(n)), sense=xp.maximize)
		bound_m.solve()
		U[i] = bound_m.getObjVal()

		bound_m.setObjective(xp.Sum(C[i,j]*bound_x[j] for j in range(n)), sense=xp.minimize)
		bound_m.solve()
		L[i] = bound_m.getObjVal()

	#add auxiliary constraints
	for i in range(n):
		m.addConstraint(sum(C[i,j]*x[j] for j in range(n))-s[i]-L[i]==y[i])
		m.addConstraint(y[i] <= (U[i]-L[i])*(1-x[i]))
		m.addConstraint(s[i] <= (U[i]-L[i])*x[i])
		m.addConstraint(y[i] >= 0)
		m.addConstraint(s[i] >= 0)

	#set objective function
	m.setObjective(sum(s[i]+x[i]*(c[i]+L[i])for i in range(n)), sense=xp.maximize)

	end = timer()
	setup_time = end-start
	#return model + setup time
	return [m, setup_time]

def qsap_standard(qsap, **kwargs):
	n = qsap.n
	m = qsap.m
	e = qsap.e
	c = qsap.c

	#create model and add variables
	mdl = xp.problem(name='qsap_standard_linearization')
	x = np.array([[xp.var(vartype=xp.binary) for i in range(n)]for j in range(m)])
	w = np.array([[[[xp.var(vartype=xp.continuous) for i in range(n)]for j in range(m)]
						for k in range(n)]for l in range(m)])
	mdl.addVariable(x,w)
	mdl.addConstraint((sum(x[i,k] for k in range(n)) == 1) for i in range(m))

	#add auxiliary constraints
	#TODO implement lhs here?
	for i in range(m-1):
		for k in range(n):
			for j in range(i+1,m):
				for l in range(n):
					mdl.addConstraint(w[i,k,j,l] <= x[i,k])
					mdl.addConstraint(w[i,k,j,l] <= x[j,l])
					mdl.addConstraint(x[i,k] + x[j,l] - 1 <= w[i,k,j,l])
					mdl.addConstraint(w[i,k,j,l] >= 0)

	#compute quadratic values contirbution to obj
	quadratic_values = 0
	for i in range(m-1):
		for j in range(i+1,m):
			for k in range(n):
				for l in range(n):
					quadratic_values = quadratic_values + (c[i,k,j,l]*(w[i,k,j,l]))

	linear_values = sum(x[i,k]*e[i,k] for k in range(n) for i in range(m))
	mdl.setObjective(linear_values + quadratic_values, sense=xp.maximize)

	#return model. no setup time for std
	return [mdl, 0]

def qsap_glovers(qsap, bounds="original", constraints="original", lhs_constraints=False, **kwargs):
	start = timer()
	n = qsap.n
	m = qsap.m
	e = qsap.e
	c = qsap.c
	mdl = xp.problem(name='qsap_glovers')
	#x = mdl.addVars(m,n,name="binary_var", vtype=GRB.BINARY)
	#TODO possibly need to flip order of loops and/or reshape here
	x = np.array([xp.var(vartype=xp.binary) for i in range(m) for j in range(n)]).reshape(m,n)
	mdl.addVariable(x)
	mdl.addConstraint((sum(x[i,k] for k in range(n)) == 1) for i in range(m))
	#let gurobi solve w/ quadratic objective function
	# mdl.setObjective(sum(sum(e[i,k]*x[i,k] for k in range(n))for i in range(m))
	# 			+ sum(sum(sum(sum(c[i,k,j,l]*x[i,k]*x[j,l] for l in range(n))for k in range(n))
	# 			for j in range(1+i,m)) for i in range(m-1)), sense=xp.maximize)
	# mdl.solve()
	# print(mdl.getObjVal())


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
		bound_mdl = xp.problem(name="bound_m")
		bound_mdl.setlogfile("xpress.log")
		if bounds=="tight":
			bound_x = np.array([xp.var(vartype=xp.continuous, lb=0, ub=1) for i in range(m) for j in range(n)]).reshape(m,n)
		elif bounds == "tighter":
			bound_x = np.array([xp.var(vartype=xp.binary, lb=0, ub=1) for i in range(m) for j in range(n)]).reshape(m,n)
		bound_mdl.addVariable(bound_x)
		bound_mdl.addConstraint((xp.Sum(bound_x[i,k] for k in range(n)) == 1) for i in range(m))
		for i in range(m-1):
			for k in range(n):
				bound_mdl.setObjective(xp.Sum(xp.Sum(c[i,k,j,l]*bound_x[j,l] for l in range(n)) for j in range(i+1,m)), sense=xp.maximize)
				u_con = bound_x[i,k] == 1
				bound_mdl.addConstraint(u_con)
				bound_mdl.solve()
				bound_mdl.delConstraint(u_con)
				U1[i,k] = bound_mdl.getObjVal()

				bound_mdl.setObjective(xp.Sum(xp.Sum(c[i,k,j,l]*bound_x[j,l] for l in range(n)) for j in range(i+1,m)), sense=xp.minimize)
				l_con = bound_x[i,k] == 0
				bound_mdl.addConstraint(l_con)
				bound_mdl.solve()
				bound_mdl.delConstraint(l_con)
				L0[i,k] = bound_mdl.getObjVal()

				if lhs_constraints:
					bound_mdl.setObjective(xp.Sum(xp.Sum(c[i,k,j,l]*bound_x[j,l] for l in range(n)) for j in range(i+1,m)), sense=xp.maximize)
					u_con = bound_x[i,k] == 0
					bound_mdl.addConstraint(u_con)
					bound_mdl.solve()
					bound_mdl.delConstraint(u_con)
					U0[i,k] = bound_mdl.getObjVal()

					bound_mdl.setObjective(xp.Sum(xp.Sum(c[i,k,j,l]*bound_x[j,l] for l in range(n)) for j in range(i+1,m)), sense=xp.minimize)
					l_con = bound_x[i,k] == 1
					bound_mdl.addConstraint(l_con)
					bound_mdl.solve()
					bound_mdl.delConstraint(l_con)
					L1[i,k] = bound_mdl.getObjVal()
	else:
		raise Exception(bounds + " is not a valid bound type for glovers")

	#add auxiliary constrains
	if constraints=="original":
		z = np.array([xp.var(vartype=xp.continuous, lb=-xp.infinity) for i in range(m) for j in range(n)]).reshape(m,n)
		mdl.addVariable(z)
		mdl.addConstraint(z[i,k] <= x[i,k]*U1[i,k] for i in range(m-1) for k in range(n))
		mdl.addConstraint(z[i,k] <= xp.Sum(xp.Sum(c[i,k,j,l]*x[j,l] for l in range(n)) for j in range(i+1,m))
										-L0[i,k]*(1-x[i,k]) for i in range(m-1) for k in range(n))
		if lhs_constraints:
			mdl.addConstraint(z[i,k] >= x[i,k]*L1[i,k] for i in range(m-1) for k in range(n))
			mdl.addConstraint(z[i,k] >= xp.Sum(xp.Sum(c[i,k,j,l]*x[j,l] for l in range(n)) for j in range(i+1,m))
										-U0[i,k]*(1-x[i,k]) for i in range(m-1) for k in range(n))
		mdl.setObjective(xp.Sum(xp.Sum(e[i,k]*x[i,k] for k in range(n))for i in range(m))
					+ xp.Sum(xp.Sum(z[i,k] for k in range(n)) for i in range(m-1)), sense=xp.maximize)
	elif constraints=="sub1":
		s = np.array([xp.var(vartype=xp.continuous, lb=0) for i in range(m) for j in range(n)]).reshape(m,n)
		mdl.addVariable(s)
		mdl.addConstraint(s[i,k] >= U1[i,k]*x[i,k]+L0[i,k]*(1-x[i,k])-xp.Sum(xp.Sum(c[i,k,j,l]*x[j,l] for l in range(n)) for j in range(i+1,m))
						for k in range(n) for i in range(m-1))
		mdl.setObjective(xp.Sum(xp.Sum(e[i,k]*x[i,k] for k in range(n))for i in range(m))
					+ xp.Sum(xp.Sum(U1[i,k]*x[i,k]-s[i,k] for k in range(n)) for i in range(m-1)), sense=xp.maximize)
	elif constraints=="sub2":
		s = np.array([xp.var(vartype=xp.continuous, lb=0) for i in range(m) for j in range(n)]).reshape(m,n)
		mdl.addVariable(s)
		mdl.addConstraint(s[i,k] >= -L0[i,k]*(1-x[i,k])-(x[i,k]*U1[i,k])+xp.Sum(xp.Sum(c[i,k,j,l]*x[j,l] for l in range(n)) for j in range(i+1,m))
		 				for k in range(n) for i in range(m-1))
		mdl.setObjective(xp.Sum(xp.Sum(e[i,k]*x[i,k] for k in range(n))for i in range(m))
					+ xp.Sum(xp.Sum(-s[i,k]-(L0[i,k]*(1-x[i,k])) + xp.Sum(xp.Sum(c[i,k,j,l]*x[j,l] for l in range(n))
					for j in range(i+1,m)) for k in range(n)) for i in range(m-1)), sense=xp.maximize)
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
	mdl = xp.problem(name='qsap_elf')
	x = np.array([[xp.var(vartype=xp.binary) for i in range(n)]for j in range(m)])
	z = np.array([[[[[[xp.var(vartype=xp.continuous, lb=-xp.infinity) for i in range(n)]
		for j in range(m)] for k in range(n)] for l in range(m)] for r in range(n)] for s in range(m)])
	mdl.addVariable(x, z)

	cons = [xp.constraint(sum(x[i,k] for k in range(n)) == 1) for i in range(m)]
	mdl.addConstraint(cons)

	#add auxiliary constraints
	for i in range(m-1):
		for j in range(i+1,m):
			for k in range(n):
				for l in range(n):
					mdl.addConstraint(z[i,k,i,k,j,l] + z[j,l,i,k,j,l] <= 1)
					mdl.addConstraint(x[i,k] + z[i,k,i,k,j,l] <= 1)
					mdl.addConstraint(x[j,l] + z[j,l,i,k,j,l] <= 1)
					mdl.addConstraint(x[i,k] + z[i,k,i,k,j,l] + z[j,l,i,k,j,l] >= 1)
					mdl.addConstraint(x[j,l] + z[i,k,i,k,j,l] + z[j,l,i,k,j,l] >= 1)

	#compute quadratic values contirbution to obj
	constant = 0
	quadratic_values = 0
	for i in range(m-1):
		for j in range(i+1,m):
			for k in range(n):
				for l in range(n):
					constant = constant + c[i,k,j,l]
					quadratic_values = quadratic_values + (c[i,k,j,l]*(z[i,k,i,k,j,l]+z[j,l,i,k,j,l]))

	linear_values = sum(x[i,k]*e[i,k] for k in range(n) for i in range(m))
	mdl.setObjective(linear_values + constant - quadratic_values, sense=xp.maximize)

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
	mdl = xp.problem(name='qsap_ss')
	x = np.array([[xp.var(vartype=xp.binary) for i in range(n)]for j in range(m)])
	s = np.array([[xp.var(vartype=xp.continuous) for i in range(n)]for j in range(m)])
	y = np.array([[xp.var(vartype=xp.continuous) for i in range(n)]for j in range(m)])
	mdl.addVariable(x,s,y)
	mdl.addConstraint((sum(x[i,k] for k in range(n)) == 1) for i in range(m))

	start = timer()
	U = np.zeros((m,n))
	L = np.zeros((m,n))
	bound_mdl = xp.problem(name='upper_bound_model')
	bound_mdl.setlogfile("xpress.log")
	bound_x = np.array([xp.var(vartype=xp.continuous, lb=0, ub=1) for i in range(m) for j in range(n)]).reshape(m,n)
	bound_mdl.addVariable(bound_x)
	bound_mdl.addConstraint((sum(bound_x[i,k] for k in range(n)) == 1) for i in range(m))
	for i in range(m-1):
		for k in range(n):
			bound_mdl.setObjective(sum(sum(c[i,k,j,l]*bound_x[j,l] for l in range(n)) for j in range(i+1,m)), sense=xp.maximize)
			bound_mdl.solve()
			U[i,k] = bound_mdl.getObjVal()

			bound_mdl.setObjective(sum(sum(c[i,k,j,l]*bound_x[j,l] for l in range(n)) for j in range(i+1,m)), sense=xp.minimize)
			bound_mdl.solve()
			L[i,k] = bound_mdl.getObjVal()
	end = timer()
	setup_time = end-start

	#add auxiliary constraints
	for i in range(m-1):
		for k in range(n):
			mdl.addConstraint(sum(sum(c[i,k,j,l]*x[j,l] for j in range(i+1,m)) for l in range(n))-s[i,k]-L[i,k]==y[i,k])
			mdl.addConstraint(y[i,k] <= (U[i,k]-L[i,k])*(1-x[i,k]))
			mdl.addConstraint(s[i,k] <= (U[i,k]-L[i,k])*x[i,k])
			mdl.addConstraint(y[i,k] >= 0)
			mdl.addConstraint(s[i,k] >= 0)

	#set objective function
	linear_values = sum(sum(e[i,k]*x[i,k] for i in range(m)) for k in range(n))
	mdl.setObjective(linear_values + sum(sum(s[i,k]+x[i,k]*(L[i,k]) for i in range(m-1)) for k in range(n)), sense=xp.maximize)

	#return model + setup time
	return [mdl, setup_time]

def no_linearization(quad, **kwargs):
	start = timer()
	n = quad.n
	c = quad.c
	m = xp.problem(name='no_linearization')
	x = np.array([xp.var(vartype=xp.binary) for i in range(n)])
	m.addVariable(x)

	if type(quad) is Knapsack: #HSP and UQP don't have cap constraint
		#add capacity constraint(s)
		for k in range(quad.m):
			m.addConstraint(xp.Sum(x[i]*quad.a[k][i] for i in range(n)) <= quad.b[k])

	#k_item constraint if necessary (if KQKP or HSP)
	if quad.num_items > 0:
		m.addConstraint(xp.Sum(x[i] for i in range(n)) == quad.num_items)

	#compute quadratic values contirbution to obj
	quadratic_values = 0
	for i in range(n):
		for j in range(i+1,n):
			quadratic_values = quadratic_values + (x[i]*x[j]*quad.C[i,j])
	#set objective function
	linear_values = xp.Sum(x[i]*c[i] for i in range(n))
	m.setObjective(linear_values + quadratic_values, sense=xp.maximize)

	end = timer()
	setup_time = end-start
	return [m, setup_time]

def solve_model(m, solve_relax=True, **kwargs):
	"""
	Takes in an unsolved gurobi model of a MIP. Solves it as well as continuous
	relaxation and returns a dictionary containing relevant solve details
	"""
	# turn off model output. otherwise prints bunch of info, clogs console
	m.setlogfile("xpress.log")
	time_limit = False
	# start timer and solve model
	m.controls.maxtime = 600
	start = timer()
	m.solve()
	end = timer()
	if("optimal" not in str(m.getProbStatusString())):
		print('time limit exceeded')
		time_limit=True
	solve_time = end-start
	objective_value = m.getObjVal()


	if solve_relax and not time_limit:
		# relax and solve to get continuous relaxation and integrality_gap
		#cols = []
		#m.getcoltype(cols,0,m.attributes.cols-1)
		#print(cols)
		#cols = np.asarray(cols)
		#bin_cols = [cols == 'B']
		#print(bin_cols)
		#m.chgcoltype(range(10),['C','C','C','C','C','C','C','C','C','C'])
		vars = m.getVariable()
		#TODO this is inneficient
		for index,var in enumerate(vars):
			if var.vartype == 1: #1 means binary
				m.chgcoltype([index],['C'])

		m.solve()
		continuous_obj_value = m.getObjVal()
		integrality_gap = ((continuous_obj_value-objective_value)/objective_value)*100
	else:
		continuous_obj_value = -1
		integrality_gap = -1
	# terminate model so not allocating resources
	#m.close()


	# create and return results dictionary
	results = {"solve_time": solve_time,
			   "objective_value": objective_value,
			   "relaxed_solution": continuous_obj_value,
			   "integrality_gap": integrality_gap,
				"time_limit":time_limit}
	return results
