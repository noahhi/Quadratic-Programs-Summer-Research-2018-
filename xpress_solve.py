from quadratics import *
import numpy as np
from timeit import default_timer as timer
import sys
sys.path.insert(0,"c:/xpressmp/bin")
import xpress as xp

#xp.addcbmsghandler(None,"xpress.log",0)

def standard_linearization(quad, lhs_constraints=True, **kwargs):
	start = timer()
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
	if type(quad) == HSP:
		#HSP doesn't habe any linear terms
		m.setObjective(quadratic_values, sense=xp.maximize)
	else:
		linear_values = sum(x[i]*c[i] for i in range(n))
		m.setObjective(linear_values + quadratic_values, sense=xp.maximize)

	end = timer()
	setup_time = end-start
	# return model + setup time
	return [m, setup_time]

def glovers_linearization(quad, bounds="tight", constraints="original", lhs_constraints=False, use_diagonal=False, **kwargs):
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
		u_bound_m = xp.problem(name='upper_bound_model')
		u_bound_m.setlogfile("xpress.log")
		l_bound_m = xp.problem(name='lower_bound_model')
		l_bound_m.setlogfile("xpress.log")
		if bounds=="tight":
			u_bound_x = np.array([xp.var(vartype=xp.continuous, lb=0, ub=1) for i in range(n)])
			l_bound_x = np.array([xp.var(vartype=xp.continuous, lb=0, ub=1) for i in range(n)])
		elif bounds=="tighter":
			u_bound_x = np.array([xp.var(vartype=xp.binary) for i in range(n)])
			l_bound_x = np.array([xp.var(vartype=xp.binary) for i in range(n)])
		u_bound_m.addVariable(u_bound_x)
		l_bound_m.addVariable(l_bound_x)
		if type(quad) is Knapsack:
			for k in range(quad.m):
				#add capacity constraints
				u_bound_m.addConstraint(xp.Sum(u_bound_x[i]*a[k][i] for i in range(n)) <= b[k])
				l_bound_m.addConstraint(xp.Sum(l_bound_x[i]*a[k][i] for i in range(n)) <= b[k])
		if quad.num_items > 0:
			u_bound_m.addConstraint(xp.Sum(u_bound_x[i] for i in range(n)) == quad.num_items)
			l_bound_m.addConstraint(xp.Sum(l_bound_x[i] for i in range(n)) == quad.num_items)
		for j in range(n):
			#for each col, solve model to find upper/lower bound
			u_bound_m.setObjective(xp.Sum(C[i, j]*u_bound_x[i] for i in range(n)), sense=xp.maximize)
			l_bound_m.setObjective(xp.Sum(C[i, j]*l_bound_x[i] for i in range(n)), sense=xp.minimize)
			u_con = u_bound_x[j] == 1
			u_bound_m.addConstraint(u_con)
			l_con = l_bound_x[j] == 0
			l_bound_m.addConstraint(l_con)
			u_bound_m.solve()
			if "optimal" not in str(u_bound_m.getProbStatusString()):
				print("non-optimal solve status: " + str(u_bound_m.getProbStatusString()) + " when solving for upper bound (U1)")
				u_bound_m.delConstraint(u_con)
				u_bound_m.addConstraint(u_bound_x[j]==0)
				m.addConstraint(x[j]==0)
				u_bound_m.solve()
			l_bound_m.solve()
			if "optimal" not in str(l_bound_m.getProbStatusString()):
				print("non-optimal solve status: " + str(l_bound_m.getProbStatusString()) + " when solving for lower bound (L0)")
				l_bound_m.delConstraint(l_con)
				l_bound_m.addConstraint(l_bound_x[j]==1)
				m.addConstraint(x[j]==1)
				l_bound_m.solve()
			U1[j] = u_bound_m.getObjVal()
			L0[j] = l_bound_m.getObjVal()
			u_bound_m.delConstraint(u_con)
			l_bound_m.delConstraint(l_con)
			if lhs_constraints:
				u_con = u_bound_x[j] == 0
				u_bound_m.addConstraint(u_con)
				l_con = l_bound_x[j] == 1
				l_bound_m.addConstraint(l_con)
				u_bound_m.solve()
				if "optimal" not in str(u_bound_m.getProbStatusString()):
					print("non-optimal solve status: " + str(u_bound_m.status) + " when solving for upper bound (U0)")
					u_bound_m.delConstraint(u_con)
					u_bound_m.addConstraint(u_bound_x[j]==1)
					m.addConstraint(x[j]==1)
					u_bound_m.solve()
				l_bound_m.solve()
				if "optimal" not in str(l_bound_m.getProbStatusString()):
					print("non-optimal solve status: " + str(l_bound_m.status) + " when solving for lower bound (L1)")
					l_bound_m.delConstraint(l_con)
					l_bound_m.addConstraint(l_bound_x[j]==0)
					m.addConstraint(x[j]==0)
					l_bound_m.solve()
				U0[j] = u_bound_m.getObjVal()
				L1[j] = l_bound_m.getObjVal()
				u_bound_m.delConstraint(u_con)
				l_bound_m.delConstraint(l_con)
	else:
		raise Exception(bounds + " is not a valid bound type for glovers")

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
		if type(quad) is HSP:
			m.setObjective(xp.Sum(z[j] for j in range(n)), sense=xp.maximize)
		else:
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
		if type(quad) is HSP:
			m.setObjective(quadratic_values, sense=xp.maximize)
		else:
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

#TODO glovers_rlt currently works only for knapsack w/ original constraints
def glovers_linearization_rlt(quad, bounds="tight", constraints="original"):
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
				# con 16
				#m.addConstraint(w[i, j] == w[j, i], name='con16'+str(i)+str(j))
				m.addConstraint(w[i, j] == w[j, i])

		for k in range(quad.m):
			for j in range(n):
				# con 12
				m.addConstraint(xp.Sum(a[k][i]*w[i, j] for i in range(n) if i != j) <= (b[k]-a[k][j])*x[j])
				# con 14
				m.addConstraint(xp.Sum(a[k][i]*y[i, j] for i in range(n) if i != j) <= b[k]*(1-x[j]))

		for j in range(n):
			for i in range(n):
				if(i == j):
					continue
				# con 13 (w>=0 implied) - imp to add anyways?
				m.addConstraint(w[i, j] <= x[j])
				# con 15 (y>=0 implied)
				m.addConstraint(y[i, j] <= 1-x[j])
				# con 17
				#m.addConstraint(y[i, j] == x[i]-w[i, j], name='con17'+str(i)+str(j))
				m.addConstraint(y[i, j] == x[i]-w[i, j])

		quadratic_values = 0
		for j in range(n):
			for i in range(n):
				if(i == j):
					continue
				quadratic_values = quadratic_values + (C[i, j]*w[i, j])
		if type(quad) is HSP:
			m.setObjective(quadratic_values, sense=xp.maximize)
		else:
			linear_values = xp.Sum(x[j]*c[j] for j in range(n))
			m.setObjective(linear_values + quadratic_values, sense=xp.maximize)


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
	m.solve()
	d = m.getDual()
	print(d)
	#print(m.getObjVal())     #this should be continuous relax solution to glover_ext.
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
		c[j] = c[j] + xp.Sum(duals17[j, i] for i in range(n))

	# simple split (this works but is not optimal)
	# for i in range(n):
		# for j in range(i+1,n):
		# D[i,j] = C[i,j]/4
		# D[j,i] = C[i,j]/4
		# E[i,j] = C[i,j]/4
		# E[j,i] = C[i,j]/4

	# create model and add variables
	m = xp.problem(name='glovers_linearization_ext_'+bounds+'_'+constraints)
	x = m.addVars(n, name="binary_var", vtype=GRB.BINARY)

	if type(quad) is Knapsack:  # HSP and UQP don't have cap constraint
		# add capacity constraint(s)
		for k in range(quad.m):
			m.addConstraint(xp.Sum(x[i]*a[k][i] for i in range(n)) <= b[k])

	#k_item constraint if necessary (if KQKP or HSP)
	if quad.num_items > 0:
		m.addConstraint(xp.Sum(x[i] for i in range(n)) == quad.num_items)

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
		u_bound_m1 = xp.problem(name='upper_bound_model1')
		l_bound_m1 = xp.problem(name='lower_bound_model1')
		u_bound_x1 = u_bound_m1.addVars(n, ub=1, lb=0.0)
		l_bound_x1 = l_bound_m1.addVars(n, ub=1, lb=0.0)
		for k in range(quad.m):
			u_bound_m1.addConstraint(xp.Sum(u_bound_x1[i]*a[k][i] for i in range(n)) <= b[k])
			l_bound_m1.addConstraint(xp.Sum(l_bound_x1[i]*a[k][i] for i in range(n)) <= b[k])
		u_bound_m2 = xp.problem(name='upper_bound_model2')
		l_bound_m2 = xp.problem(name='lower_bound_model2')
		u_bound_x2 = u_bound_m2.addVars(n, ub=1, lb=0)
		l_bound_x2 = l_bound_m2.addVars(n, ub=1, lb=0)
		for k in range(quad.m):
			u_bound_m2.addConstraint(xp.Sum(u_bound_x2[i]*a[k][i] for i in range(n)) <= b[k])
			l_bound_m2.addConstraint(xp.Sum(l_bound_x2[i]*a[k][i] for i in range(n)) <= b[k])

		for j in range(n):
			u_bound_m1.setObjective(xp.Sum(D[i, j]*u_bound_x1[i] for i in range(n) if i != j), sense=xp.maximize)
			l_bound_m1.setObjective(xp.Sum(D[i, j]*l_bound_x1[i] for i in range(n) if i != j), sense=xp.minimize)
			u_con1 = u_bound_m1.addConstraint(u_bound_x1[j] == 1)
			l_con1 = l_bound_m1.addConstraint(l_bound_x1[j] == 0)
			u_bound_m1.solve()
			l_bound_m1.solve()
			u_bound_m1.delConstraint(u_con1)
			l_bound_m1.delConstraint(l_con1)
			U1[j] = u_bound_m1.getObjVal()
			L1[j] = l_bound_m1.getObjVal()

			u_bound_m2.setObjective(xp.Sum(E[i, j]*u_bound_x2[i] for i in range(n) if i != j), sense=xp.maximize)
			l_bound_m2.setObjective(xp.Sum(E[i, j]*l_bound_x2[i] for i in range(n) if i != j), sense=xp.minimize)
			u_con2 = u_bound_m2.addConstraint(u_bound_x2[j] == 0)
			l_con2 = l_bound_m2.addConstraint(l_bound_x2[j] == 1)
			u_bound_m2.solve()
			l_bound_m2.solve()
			u_bound_m2.delConstraint(u_con2)
			l_bound_m2.delConstraint(l_con2)
			U2[j] = u_bound_m2.getObjVal()
			L2[j] = l_bound_m2.getObjVal()
	else:
		raise Exception(bounds + " is not a valid bound type for glovers")

	# add auxiliary constrains
	if(constraints == "original"):
		z1 = m.addVars(n, lb=-GRB.INFINITY)
		z2 = m.addVars(n, lb=-GRB.INFINITY)
		for j in range(n):
			m.addConstraint(z1[j] <= U1[j]*x[j])
			m.addConstraint(z2[j] <= U2[j]*(1-x[j]))
		for j in range(n):
			tempsum1 = xp.Sum(D[i, j]*x[i] for i in range(n) if i != j)
			m.addConstraint(z1[j] <= tempsum1 - L1[j]*(1-x[j]))
			tempsum2 = xp.Sum(E[i, j]*x[i] for i in range(n) if i != j)
			m.addConstraint(z2[j] <= tempsum2 - (L2[j]*x[j]))
		m.setObjective(xp.Sum(c[j]*x[j] + z1[j] + z2[j] for j in range(n)), sense=xp.maximize)
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
	m.setlogfile("xpress.log")
	time_limit = False
	# start timer and solve model
	m.controls.maxtime = 3600
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
	if type(quad)==HSP:
		#HSP doesn't habe any linear terms
		m.setObjective(quadratic_values, sense=xp.maximize)
	else:
		linear_values = xp.Sum(x[i]*c[i] for i in range(n))
		m.setObjective(linear_values + quadratic_values, sense=xp.maximize)
	end = timer()
	setup_time = end-start
	return [m, setup_time]

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
		u_bound_mdl = xp.problem(name="u_bound_m")
		u_bound_mdl.setlogfile("xpress.log")
		l_bound_mdl = xp.problem(name="l_bound_m")
		l_bound_mdl.setlogfile("xpress.log")
		if bounds=="tight":
			u_bound_x = np.array([xp.var(vartype=xp.continuous, lb=0, ub=1) for i in range(m) for j in range(n)]).reshape(m,n)
			l_bound_x = np.array([xp.var(vartype=xp.continuous, lb=0, ub=1) for i in range(m) for j in range(n)]).reshape(m,n)
		elif bounds == "tighter":
			u_bound_x = np.array([xp.var(vartype=xp.binary, lb=0, ub=1) for i in range(m) for j in range(n)]).reshape(m,n)
			l_bound_x = np.array([xp.var(vartype=xp.binary, lb=0, ub=1) for i in range(m) for j in range(n)]).reshape(m,n)
		u_bound_mdl.addVariable(u_bound_x)
		l_bound_mdl.addVariable(l_bound_x)
		u_bound_mdl.addConstraint((xp.Sum(u_bound_x[i,k] for k in range(n)) == 1) for i in range(m))
		l_bound_mdl.addConstraint((xp.Sum(l_bound_x[i,k] for k in range(n)) == 1) for i in range(m))
		for i in range(m-1):
			for k in range(n):
				u_bound_mdl.setObjective(xp.Sum(xp.Sum(c[i,k,j,l]*u_bound_x[j,l] for l in range(n)) for j in range(i+1,m)), sense=xp.maximize)
				l_bound_mdl.setObjective(xp.Sum(xp.Sum(c[i,k,j,l]*l_bound_x[j,l] for l in range(n)) for j in range(i+1,m)), sense=xp.minimize)
				u_con = u_bound_x[i,k] == 1
				u_bound_mdl.addConstraint(u_con)
				l_con = l_bound_x[i,k] == 0
				l_bound_mdl.addConstraint(l_con)
				u_bound_mdl.solve()
				l_bound_mdl.solve()
				U1[i,k] = u_bound_mdl.getObjVal()
				L0[i,k] = l_bound_mdl.getObjVal()
				u_bound_mdl.delConstraint(u_con)
				l_bound_mdl.delConstraint(l_con)
				if lhs_constraints:
					u_con = u_bound_x[i,k] == 0
					u_bound_mdl.addConstraint(u_con)
					l_con = l_bound_x[i,k] == 1
					l_bound_mdl.addConstraint(l_con)
					u_bound_mdl.solve()
					l_bound_mdl.solve()
					U0[i,k] = u_bound_mdl.getObjVal()
					L1[i,k] = l_bound_mdl.getObjVal()
					u_bound_mdl.delConstraint(u_con)
					l_bound_mdl.delConstraint(l_con)
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
	if type(quad)==HSP:
		#HSP doesn't habe any linear terms
		m.setObjective(constant-quadratic_values, sense=xp.maximize)
	else:
		linear_values = xp.Sum(x[i]*c[i] for i in range(n))
		m.setObjective(linear_values + constant - quadratic_values, sense=xp.maximize)

	end = timer()
	setup_time = end-start
	#return model + setup time
	return [m, setup_time]

def qsap_elf(qsap, **kwargs):
	start = timer()
	n = qsap.n
	m = qsap.m
	e = qsap.e
	c = qsap.c
	mdl = xp.problem(name='qsap_elf')
	x = np.array([[xp.var(vartype=xp.binary) for i in range(m)]for j in range(n)])
	z = np.array([[[[[[xp.var(vartype=xp.continuous, lb=-xp.infinity) for i in range(m)]
		for j in range(n)] for k in range(m)] for l in range(n)] for r in range(m)] for s in range(n)])
	mdl.addVariable(x, z)

	mdl.addConstraint((sum(x[i,k] for k in range(n)) == 1) for i in range(m))

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

	end = timer()
	setup_time = end-start
	#return model + setup time
	return [mdl, setup_time]

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
	u_bound_m = xp.problem(name='upper_bound_model')
	l_bound_m = xp.problem(name='lower_bound_model')
	u_bound_m.setlogfile("xpress.log")
	l_bound_m.setlogfile("xpress.log")
	u_bound_x = np.array([xp.var(vartype=xp.continuous, lb=0, ub=1) for i in range(n)])
	l_bound_x = np.array([xp.var(vartype=xp.continuous, lb=0, ub=1) for i in range(n)])
	u_bound_m.addVariable(u_bound_x)
	l_bound_m.addVariable(l_bound_x)
	if type(quad) is Knapsack:
		for k in range(quad.m):
			#add capacity constraints
			u_bound_m.addConstraint(xp.Sum(u_bound_x[i]*a[k][i] for i in range(n)) <= b[k])
			l_bound_m.addConstraint(xp.Sum(l_bound_x[i]*a[k][i] for i in range(n)) <= b[k])
	if quad.num_items > 0:
		u_bound_m.addConstraint(xp.Sum(u_bound_x[i] for i in range(n)) == quad.num_items)
		l_bound_m.addConstraint(xp.Sum(l_bound_x[i] for i in range(n)) == quad.num_items)
	for i in range(n):
		#for each col, solve model to find upper/lower bound
		u_bound_m.setObjective(xp.Sum(C[i,j]*u_bound_x[j] for j in range(n)), sense=xp.maximize)
		l_bound_m.setObjective(xp.Sum(C[i,j]*l_bound_x[j] for j in range(n)), sense=xp.minimize)
		u_bound_m.solve()
		l_bound_m.solve()
		U[i] = u_bound_m.getObjVal()
		L[i] = l_bound_m.getObjVal()

	#add auxiliary constraints
	for i in range(n):
		m.addConstraint(sum(C[i,j]*x[j] for j in range(n))-s[i]-L[i]==y[i])
		m.addConstraint(y[i] <= (U[i]-L[i])*(1-x[i]))
		m.addConstraint(s[i] <= (U[i]-L[i])*x[i])
		m.addConstraint(y[i] >= 0)
		m.addConstraint(s[i] >= 0)

	#set objective function
	if type(quad)==HSP:
		#HSP doesn't habe any linear terms
		m.setObjective(sum(s[i]+x[i]*(L[i])for i in range(n)), sense=xp.maximize)
	else:
		m.setObjective(sum(s[i]+x[i]*(c[i]+L[i])for i in range(n)), sense=xp.maximize)

	end = timer()
	setup_time = end-start
	#return model + setup time
	return [m, setup_time]

# p = QSAP()
# m = qsap_elf(p)[0]
# print(solve_model(m, solve_relax=True))


# p = Knapsack()
# p.print_info(print_C =True)
# m = extended_linear_formulation(p)[0]
# print(solve_model(m))
