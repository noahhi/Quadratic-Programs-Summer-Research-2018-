from quadratics import *
import numpy as np
from timeit import default_timer as timer
from docplex.mp.model import Model

def standard_linearization(quad, lhs_constraints=True, **kwargs):
	n = quad.n
	c = quad.c
	C = quad.C
	a = quad.a
	b = quad.b

	#create model and add variables
	m = Model(name='standard_linearization')
	x = m.binary_var_list(n, name="binary_var")
	w = m.continuous_var_matrix(keys1=n, keys2=n, lb=-m.infinity)

	if type(quad) is Knapsack: #HSP and UQP don't have cap constraint
		#add capacity constraint(s)
		for k in range(quad.m):
			m.add_constraint(m.sum(x[i]*a[k][i] for i in range(n)) <= b[k])

	#k_item constraint if necessary (if KQKP or HSP)
	if quad.num_items > 0:
		m.add_constraint(m.sum(x[i] for i in range(n)) == quad.num_items)

	#add auxiliary constraints
	for i in range(n):
		for j in range(i+1,n):
			if lhs_constraints:
				if C[i,j] > 0:
					m.add_constraint(w[i,j] <= x[i])
					m.add_constraint(w[i,j] <= x[j])
				else:
					m.add_constraint(x[i]+x[j]-1 <= w[i,j])
					m.add_constraint(w[i,j] >= 0)
			else:
				m.add_constraint(w[i,j] <= x[i])
				m.add_constraint(w[i,j] <= x[j])
				m.add_constraint(x[i]+x[j]-1 <= w[i,j])
				m.add_constraint(w[i,j] >= 0)

	#compute quadratic values contirbution to obj
	quadratic_values = 0
	for i in range(n):
		for j in range(i+1,n):
			quadratic_values = quadratic_values + (w[i,j]*(C[i,j]+C[j,i]))

	linear_values = m.sum(x[i]*c[i] for i in range(n))
	#set objective function (note - hsp has linear terms=0)
	m.maximize(linear_values + quadratic_values)

	#return model. no setup time for std
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

	#create model and add variables
	m = Model(name='glovers_linearization_'+bounds+'_'+constraints)
	x = m.binary_var_list(n, name="binary_var")

	if type(quad) is Knapsack: #HSP and UQP don't have cap constraint
		#add capacity constraint(s)
		for k in range(quad.m):
			m.add_constraint(m.sum(x[i]*a[k][i] for i in range(n)) <= b[k])

	#k_item constraint if necessary (if KQKP or HSP)
	if quad.num_items > 0:
		m.add_constraint(m.sum(x[i] for i in range(n)) == quad.num_items)

	#determine bounds for each column of C
	#U1,L1 must take item at index j, U0,L0 must not take
	U1 = np.zeros(n)
	L0 = np.zeros(n)
	U0 = np.zeros(n)
	L1 = np.zeros(n)
	start = timer()
	if(bounds=="original" or type(quad)==UQP):
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
	elif(bounds=="tight" or bounds=="tighter"):
		bound_m = Model(name='bound_model')
		if bounds == "tight":
			#for tight bounds solve for upper bound of col w/ continuous vars
			bound_x = bound_m.continuous_var_list(n, ub=1, lb=0)
		elif bounds == "tighter":
			#for even tighter bounds, instead use binary vars
			bound_x = bound_m.binary_var_list(n, ub=1, lb=0)
		if type(quad) is Knapsack:
			for k in range(quad.m):
				#add capacity constraints
				bound_m.add_constraint(bound_m.sum(bound_x[i]*a[k][i] for i in range(n)) <= b[k])
		if quad.num_items > 0:
			bound_m.add_constraint(bound_m.sum(bound_x[i] for i in range(n)) == quad.num_items)
		for j in range(n):
			# solve for upper bound U1
			bound_m.set_objective(sense="max", expr=bound_m.sum(C[i,j]*bound_x[i] for i in range(n)))
			u_con = bound_m.add_constraint(bound_x[j]==1)
			bound_m.solve()
			if "OPTIMAL_SOLUTION" not in str(bound_m.get_solve_status()):
				bound_m.remove_constraint(u_con)
				bound_m.add_constraint(bound_x[j]==0)
				m.add_constraint(x[j]==0)
				bound_m.solve()
			else:
				bound_m.remove_constraint(u_con)
			U1[j] = bound_m.objective_value

			# solve for lower bound L0
			bound_m.set_objective(sense="min", expr=bound_m.sum(C[i,j]*bound_x[i] for i in range(n)))
			l_con = bound_m.add_constraint(bound_x[j]==0)
			bound_m.solve()
			if "OPTIMAL_SOLUTION" not in str(bound_m.get_solve_status()):
				bound_m.remove_constraint(l_con)
				bound_m.add_constraint(bound_x[j]==1)
				m.add_constraint(x[j]==1)
				bound_m.solve()
			else:
				bound_m.remove_constraint(l_con)
			L0[j] = bound_m.objective_value

			if lhs_constraints:
				# solve for upper bound U0
				bound_m.set_objective(sense="max", expr=bound_m.sum(C[i,j]*bound_x[i] for i in range(n)))
				u_con = bound_m.add_constraint(bound_x[j] == 0)
				bound_m.solve()
				if "OPTIMAL_SOLUTION" not in str(bound_m.get_solve_status()):
					bound_m.remove_constraint(u_con)
					bound_m.add_constraint(bound_x[j]==1)
					m.add_constraint(x[j]==1)
					bound_m.solve()
				else:
					bound_m.remove_constraint(u_con)
				U0[j] = bound_m.objective_value

				# solve for lower bound L1
				bound_m.set_objective(sense="min", expr=bound_m.sum(C[i,j]*bound_x[i] for i in range(n)))
				l_con = bound_m.add_constraint(bound_x[j] == 1)
				bound_m.solve()
				if "OPTIMAL_SOLUTION" not in str(bound_m.get_solve_status()):
					bound_m.remove_constraint(l_con)
					bound_m.add_constraint(bound_x[j]==0)
					m.add_constraint(x[j]==0)
					bound_m.solve()
				else:
					bound_m.remove_constraint(l_con)
				L1[j] = bound_m.objective_value
	else:
		raise Exception(bounds + " is not a valid bound type for glovers")
	end = timer()
	setup_time = end-start

	#add auxiliary constrains
	if(constraints=="original"):
		#original glovers constraints
		z = m.continuous_var_list(keys=n,lb=-m.infinity)
		m.add_constraints(z[j] <= U1[j]*x[j] for j in range(n))
		if lhs_constraints:
			m.add_constraints(z[j] >= L1[j]*x[j] for j in range(n))
		for j in range(n):
			tempsum = sum(C[i,j]*x[i] for i in range(n))
			m.add_constraint(z[j] <= tempsum - L0[j]*(1-x[j]))
			if lhs_constraints:
				m.add_constraint(z[j] >= tempsum - U0[j]*(1-x[j]))
		m.maximize(m.sum(c[j]*x[j] + z[j] for j in range(n)))

	elif(constraints=="sub1" or constraints=="sub2"):
		#can make one of 2 substitutions using slack variables to further reduce # of constraints
		s = m.continuous_var_list(keys=n,lb=0)
		for j in range(n):
			tempsum = sum(C[i,j]*x[i] for i in range(n))
			if constraints=="sub1":
				m.add_constraint(s[j] >= U1[j]*x[j] - tempsum + L0[j]*(1-x[j]))
			else:
				m.add_constraint(s[j] >= -U1[j]*x[j] + tempsum - L0[j]*(1-x[j]))
		if constraints=="sub1":
			m.maximize(m.sum(c[i]*x[i] + (U1[i]*x[i]-s[i]) for i in range(n)))
		else:
			m.maximize(sum(c[j]*x[j] for j in range(n)) + sum(sum(C[i,j]*x[i] for i in range(n))-L0[j]*(1-x[j])-s[j] for j in range(n)))
	else:
		raise Exception(constraints + " is not a valid constraint type for glovers")

	#return model
	return [m,setup_time]

def glovers_linearization_prlt(quad, **kwargs):
	def prlt1_linearization(quad): #only called from within reformulate_glover (make inner func?)
		n = quad.n
		c = quad.c
		C = quad.C
		a = quad.a
		b = quad.b

		#create model and add variables
		m = Model(name='PRLT-1_linearization')
		x = m.continuous_var_list(n, lb=0, ub=1)
		w = m.continuous_var_matrix(keys1=n, keys2=n)

		if type(quad) is Knapsack: #HSP and UQP don't have cap constraint
			#add capacity constraint(s)
			for k in range(quad.m):
				m.add_constraint(m.sum(x[i]*a[k][i] for i in range(n)) <= b[k])

		#k_item constraint if necessary (if KQKP or HSP)
		if quad.num_items > 0:
			m.add_constraint(m.sum(x[i] for i in range(n)) == quad.num_items)

		#add auxiliary constraints
		for i in range(n):
			for j in range(i+1,n):
				m.add_constraint(w[i,j]==w[j,i], ctname='con16'+str(i)+str(j))

		for k in range(quad.m):
			for j in range(n):
				m.add_constraint(m.sum(a[k][i]*w[i,j] for i in range(n) if i!=j)<=(b[k]-a[k][j])*x[j])
				for i in range(n):
					m.add_constraint(w[i,j] <= x[j])

		#add objective function

		quadratic_values = 0
		for j in range(n):
			for i in range(n):
				if(i==j):
					continue
				quadratic_values = quadratic_values + (C[i,j]*w[i,j])

		linear_values = m.sum(x[j]*c[j] for j in range(n))
		m.maximize(linear_values + quadratic_values)

		#return model
		return m

	start = timer()
	m = prlt1_linearization(quad)
	m.solve()
	#print(m.objective_value)

	n = quad.n
	C = quad.C
	duals16 = np.zeros((n,n))
	for i in range(n):
		for j in range(i+1,n):
			con_name = 'con16'+str(i)+str(j)
			dual = (m.dual_values(m.get_constraint_by_name(con_name)))
			#makes sure there are no negative values in resulting quadratic matrix
			if dual > C[i][j]:
				dual = C[i][j]
			elif dual < -C[j][i]:
				dual = -C[j][i]
			duals16[i][j] = dual
	for i in range(quad.n):
		for j in range(i+1,quad.n):
			#modify quadratic coefficient matrix using duals
			duals16[j,i]=C[j,i]+duals16[i,j]
			duals16[i,j]=C[i,j]-duals16[i,j]
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
		x = m.continuous_var_list(n,name='binary_var', lb=0, ub=1)
		w = m.continuous_var_matrix(keys1=n, keys2=n, lb=0, ub=1)
		y = m.continuous_var_matrix(keys1=n, keys2=n, lb=0, ub=1)

		if type(quad) is Knapsack:
			# Multiply knapsack constraints by each x_j and (1-x_j)
			# Note: no need to include original knapsack constraints
			for k in range(quad.m):
				for j in range(n):
					#con 12
					m.add_constraint(m.sum(a[k][i]*w[i,j] for i in range(n) if i!=j)<=(b[k]-a[k][j])*x[j])
					#con 14
					m.add_constraint(m.sum(a[k][i]*y[i,j] for i in range(n) if i!=j)<=b[k]*(1-x[j]))

		#k_item constraint if necessary (if KQKP or HSP)
		if quad.num_items > 0:
			# Multiply partition constraint by each x_j
			# Note: There is no need to multiple by each (1-x_j), but must include original constraints
			m.add_constraint(m.sum(x[i] for i in range(n)) == quad.num_items)
			for j in range(n):
				m.add_constraint(m.sum(w[i,j] for i in range(n) if i!=j)== (quad.num_items-1)*x[j])

		# Add in symmetry constraints
		for i in range(n):
			for j in range(i+1,n):
				#con 16
				m.add_constraint(w[i,j]==w[j,i], ctname='con16_'+str(i)+"_"+str(j))

		for j in range(n):
			for i in range(n):
				if(i==j):
					continue
				m.add_constraint(w[i,j] <= x[j])
				m.add_constraint(y[i,j] <= 1-x[j])
				m.add_constraint(y[i,j] == x[i]-w[i,j], ctname='con17_'+str(i)+"_"+str(j))

		#add objective function
		quadratic_values = 0
		for j in range(n):
			for i in range(n):
				if(i==j):
					continue
				quadratic_values = quadratic_values + (C[i,j]*w[i,j])

		linear_values = m.sum(x[j]*c[j] for j in range(n))
		m.maximize(linear_values + quadratic_values)

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
	m.solve()

	# print()
	# print("RLT objective value = " + str(m.objective_value))
	# print()

	# Obtain the duals to the symmetry constraints
	duals16 = np.zeros((n,n))
	duals17 = np.zeros((n,n))
	for i in range(n):
		for j in range(i+1,n):
			con_name = 'con16_'+str(i)+"_"+str(j)
			duals16[i][j]=(m.dual_values(m.get_constraint_by_name(con_name)))
		for j in range(n):
			if i==j:
				continue
			con_name = 'con17_'+str(i)+"_"+str(j)
			duals17[i][j]=(m.dual_values(m.get_constraint_by_name(con_name)))

	# Delete RLT model
	m.end()

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
	x = m.binary_var_list(n, name="binary_var")

	if type(quad) is Knapsack: #HSP and UQP don't have cap constraint
		#add capacity constraint(s)
		for k in range(quad.m):
			m.add_constraint(m.sum(a[k][i]*x[i] for i in range(n)) <= b[k])

	#k_item constraint if necessary (if KQKP or HSP)
	if quad.num_items > 0:
		m.add_constraint(m.sum(x[i] for i in range(n)) == quad.num_items)

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
		bound_x = bound_m.continuous_var_list(n, ub=1, lb=0)

		# Add in original structural constraints
		if type(quad) is Knapsack: #HSP and UQP don't have cap constraint
			# Include knapsack constraints
			for k in range(quad.m):
				bound_m.add_constraint(bound_m.sum(a[k][i]*bound_x[i] for i in range(n)) <= b[k])

		#k_item constraint if necessary (if KQKP or HSP)
		if quad.num_items > 0:
			bound_m.add_constraint(bound_m.sum(bound_x[i] for i in range(n)) == quad.num_items)

		for j in range(n):
			# Solve for Ubar1
			bound_m.set_objective(sense="max", expr=bound_m.sum(Cbar[i,j]*bound_x[i] for i in range(n) if i!=j))
			xEquals1 = bound_m.add_constraint(bound_x[j]==1)
			bound_m.solve()
			if "OPTIMAL_SOLUTION" not in str(bound_m.get_solve_status()):
				# in case the xEquals1 constraint makes problem infeasible. (happens with kitem sometimes)
				bound_m.remove_constraint(xEquals1)
				bound_m.add_constraint(bound_x[j]==0)
				m.add_constraint(x[j]==0)
				bound_m.solve()
			else:
				bound_m.remove_constraint(xEquals1)
			Ubar1[j] = bound_m.objective_value

			# Solve for Lbar0
			xEquals0 = bound_m.add_constraint(bound_x[j]==0)
			bound_m.set_objective(sense="min", expr=bound_m.sum(Cbar[i,j]*bound_x[i] for i in range(n) if i!=j))
			bound_m.solve()
			if "OPTIMAL_SOLUTION" not in str(bound_m.get_solve_status()):
				# in case the xEquals1 constraint makes problem infeasible. (happens with kitem sometimes)
				bound_m.remove_constraint(xEquals0)
				bound_m.add_constraint(bound_x[j]==1)
				m.add_constraint(x[j]==1)
				bound_m.solve()
			else:
				bound_m.remove_constraint(xEquals0)
			Lbar0[j] = bound_m.objective_value

			# Solve for Uhat0
			bound_m.set_objective(sense="max", expr=bound_m.sum(Chat[i,j]*bound_x[i] for i in range(n) if i!=j))
			xEquals0 = bound_m.add_constraint(bound_x[j]==0)
			bound_m.solve()
			if "OPTIMAL_SOLUTION" not in str(bound_m.get_solve_status()):
				# in case the xEquals1 constraint makes problem infeasible. (happens with kitem sometimes)
				bound_m.remove_constraint(xEquals0)
				bound_m.add_constraint(bound_x[j]==1)
				m.add_constraint(x[j]==1)
				bound_m.solve()
			else:
				bound_m.remove_constraint(xEquals0)
			Uhat0[j] = bound_m.objective_value

			# Solve for Lhat1
			bound_m.set_objective(sense="min", expr=bound_m.sum(Chat[i,j]*bound_x[i] for i in range(n) if i!=j))
			xEquals1 = bound_m.add_constraint(bound_x[j]==1)
			bound_m.solve()
			if "OPTIMAL_SOLUTION" not in str(bound_m.get_solve_status()):
				# in case the xEquals1 constraint makes problem infeasible. (happens with kitem sometimes)
				bound_m.remove_constraint(xEquals1)
				bound_m.add_constraint(bound_x[j]==0)
				m.add_constraint(x[j]==0)
				bound_m.solve()
			else:
				bound_m.remove_constraint(xEquals1)
			Lhat1[j] = bound_m.objective_value

		# Delete bound model
		bound_m.end()
	else:
		raise Exception(bounds + " is not a valid bound type for glovers")

	end = timer()
	setup_time = end-start

	if(constraints=="original"):
		z1 = m.continuous_var_list(keys=n,lb=-m.infinity)
		z2 = m.continuous_var_list(keys=n,lb=-m.infinity)

		for j in range(n):
			m.add_constraint(z1[j] <= Ubar1[j]*x[j])
			tempsum1 = sum(Cbar[i,j]*x[i] for i in range(n) if i!=j)
			m.add_constraint(z1[j] <= tempsum1 - Lbar0[j]*(1-x[j]))

			m.add_constraint(z2[j] <= Uhat0[j]*(1-x[j]))
			tempsum2 = sum(Chat[i,j]*x[i] for i in range(n) if i!=j)
			m.add_constraint(z2[j] <= tempsum2 - (Lhat1[j]*x[j]))

		# Set up the objective function
		m.maximize(m.sum(c[j]*x[j] + z1[j] + z2[j] for j in range(n)))
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
	x = m.binary_var_list(n, name="binary_var")
	z = m.continuous_var_cube(keys1=n, keys2=n, keys3=n, lb=-m.infinity)

	if type(quad) is Knapsack: #HSP and UQP don't have cap constraint
		#add capacity constraint(s)
		for k in range(quad.m):
			m.add_constraint(m.sum(x[i]*a[k][i] for i in range(n)) <= b[k])

	#k_item constraint if necessary (if KQKP or HSP)
	if quad.num_items > 0:
		m.add_constraint(m.sum(x[i] for i in range(n)) == quad.num_items)

	#add auxiliary constraints
	for i in range(n):
		for j in range(i+1,n):
			m.add_constraint(z[i,i,j]+z[j,i,j] <= 1)
			if C[i,j] < 0:
				m.add_constraint(x[i] + z[i,i,j] <= 1)
				m.add_constraint(x[j] + z[j,i,j] <= 1)
			elif C[i,j] > 0:
				m.add_constraint(x[i] + z[i,i,j] + z[j,i,j] >= 1)
				m.add_constraint(x[j] + z[i,i,j] + z[j,i,j] >= 1)

	#compute quadratic values contirbution to obj
	constant = 0
	quadratic_values = 0
	for i in range(n):
		for j in range(i+1,n):
			constant = constant + C[i,j]
			quadratic_values = quadratic_values + (C[i,j]*(z[i,i,j]+z[j,i,j]))
	#set objective function

	linear_values = m.sum(x[i]*c[i] for i in range(n))
	m.maximize(linear_values + constant - quadratic_values)

	#return model + setup time=0
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
	x = m.binary_var_list(n, name="binary_var")
	s = m.continuous_var_list(n)
	y = m.continuous_var_list(n)

	if type(quad) is Knapsack: #HSP and UQP don't have cap constraint
		#add capacity constraint(s)
		for k in range(quad.m):
			m.add_constraint(m.sum(x[i]*a[k][i] for i in range(n)) <= b[k])

	#k_item constraint if necessary (if KQKP or HSP)
	if quad.num_items > 0:
		m.add_constraint(m.sum(x[i] for i in range(n)) == quad.num_items)

	start = timer()
	U = np.zeros(n)
	L = np.zeros(n)
	bound_m = Model(name='bound_model')
	bound_x = bound_m.continuous_var_list(n, ub=1, lb=0)
	if type(quad) is Knapsack:
		for k in range(quad.m):
			#add capacity constraints
			bound_m.add_constraint(bound_m.sum(bound_x[i]*a[k][i] for i in range(n)) <= b[k])
	if quad.num_items > 0:
		bound_m.add_constraint(bound_m.sum(bound_x[i] for i in range(n)) == quad.num_items)
	for i in range(n):
		# solve for upper bound
		bound_m.set_objective(sense="max", expr=bound_m.sum(C[i,j]*bound_x[j] for j in range(n)))
		bound_m.solve()
		U[i] = bound_m.objective_value

		# solve for lower bound
		bound_m.set_objective(sense="min", expr=bound_m.sum(C[i,j]*bound_x[j] for j in range(n)))
		bound_m.solve()
		L[i] = bound_m.objective_value
	end = timer()
	setup_time = end-start
	#add auxiliary constraints
	for i in range(n):
		m.add_constraint(sum(C[i,j]*x[j] for j in range(n))-s[i]-L[i]==y[i])
		m.add_constraint(y[i] <= (U[i]-L[i])*(1-x[i]))
		m.add_constraint(s[i] <= (U[i]-L[i])*x[i])
		m.add_constraint(y[i] >= 0)
		m.add_constraint(s[i] >= 0)

	#set objective function
	m.maximize(sum(s[i]+x[i]*(c[i]+L[i])for i in range(n)))
	#return model + setup time
	return [m, setup_time]

def qsap_standard(qsap, **kwargs):
	n = qsap.n
	m = qsap.m
	e = qsap.e
	c = qsap.c

	#create model and add variables
	mdl = Model(name='qsap_standard_linearization')
	x = mdl.binary_var_matrix(m,n,name="binary_var")
	w = np.array([[[[mdl.continuous_var() for i in range(n)]for j in range(m)]
						for k in range(n)]for l in range(m)])

	mdl.add_constraints((sum(x[i,k] for k in range(n)) == 1) for i in range(m))

	#add auxiliary constraints
	#TODO implement lhs here?
	for i in range(m-1):
		for k in range(n):
			for j in range(i+1,m):
				for l in range(n):
					mdl.add_constraint(w[i,k,j,l] <= x[i,k])
					mdl.add_constraint(w[i,k,j,l] <= x[j,l])
					mdl.add_constraint(x[i,k] + x[j,l] - 1 <= w[i,k,j,l])
					mdl.add_constraint(w[i,k,j,l] >= 0)

	#compute quadratic values contirbution to obj
	quadratic_values = 0
	for i in range(m-1):
		for j in range(i+1,m):
			for k in range(n):
				for l in range(n):
					quadratic_values = quadratic_values + (c[i,k,j,l]*(w[i,k,j,l]))

	linear_values = mdl.sum(x[i,k]*e[i,k] for k in range(n) for i in range(m))
	mdl.maximize(linear_values + quadratic_values)

	#return model. no setup time for std
	return [mdl, 0]

def qsap_glovers(qsap, bounds="original", constraints="original", lhs_constraints=False, **kwargs):
	n = qsap.n
	m = qsap.m
	e = qsap.e
	c = qsap.c
	mdl = Model(name='qsap_glovers')
	x = mdl.binary_var_matrix(keys1=m,keys2=n,name="binary_var")
	mdl.add_constraints((sum(x[i,k] for k in range(n)) == 1) for i in range(m))

	#let cplex solve w/ quadratic objective function
	# mdl.maximize(sum(sum(e[i,k]*x[i,k] for k in range(n))for i in range(m))
	# 			+ sum(sum(sum(sum(c[i,k,j,l]*x[i,k]*x[j,l] for l in range(n))for k in range(n))
	# 			for j in range(1+i,m)) for i in range(m-1)))
	# mdl.solve()
	# mdl.print_solution()


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
		bound_mdl = Model(name="u_bound_m")
		if bounds=="tight":
			bound_x = bound_mdl.continuous_var_matrix(keys1=m,keys2=n,ub=1,lb=0)
		elif bounds == "tighter":
			bound_x = bound_mdl.binary_var_matrix(keys1=m,keys2=n)
		bound_mdl.add_constraints((sum(bound_x[i,k] for k in range(n)) == 1) for i in range(m))
		for i in range(m-1):
			for k in range(n):
				# solve for upper bound U1
				bound_mdl.set_objective(sense="max", expr=sum(sum(c[i,k,j,l]*bound_x[j,l] for l in range(n)) for j in range(i+1,m)))
				u_con = bound_mdl.add_constraint(bound_x[i,k]==1)
				bound_mdl.solve()
				bound_mdl.remove_constraint(u_con)
				U1[i,k] = bound_mdl.objective_value

				# solve for lower bound L0
				bound_mdl.set_objective(sense="min", expr=sum(sum(c[i,k,j,l]*bound_x[j,l] for l in range(n)) for j in range(i+1,m)))
				l_con = bound_mdl.add_constraint(bound_x[i,k]==0)
				bound_mdl.solve()
				bound_mdl.remove_constraint(l_con)
				L0[i,k] = bound_mdl.objective_value

				if lhs_constraints:
					# solve for upper bound U0
					bound_mdl.set_objective(sense="max", expr=sum(sum(c[i,k,j,l]*bound_x[j,l] for l in range(n)) for j in range(i+1,m)))
					u_con = bound_mdl.add_constraint(bound_x[i,k] == 0)
					bound_mdl.solve()
					bound_mdl.remove_constraint(u_con)
					U0[i,k] = bound_mdl.objective_value

					# solve for lower bound L1
					bound_mdl.set_objective(sense="min", expr=sum(sum(c[i,k,j,l]*bound_x[j,l] for l in range(n)) for j in range(i+1,m)))
					l_con = bound_mdl.add_constraint(bound_x[i,k] == 1)
					bound_mdl.solve()
					L1[i,k] = bound_mdl.objective_value
					bound_mdl.remove_constraint(l_con)
	else:
		raise Exception(bounds + " is not a valid bound type for glovers")
	end = timer()
	setup_time = end-start

	#add auxiliary constrains
	if constraints=="original":
		z = mdl.continuous_var_matrix(keys1=m,keys2=n,lb=-mdl.infinity)
		mdl.add_constraints(z[i,k] <= x[i,k]*U1[i,k] for i in range(m-1) for k in range(n))
		mdl.add_constraints(z[i,k] <= sum(sum(c[i,k,j,l]*x[j,l] for l in range(n)) for j in range(i+1,m))
										-L0[i,k]*(1-x[i,k]) for i in range(m-1) for k in range(n))
		if lhs_constraints:
			mdl.add_constraints(z[i,k] >= x[i,k]*L1[i,k] for i in range(m-1) for k in range(n))
			mdl.add_constraints(z[i,k] >= sum(sum(c[i,k,j,l]*x[j,l] for l in range(n)) for j in range(i+1,m))
										-U0[i,k]*(1-x[i,k]) for i in range(m-1) for k in range(n))
		mdl.maximize(sum(sum(e[i,k]*x[i,k] for k in range(n))for i in range(m))
					+ sum(sum(z[i,k] for k in range(n)) for i in range(m-1)))
	elif constraints=="sub1":
		s = mdl.continuous_var_matrix(keys1=m,keys2=n,lb=0)
		mdl.add_constraints(s[i,k] >= U1[i,k]*x[i,k]+L0[i,k]*(1-x[i,k])-sum(sum(c[i,k,j,l]*x[j,l] for l in range(n)) for j in range(i+1,m))
						for k in range(n) for i in range(m-1))
		mdl.maximize(sum(sum(e[i,k]*x[i,k] for k in range(n))for i in range(m))
					+ sum(sum(U1[i,k]*x[i,k]-s[i,k] for k in range(n)) for i in range(m-1)))
	elif constraints=="sub2":
		s = mdl.continuous_var_matrix(keys1=m,keys2=n,lb=0)
		mdl.add_constraints(s[i,k] >= -L0[i,k]*(1-x[i,k])-(x[i,k]*U1[i,k])+sum(sum(c[i,k,j,l]*x[j,l] for l in range(n)) for j in range(i+1,m))
		 				for k in range(n) for i in range(m-1))
		mdl.maximize(sum(sum(e[i,k]*x[i,k] for k in range(n))for i in range(m))
					+ sum(sum(-s[i,k]-(L0[i,k]*(1-x[i,k])) + sum(sum(c[i,k,j,l]*x[j,l] for l in range(n))
					for j in range(i+1,m)) for k in range(n)) for i in range(m-1)))
	else:
		raise Exception(constraints + " is not a valid constraint type for glovers")

	#return model
	return [mdl,setup_time]

def qsap_elf(qsap, **kwargs):
	n = qsap.n
	m = qsap.m
	e = qsap.e
	c = qsap.c
	mdl = Model(name='qsap_elf')
	x = mdl.binary_var_matrix(m,n,name="binary_var")
	z = np.array([[[[[[mdl.continuous_var() for i in range(n)]for j in range(m)]
						for k in range(n)]for l in range(m)] for s in range(n)] for t in range(m)])
	mdl.add_constraints((sum(x[i,k] for k in range(n)) == 1) for i in range(m))

	#add auxiliary constraints
	for i in range(m-1):
		for j in range(i+1,m):
			for k in range(n):
				for l in range(n):
					mdl.add_constraint(z[i,k,i,k,j,l] + z[j,l,i,k,j,l] <= 1)
					mdl.add_constraint(x[i,k] + z[i,k,i,k,j,l] <= 1)
					mdl.add_constraint(x[j,l] + z[j,l,i,k,j,l] <= 1)
					mdl.add_constraint(x[i,k] + z[i,k,i,k,j,l] + z[j,l,i,k,j,l] >= 1)
					mdl.add_constraint(x[j,l] + z[i,k,i,k,j,l] + z[j,l,i,k,j,l] >= 1)

	#compute quadratic values contirbution to obj
	constant = 0
	quadratic_values = 0
	for i in range(m-1):
		for j in range(i+1,m):
			for k in range(n):
				for l in range(n):
					constant = constant + c[i,k,j,l]
					quadratic_values = quadratic_values + (c[i,k,j,l]*(z[i,k,i,k,j,l]+z[j,l,i,k,j,l]))

	linear_values = mdl.sum(x[i,k]*e[i,k] for k in range(n) for i in range(m))
	mdl.maximize(linear_values + constant - quadratic_values)

	#return model + setup time=0
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
	x = mdl.binary_var_matrix(m,n,name="binary_var")
	s = mdl.continuous_var_matrix(m,n)
	y = mdl.continuous_var_matrix(m,n)

	mdl.add_constraints((sum(x[i,k] for k in range(n)) == 1) for i in range(m))

	start = timer()
	U = np.zeros((m,n))
	L = np.zeros((m,n))
	bound_mdl = Model(name='upper_bound_model')
	bound_x = bound_mdl.continuous_var_matrix(keys1=m,keys2=n,ub=1,lb=0)
	bound_mdl.add_constraints((sum(bound_x[i,k] for k in range(n)) == 1) for i in range(m))
	for i in range(m-1):
		for k in range(n):
			# solve for upper bound
			bound_mdl.set_objective(sense="max", expr=sum(sum(c[i,k,j,l]*bound_x[j,l] for l in range(n)) for j in range(i+1,m)))
			bound_mdl.solve()
			U[i,k] = bound_mdl.objective_value

			# solve for lower bound
			bound_mdl.set_objective(sense="min", expr=sum(sum(c[i,k,j,l]*bound_x[j,l] for l in range(n)) for j in range(i+1,m)))
			bound_mdl.solve()
			L[i,k] = bound_mdl.objective_value

	end = timer()
	setup_time = end-start
	#add auxiliary constraints
	for i in range(m-1):
		for k in range(n):
			mdl.add_constraint(sum(sum(c[i,k,j,l]*x[j,l] for j in range(i+1,m)) for l in range(n))-s[i,k]-L[i,k]==y[i,k])
			mdl.add_constraint(y[i,k] <= (U[i,k]-L[i,k])*(1-x[i,k]))
			mdl.add_constraint(s[i,k] <= (U[i,k]-L[i,k])*x[i,k])

	#set objective function
	linear_values = sum(sum(e[i,k]*x[i,k] for i in range(m)) for k in range(n))
	mdl.maximize(linear_values + sum(sum(s[i,k]+(x[i,k]*L[i,k]) for i in range(m-1)) for k in range(n)))

	#return model + setup time
	return [mdl, setup_time]

def no_linearization(quad, **kwargs):
	n = quad.n
	c = quad.c
	m = Model(name='no_linearization')
	x = m.binary_var_list(n, name="binary_var")
	if type(quad) is Knapsack: #HSP and UQP don't have cap constraint
		#add capacity constraint(s)
		for k in range(quad.m):
			m.add_constraint(m.sum(x[i]*quad.a[k][i] for i in range(n)) <= quad.b[k])

	#k_item constraint if necessary (if KQKP or HSP)
	if quad.num_items > 0:
		m.add_constraint(m.sum(x[i] for i in range(n)) == quad.num_items)

	#compute quadratic values contirbution to obj
	quadratic_values = 0
	for i in range(n):
		for j in range(i+1,n):
			quadratic_values = quadratic_values + (x[i]*x[j]*quad.C[i,j])
	#set objective function
	linear_values = m.sum(x[i]*c[i] for i in range(n))
	m.maximize(linear_values + quadratic_values)

	return [m, 0]

def solve_model(model, solve_relax=True, **kwargs):
	#use with block to automatically call m.end() when finished
	with model as m:
		time_limit = False
		m.set_time_limit(600)
		start = timer()
		m.solve()
		end = timer()
		if("FEASIBLE" in str(m.get_solve_status())):
			print('time limit exceeded')
			time_limit=True
		solve_time = end-start
		objective_value = m.objective_value
		#print(objective_value)

		if solve_relax and not time_limit:
			#compute continuous relaxation and integrality_gap
			i = 0
			while m.get_var_by_name("binary_var_"+str(i)) is not None:
				relaxation_var = m.get_var_by_name("binary_var_"+str(i))
				m.set_var_type(relaxation_var,m.continuous_vartype)
				i+=1

			#TODO this is ugly way to find all binary vars for QSAP and relax themself.
			#could use method I use in gurobi?
			if i == 0: #this should only happen if model is for a QSAP problem
				j = 0
				while m.get_var_by_name("binary_var_"+str(i)+"_"+str(j)) is not None:
					j=0
					while m.get_var_by_name("binary_var_"+str(i)+"_"+str(j)) is not None:
						relaxation_var = m.get_var_by_name("binary_var_"+str(i)+"_"+str(j))
						m.set_var_type(relaxation_var,m.continuous_vartype)
						j+=1
					j-=1
					i+=1

			m.solve()
			continuous_obj_value = m.objective_value
			integrality_gap=((continuous_obj_value-objective_value)/objective_value)*100
		else:
			continuous_obj_value = -1
			integrality_gap = -1


	#create and return results dictionary
	results = {"solve_time":solve_time,
				"objective_value":objective_value,
				"relaxed_solution":continuous_obj_value,
				"integrality_gap":integrality_gap,
				"time_limit":time_limit}
	return results
