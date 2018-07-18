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
	if(bounds=="original"):
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
		u_bound_m = Model(name='upper_bound_model')
		l_bound_m = Model(name='lower_bound_model')
		if bounds == "tight":
			#for tight bounds solve for upper bound of col w/ continuous vars
			u_bound_x = u_bound_m.continuous_var_list(n, ub=1, lb=0)
			l_bound_x = l_bound_m.continuous_var_list(n, ub=1, lb=0)
		elif bounds == "tighter":
			#for even tighter bounds, instead use binary vars
			u_bound_x = u_bound_m.binary_var_list(n, ub=1, lb=0)
			l_bound_x = l_bound_m.binary_var_list(n, ub=1, lb=0)
		if type(quad) is Knapsack:
			for k in range(quad.m):
				#add capacity constraints
				u_bound_m.add_constraint(u_bound_m.sum(u_bound_x[i]*a[k][i] for i in range(n)) <= b[k])
				l_bound_m.add_constraint(l_bound_m.sum(l_bound_x[i]*a[k][i] for i in range(n)) <= b[k])
		if quad.num_items > 0:
			u_bound_m.add_constraint(u_bound_m.sum(u_bound_x[i] for i in range(n)) == quad.num_items)
			l_bound_m.add_constraint(l_bound_m.sum(l_bound_x[i] for i in range(n)) == quad.num_items)
		for j in range(n):
			#for each col, solve model to find upper/lower bound
			u_bound_m.set_objective(sense="max", expr=u_bound_m.sum(C[i,j]*u_bound_x[i] for i in range(n)))
			l_bound_m.set_objective(sense="min", expr=l_bound_m.sum(C[i,j]*l_bound_x[i] for i in range(n)))
			u_con = u_bound_m.add_constraint(u_bound_x[j]==1)
			l_con = l_bound_m.add_constraint(l_bound_x[j]==0)
			u_bound_m.solve()
			if "OPTIMAL_SOLUTION" not in str(u_bound_m.get_solve_status()):
				#print(str(u_bound_m.get_solve_status()) + " when solving for upper bound (U1)")
				u_bound_m.remove_constraint(u_con)
				u_bound_m.add_constraint(u_bound_x[j]==0)
				m.add_constraint(x[j]==0)
				u_bound_m.solve()
			else:
				u_bound_m.remove_constraint(u_con)
			l_bound_m.solve()
			if "OPTIMAL_SOLUTION" not in str(l_bound_m.get_solve_status()):
				#print(str(l_bound_m.get_solve_status()) + " when solving for lower bound (L0)")
				l_bound_m.remove_constraint(l_con)
				l_bound_m.add_constraint(l_bound_x[j]==1)
				m.add_constraint(x[j]==1)
				l_bound_m.solve()
			else:
				l_bound_m.remove_constraint(l_con)
			U1[j] = u_bound_m.objective_value
			L0[j] = l_bound_m.objective_value
			if lhs_constraints:
				u_con = u_bound_m.add_constraint(u_bound_x[j] == 0)
				l_con = l_bound_m.add_constraint(l_bound_x[j] == 1)
				u_bound_m.solve()
				if "OPTIMAL_SOLUTION" not in str(u_bound_m.get_solve_status()):
					#TODO double check optimal sol result w/ std lin
					#print(str(u_bound_m.get_solve_status()) + " when solving for upper bound (U0)")
					u_bound_m.remove_constraint(u_con)
					u_bound_m.add_constraint(u_bound_x[j]==1)
					m.add_constraint(x[j]==1)
					u_bound_m.solve()
				else:
					u_bound_m.remove_constraint(u_con)
				l_bound_m.solve()
				if "OPTIMAL_SOLUTION" not in str(l_bound_m.get_solve_status()):
					#print(str(l_bound_m.get_solve_status()) + " when solving for lower bound (L1)")
					l_bound_m.remove_constraint(l_con)
					l_bound_m.add_constraint(l_bound_x[j]==0)
					m.add_constraint(x[j]==0)
					l_bound_m.solve()
				else:
					l_bound_m.remove_constraint(l_con)
				U0[j] = u_bound_m.objective_value
				L1[j] = l_bound_m.objective_value
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

#TODO glovers_rlt currently works only for knapsack w/ original constraints
def glovers_linearization_rlt(quad, bounds="tight", constraints="original", **kwargs):
	def rlt1_linearization(quad):
		n = quad.n
		c = quad.c
		C = quad.C
		a = quad.a
		b = quad.b

		#create model and add variables
		m = Model(name='RLT-1_linearization')
		x = m.continuous_var_list(n,name='binary_var', lb=0, ub=1) #named binary_var so can easily switch for debug
		w = m.continuous_var_matrix(keys1=n, keys2=n, lb=0, ub=1)
		y = m.continuous_var_matrix(keys1=n, keys2=n, lb=0, ub=1)

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
				#con 16
				m.add_constraint(w[i,j]==w[j,i], ctname='con16'+str(i)+str(j))

		for k in range(quad.m):
			for j in range(n):
				#con 12
				m.add_constraint(m.sum(a[k][i]*w[i,j] for i in range(n) if i!=j)<=(b[k]-a[k][j])*x[j])
				#con 14
				m.add_constraint(m.sum(a[k][i]*y[i,j] for i in range(n) if i!=j)<=b[k]*(1-x[j]))

		for j in range(n):
			for i in range(n):
				if(i==j):
					continue
				#con 13 (w>=0 implied) - imp to add anyways?
				m.add_constraint(w[i,j] <= x[j])
				#con 15 (y>=0 implied)
				m.add_constraint(y[i,j] <= 1-x[j])
				#con 17
				m.add_constraint(y[i,j] == x[i]-w[i,j], ctname='con17'+str(i)+str(j))

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
	n = quad.n
	c = quad.c
	C = quad.C
	a = quad.a
	b = quad.b

	#model with rlt1, solve continuous relax and get duals to constraints 16,17
	m = rlt1_linearization(quad)
	m.solve()
	print(m.objective_value)
	duals16 = np.zeros((n,n))
	duals17 = np.zeros((n,n))
	for i in range(n):
		for j in range(i+1,n):
			con_name = 'con16'+str(i)+str(j)
			duals16[i][j]=(m.dual_values(m.get_constraint_by_name(con_name)))
		for j in range(n):
			if i==j:
				continue
			con_name = 'con17'+str(i)+str(j)
			duals17[i][j]=(m.dual_values(m.get_constraint_by_name(con_name)))
	#print(duals17)
	D = np.zeros((n,n))
	E = np.zeros((n,n))
	#optimal split, found using dual vars from rlt1 continuous relaxation
	for i in range(n):
		for j in range(n):
			if i==j:
				continue
			if i<j:
				D[i,j] = C[i,j]-duals16[i,j]-duals17[i,j]
				E[i,j] = -duals17[i,j]
			if i>j:
				D[i,j] = C[i,j]+duals16[j,i]-duals17[i,j]
				E[i,j] = -duals17[i,j]
	#E = -duals17

	#update linear values as well
	for j in range(n):
		c[j] = c[j] + sum(duals17[j,i] for i in range(n) if i!=j)

	#simple split (this works but is not optimal)
	# for i in range(n):
		# for j in range(i+1,n):
			# D[i,j] = C[i,j]/4
			# D[j,i] = C[i,j]/4
			# E[i,j] = C[i,j]/4
			# E[j,i] = C[i,j]/4

	#create model and add variables
	m = Model(name='glovers_linearization_rlt_'+bounds+'_'+constraints)
	x = m.binary_var_list(n, name="binary_var")

	if type(quad) is Knapsack: #HSP and UQP don't have cap constraint
		#add capacity constraint(s)
		for k in range(quad.m):
			m.add_constraint(m.sum(x[i]*a[k][i] for i in range(n)) <= b[k])

	#k_item constraint if necessary (if KQKP or HSP)
	if quad.num_items > 0:
		m.add_constraint(m.sum(x[i] for i in range(n)) == quad.num_items)

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
		u_bound_x1 = u_bound_m1.continuous_var_list(n, ub=1, lb=0)
		l_bound_x1 = l_bound_m1.continuous_var_list(n, ub=1, lb=0)
		for k in range(quad.m):
			u_bound_m1.add_constraint(u_bound_m1.sum(u_bound_x1[i]*a[k][i] for i in range(n)) <= b[k])
			l_bound_m1.add_constraint(l_bound_m1.sum(l_bound_x1[i]*a[k][i] for i in range(n)) <= b[k])
		u_bound_m2 = Model(name='upper_bound_model2')
		l_bound_m2 = Model(name='lower_bound_model2')
		u_bound_x2 = u_bound_m2.continuous_var_list(n, ub=1, lb=0)
		l_bound_x2 = l_bound_m2.continuous_var_list(n, ub=1, lb=0)
		for k in range(quad.m):
			u_bound_m2.add_constraint(u_bound_m2.sum(u_bound_x2[i]*a[k][i] for i in range(n)) <= b[k])
			l_bound_m2.add_constraint(l_bound_m2.sum(l_bound_x2[i]*a[k][i] for i in range(n)) <= b[k])

		for j in range(n):
			u_bound_m1.set_objective(sense="max", expr=u_bound_m1.sum(D[i,j]*u_bound_x1[i] for i in range(n) if i!=j))
			l_bound_m1.set_objective(sense="min", expr=l_bound_m1.sum(D[i,j]*l_bound_x1[i] for i in range(n) if i!=j))
			u_con1 = u_bound_m1.add_constraint(u_bound_x1[j]==1)
			l_con1 = l_bound_m1.add_constraint(l_bound_x1[j]==0)
			u_bound_m1.solve()
			l_bound_m1.solve()
			u_bound_m1.remove_constraint(u_con1)
			l_bound_m1.remove_constraint(l_con1)
			U1[j] = u_bound_m1.objective_value
			L1[j] = l_bound_m1.objective_value

			u_bound_m2.set_objective(sense="max", expr=u_bound_m2.sum(E[i,j]*u_bound_x2[i] for i in range(n) if i!=j))
			l_bound_m2.set_objective(sense="min", expr=l_bound_m2.sum(E[i,j]*l_bound_x2[i] for i in range(n) if i!=j))
			u_con2 = u_bound_m2.add_constraint(u_bound_x2[j]==0)
			l_con2 = l_bound_m2.add_constraint(l_bound_x2[j]==1)
			u_bound_m2.solve()
			l_bound_m2.solve()
			u_bound_m2.remove_constraint(u_con2)
			l_bound_m2.remove_constraint(l_con2)
			U2[j] = u_bound_m2.objective_value
			L2[j] = l_bound_m2.objective_value
	else:
		raise Exception(bounds + " is not a valid bound type for glovers")

	#add auxiliary constrains
	if(constraints=="original"):
		z1 = m.continuous_var_list(keys=n,lb=-m.infinity)
		z2 = m.continuous_var_list(keys=n,lb=-m.infinity)
		for j in range(n):
			m.add_constraint(z1[j] <= U1[j]*x[j])
			m.add_constraint(z2[j] <= U2[j]*(1-x[j]))
		for j in range(n):
			tempsum1 = sum(D[i,j]*x[i] for i in range(n) if i!=j)
			m.add_constraint(z1[j] <= tempsum1 - L1[j]*(1-x[j]))
			tempsum2 = sum(E[i,j]*x[i] for i in range(n) if i!=j)
			m.add_constraint(z2[j] <= tempsum2 - (L2[j]*x[j]))
		m.maximize(m.sum(c[j]*x[j] + z1[j] + z2[j] for j in range(n)))
	else:
		raise Exception(constraints + " is not a valid constraint type for glovers")

	end = timer()
	setup_time = end-start

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
	u_bound_m = Model(name='upper_bound_model')
	l_bound_m = Model(name='lower_bound_model')
	u_bound_x = u_bound_m.continuous_var_list(n, ub=1, lb=0)
	l_bound_x = l_bound_m.continuous_var_list(n, ub=1, lb=0)
	if type(quad) is Knapsack:
		for k in range(quad.m):
			#add capacity constraints
			u_bound_m.add_constraint(u_bound_m.sum(u_bound_x[i]*a[k][i] for i in range(n)) <= b[k])
			l_bound_m.add_constraint(l_bound_m.sum(l_bound_x[i]*a[k][i] for i in range(n)) <= b[k])
	if quad.num_items > 0:
		u_bound_m.add_constraint(u_bound_m.sum(u_bound_x[i] for i in range(n)) == quad.num_items)
		l_bound_m.add_constraint(l_bound_m.sum(l_bound_x[i] for i in range(n)) == quad.num_items)
	for i in range(n):
		#for each col, solve model to find upper/lower bound
		u_bound_m.set_objective(sense="max", expr=u_bound_m.sum(C[i,j]*u_bound_x[j] for j in range(n)))
		l_bound_m.set_objective(sense="min", expr=l_bound_m.sum(C[i,j]*l_bound_x[j] for j in range(n)))
		u_bound_m.solve()
		l_bound_m.solve()
		U[i] = u_bound_m.objective_value
		L[i] = l_bound_m.objective_value
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
	start = timer()
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
			u_bound_x = u_bound_mdl.continuous_var_matrix(keys1=m,keys2=n,ub=1,lb=0)
			l_bound_x = l_bound_mdl.continuous_var_matrix(keys1=m,keys2=n,ub=1,lb=0)
		elif bounds == "tighter":
			u_bound_x = u_bound_mdl.binary_var_matrix(keys1=m,keys2=n)
			l_bound_x = l_bound_mdl.binary_var_matrix(keys1=m,keys2=n)
		u_bound_mdl.add_constraints((sum(u_bound_x[i,k] for k in range(n)) == 1) for i in range(m))
		l_bound_mdl.add_constraints((sum(l_bound_x[i,k] for k in range(n)) == 1) for i in range(m))
		for i in range(m-1):
			for k in range(n):
				u_bound_mdl.set_objective(sense="max", expr=sum(sum(c[i,k,j,l]*u_bound_x[j,l] for l in range(n)) for j in range(i+1,m)))
				l_bound_mdl.set_objective(sense="min", expr=sum(sum(c[i,k,j,l]*l_bound_x[j,l] for l in range(n)) for j in range(i+1,m)))
				u_con = u_bound_mdl.add_constraint(u_bound_x[i,k]==1)
				l_con = l_bound_mdl.add_constraint(l_bound_x[i,k]==0)
				u_bound_mdl.solve()
				l_bound_mdl.solve()
				U1[i,k] = u_bound_mdl.objective_value
				L0[i,k] = l_bound_mdl.objective_value
				u_bound_mdl.remove_constraint(u_con)
				l_bound_mdl.remove_constraint(l_con)
				if lhs_constraints:
					u_con = u_bound_mdl.add_constraint(u_bound_x[i,k] == 0)
					l_con = l_bound_mdl.add_constraint(l_bound_x[i,k] == 1)
					u_bound_mdl.solve()
					l_bound_mdl.solve()
					U0[i,k] = u_bound_mdl.objective_value
					L1[i,k] = l_bound_mdl.objective_value
					u_bound_mdl.remove_constraint(u_con)
					l_bound_mdl.remove_constraint(l_con)
	else:
		raise Exception(bounds + " is not a valid bound type for glovers")

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
	u_bound_mdl = Model(name='upper_bound_model')
	l_bound_mdl = Model(name='lower_bound_model')
	u_bound_x = u_bound_mdl.continuous_var_matrix(keys1=m,keys2=n,ub=1,lb=0)
	l_bound_x = l_bound_mdl.continuous_var_matrix(keys1=m,keys2=n,ub=1,lb=0)
	u_bound_mdl.add_constraints((sum(u_bound_x[i,k] for k in range(n)) == 1) for i in range(m))
	l_bound_mdl.add_constraints((sum(l_bound_x[i,k] for k in range(n)) == 1) for i in range(m))
	for i in range(m-1):
		for k in range(n):
			u_bound_mdl.set_objective(sense="max", expr=sum(sum(c[i,k,j,l]*u_bound_x[j,l] for l in range(n)) for j in range(i+1,m)))
			l_bound_mdl.set_objective(sense="min", expr=sum(sum(c[i,k,j,l]*l_bound_x[j,l] for l in range(n)) for j in range(i+1,m)))
			u_bound_mdl.solve()
			l_bound_mdl.solve()
			U[i,k] = u_bound_mdl.objective_value
			L[i,k] = l_bound_mdl.objective_value
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


# p = QSAP(n=3,m=4)
# print(solve_model(qsap_glovers(p)[0]))
# p.reorder(take_max=False, flip_order=False)
# print(solve_model(qsap_glovers(p)[0]))

# p = Knapsack(n=15)
# print(solve_model(glovers_linearization(p)[0]))
# print(solve_model(glovers_linearization_rlt(p)[0]))
