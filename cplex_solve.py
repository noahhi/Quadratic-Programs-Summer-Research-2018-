from quadratics import Knapsack, UQP, HSP
import numpy as np
from timeit import default_timer as timer
from docplex.mp.model import Model

def standard_linearization(quad, con1=True, con2=True, con3=True, con4=True):
	start = timer()
	n = quad.n
	c = quad.c
	C = quad.C
	a = quad.a
	b = quad.b

	#create model and add variables
	m = Model(name='standard_linearization')
	x = m.binary_var_list(n, name="binary_var")
	w = m.continuous_var_matrix(keys1=n, keys2=n)

	if type(quad) is Knapsack: #HSP and UQP don't have cap constraint
		#add capacity constraint(s)
		for k in range(quad.m):
			m.add_constraint(m.sum(x[i]*a[k][i] for i in range(n)) <= b[k])
		#k_item constraint(s) if necessary (if KQKP)
		for k in range(len(quad.num_items)):
			m.add_constraint(m.sum(x[i] for i in range(n)) == quad.num_items[k])
	elif type(quad) is HSP:
		m.add_constraint(m.sum(x[i] for i in range(n)) == quad.num_items)

	#add auxiliary constraints
	for i in range(n):
		for j in range(i+1,n):
			if(con1):
				m.add_constraint(w[i,j] <= x[i])
			if(con2):
				m.add_constraint(w[i,j] <= x[j])
			if(con3):
				m.add_constraint(x[i]+x[j]-1 <= w[i,j])
			if(con4):
				m.add_constraint(w[i,j] >= 0)

	#add objective function
	if type(quad)==HSP: #only HSP has different objective function
		quadratic_values = 0
		for i in range(n):
			for j in range(i+1,n):
				quadratic_values = quadratic_values + (w[i,j]*(C[i,j]+C[j,i]))
		m.maximize(quadratic_values)
	else:
		linear_values = m.sum(x[i]*c[i] for i in range(n))
		quadratic_values = 0
		for i in range(n):
			for j in range(i+1,n):
				quadratic_values = quadratic_values + (w[i,j]*(C[i,j]+C[j,i]))
		m.maximize(linear_values + quadratic_values)

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
	x = m.binary_var_list(n, name="binary_var")


	if type(quad) is Knapsack: #HSP and UQP don't have cap constraint
		#add capacity constraint(s)
		for k in range(quad.m):
			m.add_constraint(m.sum(x[i]*a[k][i] for i in range(n)) <= b[k])
		#k_item constraint(s) if necessary
		for k in range(len(quad.num_items)):
			m.add_constraint(m.sum(x[i] for i in range(n)) == quad.num_items[k])
	elif type(quad) is HSP:
		m.add_constraint(m.sum(x[i] for i in range(n)) == quad.num_items)

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
		u_bound_x = u_bound_m.continuous_var_list(n, ub=1)
		l_bound_x = l_bound_m.continuous_var_list(n, ub=1)
		for k in range(quad.m):
			#TODO this gives errors for HSP and UQP
			u_bound_m.add_constraint(u_bound_m.sum(u_bound_x[i]*a[k][i] for i in range(n)) <= b[k])
			l_bound_m.add_constraint(l_bound_m.sum(l_bound_x[i]*a[k][i] for i in range(n)) <= b[k])
		for j in range(n):
			u_bound_m.set_objective(sense="max", expr=u_bound_m.sum(C[i,j]*u_bound_x[i] for i in range(n)))
			l_bound_m.set_objective(sense="min", expr=l_bound_m.sum(C[i,j]*l_bound_x[i] for i in range(n)))
			u_con = u_bound_m.add_constraint(u_bound_x[j]==1)
			l_con = l_bound_m.add_constraint(l_bound_x[j]==0)
			u_bound_m.solve()
			l_bound_m.solve()
			u_bound_m.remove_constraint(u_con)
			l_bound_m.remove_constraint(l_con)
			U[j] = u_bound_m.objective_value
			L[j] = l_bound_m.objective_value
	else:
		raise Exception(bounds + " is not a valid bound type for glovers")

	#add auxiliary constrains
	if(constraints=="original"):
		z = m.continuous_var_list(keys=n,lb=-m.infinity)
		m.add_constraints(z[j] <= U[j]*x[j] for j in range(n))
		for j in range(n):
			tempsum = sum(C[i,j]*x[i] for i in range(n))
			m.add_constraint(z[j] <= tempsum - L[j]*(1-x[j]))
		if type(quad) is HSP:
			m.maximize(m.sum(z[j] for j in range(n)))
		else:
			m.maximize(m.sum(c[j]*x[j] + z[j] for j in range(n)))
	elif(constraints=="sub1"):
		s = m.continuous_var_list(keys=n)
		for j in range(n):
			tempsum = sum(C[i,j]*x[i] for i in range(n))
			m.add_constraint(s[j] >= U[j]*x[j] - tempsum + L[j]*(1-x[j]))
		m.maximize(m.sum(c[i]*x[i] + (U[i]*x[i]-s[i]) for i in range(n)))
	elif(constraints=="sub2"):
		s = m.continuous_var_list(keys=n)
		for j in range(n):
			tempsum = sum(C[i,j]*x[i] for i in range(n))
			m.add_constraint(s[j] >= -U[j]*x[j] + tempsum - L[j]*(1-x[j]))
		m.maximize(m.sum(c[i]*x[i] + m.sum(C[i,j]*x[j] for j in range(n))-L[i]*(1-x[i])-s[i] for i in range(n)))
	else:
		raise Exception(constraints + " is not a valid constraint type for glovers")

	end = timer()
	setup_time = end-start

	#return model
	return [m,setup_time]

def rlt1_linearization(quad):
	n = quad.n
	c = quad.c
	C = quad.C
	a = quad.a
	b = quad.b

	#create model and add variables
	m = Model(name='RLT-1_linearization')
	x = m.continuous_var_list(n,name='binary_var', lb=0, ub=1) #named binary_var so can easily switch for debug
	w = m.continuous_var_matrix(keys1=n, keys2=n)
	y = m.continuous_var_matrix(keys1=n, keys2=n)

	#add capacity constraint(s)
	for k in range(quad.m):
		m.add_constraint(m.sum(x[i]*a[k][i] for i in range(n)) <= b[k])

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
	linear_values = m.sum(x[j]*c[j] for j in range(n))
	quadratic_values = 0
	for j in range(n):
		for i in range(n):
			if(i==j):
				continue
			quadratic_values = quadratic_values + (C[i,j]*w[i,j])
	m.maximize(linear_values + quadratic_values)

	#return model
	return m

def glovers_linearization_ext(quad, bounds="tight", constraints="original"):
	start = timer()
	n = quad.n
	c = quad.c
	C = quad.C
	a = quad.a
	b = quad.b

	#model with rlt1, solve continuous relax and get duals to constraints 16,17
	m = rlt1_linearization(quad)
	m.solve()
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
	x = m.binary_var_list(n, name="binary_var")

	#add capacity constraint(s)
	for k in range(quad.m):
		m.add_constraint(m.sum(x[i]*a[k][i] for i in range(n)) <= b[k])

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
		u_bound_x1 = u_bound_m1.continuous_var_list(n, ub=1)
		l_bound_x1 = l_bound_m1.continuous_var_list(n, ub=1)
		for k in range(quad.m):
			u_bound_m1.add_constraint(u_bound_m1.sum(u_bound_x1[i]*a[k][i] for i in range(n)) <= b[k])
			l_bound_m1.add_constraint(l_bound_m1.sum(l_bound_x1[i]*a[k][i] for i in range(n)) <= b[k])
		u_bound_m2 = Model(name='upper_bound_model2')
		l_bound_m2 = Model(name='lower_bound_model2')
		u_bound_x2 = u_bound_m2.continuous_var_list(n, ub=1)
		l_bound_x2 = l_bound_m2.continuous_var_list(n, ub=1)
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
	#substituted constraints not yet implemented here
	# elif(constraints=="sub1"):
		# s = m.continuous_var_list(keys=n)
		# for j in range(n):
			# tempsum = sum(C[i,j]*x[i] for i in range(n))
			# m.add_constraint(s[j] >= U[j]*x[j] - tempsum + L[j]*(1-x[j]))
		# m.maximize(m.sum(c[i]*x[i] + (U[i]*x[i]-s[i]) for i in range(n)))
	# elif(constraints=="sub2"):
		# s = m.continuous_var_list(keys=n)
		# for j in range(n):
			# tempsum = sum(C[i,j]*x[i] for i in range(n))
			# m.add_constraint(s[j] >= -U[j]*x[j] + tempsum - L[j]*(1-x[j]))
		# m.maximize(m.sum(c[i]*x[i] + m.sum(C[i,j]*x[j] for j in range(n))-L[i]*(1-x[i])-s[i] for i in range(n)))
	else:
		raise Exception(constraints + " is not a valid constraint type for glovers")

	end = timer()
	setup_time = end-start

	#return model
	return [m,setup_time]

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

	#add capacity constraint
	for k in range(quad.m):
		m.add_constraint(m.sum(x[i]*a[k][i] for i in range(n)) <= b[k])

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
	linear_values = m.sum(x[j]*c[j] for j in range(n))
	quadratic_values = 0
	for j in range(n):
		for i in range(n):
			if(i==j):
				continue
			quadratic_values = quadratic_values + (C[i,j]*w[i,j])
	m.maximize(linear_values + quadratic_values)

	#return model
	return m

def reformulate_glover(quad):
	start = timer()
	m = prlt1_linearization(quad)
	duals16 = np.zeros((n,n))
	for i in range(n):
		for j in range(i+1,n):
			con_name = 'con16'+str(i)+str(j)
			duals16[i][j]=(m.dual_values(m.get_constraint_by_name(con_name)))
	C = quad.C
	for i in range(quad.n):
		for j in range(i+1,quad.n):
			duals[j,i]=C[j,i]+duals[i,j]
			duals[i,j]=C[i,j]-duals[i,j]
	quad.C = duals
	new_m = glovers_linearization(quad, bounds="tight", constraints="original")[0]
	end = timer()
	setup_time = end-start
	return [new_m, setup_time]

def solve_model(model):
	#use with block to automatically call m.end() when finished
	with model as m:
		start = timer()
		assert m.solve(), "solve failed"
		end = timer()
		solve_time = end-start
		objective_value = m.objective_value

		#compute continuous relaxation and integrality_gap
		i = 0
		while m.get_var_by_name("binary_var_"+str(i)) is not None:
			relaxation_var = m.get_var_by_name("binary_var_"+str(i))
			m.set_var_type(relaxation_var,m.continuous_vartype)
			i+=1

		assert m.solve(), "solve failed"
		continuous_obj_value = m.objective_value
	integrality_gap=((continuous_obj_value-objective_value)/objective_value)*100

	#create and return results dictionary
	results = {"solve_time":solve_time,
				"objective_value":objective_value,
				"relaxed_solution":continuous_obj_value,
				"integrality_gap":integrality_gap}
	return results

knap = Knapsack(n=30)
m = glovers_linearization_ext(knap)[0]
r = solve_model(m)
print(r.get("objective_value"))
print(r.get("relaxed_solution"))
# m.solve()
# print(m.get_solve_details().best_bound)
# print(m.get_solve_details().mip_relative_gap)
# r = solve_model(m)
# print(r.get("relaxed_solution"))
#m.solve()
#print(m.get_solve_details().best_bound)
#print(m.objective_value)


#TODO something funky going on here (with duals?). glover_Ext should minimize int_gap but it isnt!?
