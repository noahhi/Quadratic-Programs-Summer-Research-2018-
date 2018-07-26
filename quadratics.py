import numpy as np
import random
import sys
from docplex.mp.model import Model
from timeit import default_timer as timer

#copy of cplex glovers used for refined_reorder method
#NOTE: trying to import cplex --> circular import errors
def glovers_linearization(quad, bounds="tight", constraints="original", lhs_constraints=False, use_diagonal=False, solve_continuous=False, **kwargs):
	n = quad.n
	c = quad.c
	C = quad.C
	a = quad.a
	b = quad.b

	# put linear terms along diagonal of quadratic matrix. set linear terms to zero
	if use_diagonal:
		for i in range(n):
			C[i, i] = c[i]
			c[i] = 0

	# create model and add variables
	m = Model(name='glovers_linearization_'+bounds+'_'+constraints)
	if solve_continuous:
		x = m.continuous_var_list(n, name="decision_var")
	else:
		x = m.binary_var_list(n, name="binary_var")

	if type(quad) is Knapsack:	# HSP and UQP don't have cap constraint
		# add capacity constraint(s)
		for k in range(quad.m):
			m.add_constraint(m.sum(x[i]*a[k][i] for i in range(n)) <= b[k])

	# k_item constraint if necessary (if KQKP or HSP)
	if quad.num_items > 0:
		m.add_constraint(m.sum(x[i] for i in range(n)) == quad.num_items)

	# determine bounds for each column of C
	# U1,L1 must take item at index j, U0,L0 must not take
	U1 = np.zeros(n)
	L0 = np.zeros(n)
	U0 = np.zeros(n)
	L1 = np.zeros(n)
	start = timer()
	if(bounds == "original" or type(quad) == UQP):
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
	elif(bounds == "tight" or bounds == "tighter"):
		u_bound_m = Model(name='upper_bound_model')
		l_bound_m = Model(name='lower_bound_model')
		if bounds == "tight":
			# for tight bounds solve for upper bound of col w/ continuous vars
			u_bound_x = u_bound_m.continuous_var_list(n, ub=1, lb=0)
			l_bound_x = l_bound_m.continuous_var_list(n, ub=1, lb=0)
		elif bounds == "tighter":
			# for even tighter bounds, instead use binary vars
			u_bound_x = u_bound_m.binary_var_list(n, ub=1, lb=0)
			l_bound_x = l_bound_m.binary_var_list(n, ub=1, lb=0)
		if type(quad) is Knapsack:
			for k in range(quad.m):
				# add capacity constraints
				u_bound_m.add_constraint(u_bound_m.sum(
					u_bound_x[i]*a[k][i] for i in range(n)) <= b[k])
				l_bound_m.add_constraint(l_bound_m.sum(
					l_bound_x[i]*a[k][i] for i in range(n)) <= b[k])
		if quad.num_items > 0:
			u_bound_m.add_constraint(u_bound_m.sum(
				u_bound_x[i] for i in range(n)) == quad.num_items)
			l_bound_m.add_constraint(l_bound_m.sum(
				l_bound_x[i] for i in range(n)) == quad.num_items)
		for j in range(n):
			# for each col, solve model to find upper/lower bound
			u_bound_m.set_objective(sense="max", expr=u_bound_m.sum(
				C[i, j]*u_bound_x[i] for i in range(n)))
			l_bound_m.set_objective(sense="min", expr=l_bound_m.sum(
				C[i, j]*l_bound_x[i] for i in range(n)))
			u_con = u_bound_m.add_constraint(u_bound_x[j] == 1)
			l_con = l_bound_m.add_constraint(l_bound_x[j] == 0)
			u_bound_m.solve()
			if "OPTIMAL_SOLUTION" not in str(u_bound_m.get_solve_status()):
				#print(str(u_bound_m.get_solve_status()) + " when solving for upper bound (U1)")
				u_bound_m.remove_constraint(u_con)
				u_bound_m.add_constraint(u_bound_x[j] == 0)
				m.add_constraint(x[j] == 0)
				u_bound_m.solve()
			else:
				u_bound_m.remove_constraint(u_con)
			l_bound_m.solve()
			if "OPTIMAL_SOLUTION" not in str(l_bound_m.get_solve_status()):
				#print(str(l_bound_m.get_solve_status()) + " when solving for lower bound (L0)")
				l_bound_m.remove_constraint(l_con)
				l_bound_m.add_constraint(l_bound_x[j] == 1)
				m.add_constraint(x[j] == 1)
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
					#print(str(u_bound_m.get_solve_status()) + " when solving for upper bound (U0)")
					u_bound_m.remove_constraint(u_con)
					u_bound_m.add_constraint(u_bound_x[j] == 1)
					m.add_constraint(x[j] == 1)
					u_bound_m.solve()
				else:
					u_bound_m.remove_constraint(u_con)
				l_bound_m.solve()
				if "OPTIMAL_SOLUTION" not in str(l_bound_m.get_solve_status()):
					#print(str(l_bound_m.get_solve_status()) + " when solving for lower bound (L1)")
					l_bound_m.remove_constraint(l_con)
					l_bound_m.add_constraint(l_bound_x[j] == 0)
					m.add_constraint(x[j] == 0)
					l_bound_m.solve()
				else:
					l_bound_m.remove_constraint(l_con)
				U0[j] = u_bound_m.objective_value
				L1[j] = l_bound_m.objective_value
	else:
		raise Exception(bounds + " is not a valid bound type for glovers")
	end = timer()
	setup_time = end-start

	# add auxiliary constrains
	if(constraints == "original"):
		# original glovers constraints
		z = m.continuous_var_list(keys=n, lb=-m.infinity)
		m.add_constraints(z[j] <= U1[j]*x[j] for j in range(n))
		if lhs_constraints:
			m.add_constraints(z[j] >= L1[j]*x[j] for j in range(n))
		for j in range(n):
			tempsum = sum(C[i, j]*x[i] for i in range(n))
			m.add_constraint(z[j] <= tempsum - L0[j]*(1-x[j]))
			if lhs_constraints:
				m.add_constraint(z[j] >= tempsum - U0[j]*(1-x[j]))
		m.maximize(m.sum(c[j]*x[j] + z[j] for j in range(n)))

	elif(constraints == "sub1" or constraints == "sub2"):
		# can make one of 2 substitutions using slack variables to further reduce # of constraints
		s = m.continuous_var_list(keys=n, lb=0)
		for j in range(n):
			tempsum = sum(C[i, j]*x[i] for i in range(n))
			if constraints == "sub1":
				m.add_constraint(s[j] >= U1[j]*x[j] - tempsum + L0[j]*(1-x[j]))
			else:
				m.add_constraint(s[j] >= -U1[j]*x[j] +
								 tempsum - L0[j]*(1-x[j]))
		if constraints == "sub1":
			m.maximize(m.sum(c[i]*x[i] + (U1[i]*x[i]-s[i]) for i in range(n)))
		else:
			m.maximize(sum(c[j]*x[j] for j in range(n)) + sum(sum(C[i, j]*x[i]
																  for i in range(n))-L0[j]*(1-x[j])-s[j] for j in range(n)))
	else:
		raise Exception(
			constraints + " is not a valid constraint type for glovers")

	# return model
	return [m, setup_time]

def qsap_glovers(qsap, bounds="original", constraints="original", lhs_constraints=False, solve_continuous=False, **kwargs):
	"""
	glovers linearization for the quadratic semi-assignment problem
	"""
	n = qsap.n
	m = qsap.m
	e = qsap.e
	c = qsap.c
	mdl = Model(name='qsap_glovers')
	if solve_continuous:
		x = mdl.continuous_var_matrix(keys1=m,keys2=n,name="decision_var")
	else:
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
		for i in range(m):
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
		for i in range(m):
			for k in range(n):
				# solve for upper bound U1
				bound_mdl.set_objective(sense="max", expr=sum(sum(c[i,k,j,l]*bound_x[j,l] for l in range(n)) for j in range(m)))
				u_con = bound_mdl.add_constraint(bound_x[i,k]==1)
				bound_mdl.solve()
				bound_mdl.remove_constraint(u_con)
				U1[i,k] = bound_mdl.objective_value

				# solve for lower bound L0
				bound_mdl.set_objective(sense="min", expr=sum(sum(c[i,k,j,l]*bound_x[j,l] for l in range(n)) for j in range(m)))
				l_con = bound_mdl.add_constraint(bound_x[i,k]==0)
				bound_mdl.solve()
				bound_mdl.remove_constraint(l_con)
				L0[i,k] = bound_mdl.objective_value

				if lhs_constraints:
					# solve for upper bound U0
					bound_mdl.set_objective(sense="max", expr=sum(sum(c[i,k,j,l]*bound_x[j,l] for l in range(n)) for j in range(m)))
					u_con = bound_mdl.add_constraint(bound_x[i,k] == 0)
					bound_mdl.solve()
					bound_mdl.remove_constraint(u_con)
					U0[i,k] = bound_mdl.objective_value

					# solve for lower bound L1
					bound_mdl.set_objective(sense="min", expr=sum(sum(c[i,k,j,l]*bound_x[j,l] for l in range(n)) for j in range(m)))
					l_con = bound_mdl.add_constraint(bound_x[i,k] == 1)
					bound_mdl.solve()
					L1[i,k] = bound_mdl.objective_value
					bound_mdl.remove_constraint(l_con)
		# end bound model
		bound_mdl.end()
	else:
		raise Exception(bounds + " is not a valid bound type for glovers")
	end = timer()
	setup_time = end-start

	# for i in range(m):
	# 	for j in range(n):
	# 		U1[i,j] = 1000
	# 		L0[i,j] = -1000

	#add auxiliary constrains
	if constraints=="original": #TODO make sure symmetric works everywhere
		z = mdl.continuous_var_matrix(keys1=m,keys2=n,lb=-mdl.infinity)
		mdl.add_constraints(z[i,k] <= x[i,k]*U1[i,k] for i in range(m) for k in range(n))
		mdl.add_constraints(z[i,k] <= sum(sum(c[i,k,j,l]*x[j,l] for l in range(n)) for j in range(m))
										-L0[i,k]*(1-x[i,k]) for i in range(m) for k in range(n))
		if lhs_constraints:
			mdl.add_constraints(z[i,k] >= x[i,k]*L1[i,k] for i in range(m) for k in range(n))
			mdl.add_constraints(z[i,k] >= sum(sum(c[i,k,j,l]*x[j,l] for l in range(n)) for j in range(m))
										-U0[i,k]*(1-x[i,k]) for i in range(m) for k in range(n))
		mdl.maximize(sum(sum(e[i,k]*x[i,k] for k in range(n))for i in range(m))
					+ sum(sum(z[i,k] for k in range(n)) for i in range(m)))
	elif constraints=="sub1":
		s = mdl.continuous_var_matrix(keys1=m,keys2=n,lb=0)
		mdl.add_constraints(s[i,k] >= U1[i,k]*x[i,k]+L0[i,k]*(1-x[i,k])-sum(sum(c[i,k,j,l]*x[j,l] for l in range(n)) for j in range(m))
						for k in range(n) for i in range(m))
		mdl.maximize(sum(sum(e[i,k]*x[i,k] for k in range(n))for i in range(m))
					+ sum(sum(U1[i,k]*x[i,k]-s[i,k] for k in range(n)) for i in range(m)))
	elif constraints=="sub2":
		s = mdl.continuous_var_matrix(keys1=m,keys2=n,lb=0)
		mdl.add_constraints(s[i,k] >= -L0[i,k]*(1-x[i,k])-(x[i,k]*U1[i,k])+sum(sum(c[i,k,j,l]*x[j,l] for l in range(n)) for j in range(m))
		 				for k in range(n) for i in range(m))
		mdl.maximize(sum(sum(e[i,k]*x[i,k] for k in range(n))for i in range(m))
					+ sum(sum(-s[i,k]-(L0[i,k]*(1-x[i,k])) + sum(sum(c[i,k,j,l]*x[j,l] for l in range(n))
					for j in range(m)) for k in range(n)) for i in range(m)))
	else:
		raise Exception(constraints + " is not a valid constraint type for glovers")

	#return model
	return [mdl,setup_time]


class Quadratic:  # base class for all quadratic problem types
	def __init__(self, seed_=0, n_=10, m_=1, density_=100, symmetric_=False, k_item_=False):
		"""
		:param seed: The seed used for random number generation
		:param n: The number of items in the knapsack
		:param m: For multiple knapsack constraints
		:param density: The density of non-zero value items in the knapsack, value between 1-100
		:param symmetric: Determines if coefficient matrix should be symmetric or upper triangular
		:param k_item: Turns into a k-item Quadratic Knapsack Problem
		"""
		# declare/initialize common attributes
		self.seed = seed_
		np.random.seed(seed_)
		self.n = n_
		self.m = m_
		self.density = density_
		self.symmetric = symmetric_
		self.k_item = k_item_
		self.num_items = -1
		self.a = []
		self.b = []
		self.c = np.zeros(n_)
		self.C = np.zeros((n_, n_))

	def print_info(self, print_C=True):
		"""
		print all relevant information to a problem instance

		param print_C: can choose to not print coefficient matrix to avoid cluterring console
		"""
		print('\nPROBLEM INSTANCE INFO:')
		print('Type: ' + str(type(self)))
		print('n = ' + str(self.n) + ' (number of items)')
		print('m = ' + str(self.m) + ' (multiple knapsacks)')
		print('b = ' + str(self.b) + ' (capacity constraint(s))')
		print('k = ' + str(self.num_items) + ' (num_items to choose)')
		print('a = ' + str(self.a) + ' (item weights)')
		print('c = ' + str(self.c) + ' (item values)')
		# may not want to print C if too big for console
		if print_C:
			print('quadratic objective coeff vector C = \n' + str(self.C))
		print("")

	def reorder(self, take_max=False, flip_order=False, vm=None):
		C = self.C
		n = self.n

		#make a copy of C where C[j,i]=C[i,j]
		symC_copy = np.copy(C)
		i_lower = np.tril_indices(n)
		symC_copy[i_lower] = symC_copy.T[i_lower]

		if type(vm) != np.ndarray:
			#if no value_matrix is passed --> doing basic/inital reorder. VM=abs(C)
			value_matrix = np.absolute(symC_copy)
		else:
			#if a value_matrix is passed, we are doing refined_reorder. Use this VM
			value_matrix = vm

		col_sums, new_order = [], []
		for i in range(self.n):
			col_sums.append(sum(value_matrix[i]))
		# TODO this is a bit ineffiecient
		for i in range(self.n):
			if take_max:
				low_index = col_sums.index(max(col_sums))
				col_sums[low_index] = -sys.maxsize
			else:
				low_index = col_sums.index(min(col_sums))
				col_sums[low_index] = sys.maxsize
			new_order.append(low_index)
			for j in range(self.n):
				col_sums[j] -= value_matrix[low_index][j]

		if flip_order:
			new_order.reverse()

		#apply new order to coefficient matrix C
		for i in range(n):
			for j in range(i+1, n):
				C[i, j] = symC_copy[new_order[i], new_order[j]]

		#apply new order to c
		# TODO possible this could cause issues(make sure c is reordered correctly)
		self.c = [self.c[i] for i in new_order]

		#apply new order to knapsack constraints if multiple
		if type(self) == Knapsack:
			for j in range(self.m):
				self.a[j] = [self.a[j][i] for i in new_order]

	def reorder_refined(self, k, take_max=False, flip_order=False):
		n = self.n

		# start by applying the original reorder method
		self.reorder(take_max=take_max, flip_order=flip_order)

		for iter in range(k):
			# model and solve continuous relax
			m = glovers_linearization(
				self, solve_continuous=True)[0]
			m.solve()
			value_matrix = np.zeros((n, n))
			for i in range(n-1):
				for j in range(i+1, n):
					value_matrix[i, j] = self.C[i, j]*m.get_var_by_name("decision_var_"+str(
						i)).solution_value*m.get_var_by_name("decision_var_"+str(j)).solution_value
					if self.C[i, j] < 0:
						value_matrix[i, j] = value_matrix[i, j]*-1
					value_matrix[j, i] = value_matrix[i, j]
			self.reorder(vm=value_matrix, take_max=take_max, flip_order=flip_order)


class Knapsack(Quadratic):	# includes QKP, KQKP, QMKP
	def __init__(self, seed=0, n=10, m=1, density=100, symmetric=False, k_item=False, mixed_sign=False):
		super().__init__(seed_=seed, n_=n, m_=m, density_=density,
						 symmetric_=symmetric, k_item_=k_item)

		if k_item:
			self.b.append(-1)
			while(True):
				# item weights
				self.a = np.random.randint(low=1, high=101, size=(m, n))
				self.num_items = (np.random.randint(low=2, high=int(n/4)+1))
				#self.b.append(np.random.randint(low=50, high=np.sum(self.a)))
				self.b[0] = (np.random.randint(
					low=50, high=(30*self.num_items)+1))
				copy = self.a.copy()
				copy.sort()
				copy = copy[0].tolist()
				count = 0
				i = 0
				kmax = n
				for weight in copy:
					if(count+weight > self.b[0]):
						kmax = i
						break
					i += 1
					count += weight
				if self.num_items < kmax:
					break
		else:
			self.a = np.random.randint(low=1, high=101, size=(m, n))
			# knapsack capacity(s)
			for i in range(m):
				# find heaviest item. make sure cap constraint allows for taking it
				amax = self.a[i].max()
				# Note: upper bound used in org paper for kitem was: high=30*self.num_items
				potential_cap_con = np.random.randint(
					low=50, high=np.sum(self.a[i])+1)
				while potential_cap_con < amax:
					print("had to come up with new capacity constraint")
					potential_cap_con = np.random.randint(
						low=50, high=np.sum(self.a[i])+1)
				self.b.append(potential_cap_con)

		# item values
		for i in range(n):
			if(np.random.randint(1, 101) <= density):
				self.c[i] = np.random.randint(low=1, high=101)
				#self.c[i] = np.random.normal(50,10)
				if mixed_sign and np.random.choice([0, 1]):
					self.c = self.c * -1
		# pair values
		for i in range(n):
			for j in range(i+1, n):
				if(np.random.randint(1, 101) <= density):
					self.C[i, j] = np.random.randint(low=1, high=101)
					#self.C[i,j] = np.random.normal(50,10)
					if mixed_sign and np.random.choice([0, 1]):
						self.C = self.C * -1
					if(symmetric):
						self.C[i, j] = self.C[i, j]/2
						self.C[j, i] = self.C[i, j]


class UQP(Quadratic):  # unconstrained 0-1 quadratic program
	"""
	:objective: Linear and Quadratic values. Same objective as QKP. C[i,j] Can be negative!
	:constraints: NO CONSTRAINTS (--> trivial if C[i,j] is positive for all i,j)
	"""

	def __init__(self, seed=0, n=10, m=1, density=100, symmetric=False):
		super().__init__(seed_=seed, n_=n, m_=m, density_=density, symmetric_=symmetric)
		# #item values
		# for i in range(n):
		#	if(np.random.randint(1,101) <= density):
		#		self.c[i] = np.random.randint(low=-100, high=101)
		# #pair values
		# for i in range(n):
		#	for j in range(i+1,n):
		#		if(np.random.randint(1,101) <= density):
		#			self.C[i,j] = np.random.randint(low=-100, high=101)
		#			if(symmetric):
		#				self.C[i,j] = self.C[i,j]/2
		#				self.C[j,i] = self.C[i,j]

		# Boolean least squares problem generation
		D = np.asmatrix(np.random.normal(0, 1, (n, n)))
		y = np.asmatrix(np.random.randint(0, 2, n)).transpose()
		d = D@y + np.random.normal(0, 1)
		Q = D@D.T
		q = -2*d.T@D
		for i in range(n):
			self.c[i] = -(q.item(i) + Q.item((i, i)))
			for j in range(i+1, n):
				self.C[i, j] = -(2*Q[i, j])
				# TODO symmetric for glovers only
				if symmetric:
					self.C[i, j] = 0.5*self.C[i, j]
					self.C[j, i] = self.C[i, j]

		# print(self.c)
		# print(self.C)


class HSP(Quadratic):  # heaviest k-subgraph problem
	"""
	:objective: Only has quadratic values (no linear coefficients). C[i,j] = 0 or 1
	:constraints: K-item constraint. Must take exactly k items
	"""

	def __init__(self, seed=0, n=16, density=50, symmetric=False, k_ratio=0.5):
		super().__init__(seed_=seed, n_=n, density_=density, symmetric_=symmetric)
		self.num_items = int(k_ratio*n)
		# pair values
		for i in range(n):
			for j in range(i+1, n):
				if(np.random.randint(1, 101) <= density):
					self.C[i, j] = 1
					if symmetric:
						self.C[i, j] = 0.5
						self.C[j, i] = 0.5


class QSAP:	 # quadratic semi assignment problem
	def __init__(self, seed=0, n=3, m=10, symmetric=False):
		"""
		param n: number of processors
		param m: number of tasks (each task is assigned to exactly one processor)
		"""
		self.seed = seed
		np.random.seed(seed)
		self.n = n
		self.m = m
		self.e = np.zeros((m, n))
		for i in range(m):
			for k in range(n):
				self.e[i, k] = np.random.randint(-50, 51)

		self.c = np.zeros((m, n, m, n))
		for i in range(m-1):
			for k in range(n):
				for j in range(i+1, m):
					for l in range(n):
						self.c[i, k, j, l] = np.random.randint(-50, 51)
						if symmetric:
							self.c[i, k, j, l] = self.c[i, k, j, l]*0.5
							self.c[j, l, i, k] = self.c[i, k, j, l]

	def reorder(self, take_max=False, flip_order=False, vm=None):
		#C = self.c
		n = self.n
		m = self.m
		e = self.e
		if type(vm) != np.ndarray:
			# generate value matrix. a symmetric copy of quadratic terms matrix (all positive)
			value_matrix = np.zeros(shape=(m, n, m, n))
			for i in range(m-1):
				for k in range(n):
					for j in range(i+1, m):
						for l in range(n):
							val = self.c[i, k, j, l]
							value_matrix[i, k, j, l] = abs(val)
							value_matrix[j, l, i, k] = abs(val)
		else:
			value_matrix=vm

		#print(value_matrix)
		col_sums = np.zeros((m, n))
		old_order = []
		new_order = []
		map = {}
		for i in range(m):
			for k in range(n):
				old_order.append((i, k))
				# compute 'column' sums. (ie. sum of all j,l terms for an i,k pair)
				col_sums[i, k] = sum(value_matrix[i, k, j, l] for j in range(m) for l in range(n))

		# find the minimum col_sum, and record its index as the next elem in our new order
		for i in range(n*m):
			minv = col_sums[0, 0]
			mindex = (0, 0)
			for (x, y), v in np.ndenumerate(col_sums):
				if take_max:
					if v > minv:
						minv = v
						mindex = (x, y)
				else:
					if v < minv:
						minv = v
						mindex = (x, y)
			if take_max:
				col_sums[mindex[0], mindex[1]] = -sys.maxsize
			else:
				# need to ignore/remove current mindex
				col_sums[mindex[0], mindex[1]] = sys.maxsize
			new_order.append(mindex)
			#map[old_order[i]] = mindex
			for i in range(m):
				for k in range(n):
					# update other terms according to paper
					col_sums[i, k] -= value_matrix[i, k, mindex[0], mindex[1]]

		if flip_order:
			new_order.reverse()

		for i in range(n*m):
			map[old_order[i]] = new_order[i]

		C = np.zeros((m,n,m,n))
		for (x,y) in new_order:
			for j in range(m):
				for l in range(n):
					C[x,y,j,l] = self.c[j,l,x,y]
					self.c[j,l,x,y] = 0
			for j in range(m):
				for l in range(n):
					C[x,y,j,l] += self.c[x,y,j,l]


					self.c[x,y,j,l] = 0

		self.c = C
		# apply reordering to quadratic terms matrix
		# for i in range(m):
		# 	for k in range(n):
		# 		for j in range(m):
		# 			for l in range(n):
		# 				(x1, y1) = map[(i, k)]
		# 				(x2, y2) = map[(j, l)]
		# 				C[i, k, j, l] = neg_mask[x1, y1, x2, y2]

						#p.c[1,0,1,1] = -29
						#p.c[0,0,0,1] = 8
		# apply reordering to linear terms matrix
		#e = np.array([e[i, k] for (i, k) in new_order]).reshape(m,n)
		#self.e = e

	def reorder_refined(self, k, take_max=False, flip_order=False):
		n = self.n
		m = self.m

		# start by applying the original reorder method
		self.reorder(take_max=take_max, flip_order=flip_order)

		for iter in range(k):
			# model and solve continuous relax
			mdl = qsap_glovers(self, solve_continuous=True)[0]
			mdl.solve()
			#print(mdl.solution)
			value_matrix = np.zeros(shape=(m, n, m, n))
			for i in range(m-1):
				for k in range(n):
					for j in range(i+1, m):
						for l in range(n):
							value_matrix[i,k,j,l] = self.c[i,k,j,l]*mdl.get_var_by_name("decision_var_"+str(
								i)+"_"+str(k)).solution_value*mdl.get_var_by_name("decision_var_"+str(j)+"_"+str(l)).solution_value
							if self.c[i,k,j,l] < 0:
								value_matrix[i,k,j,l] = value_matrix[i,k,j,l]*-1
							value_matrix[j,l,i,k] = value_matrix[j,l,i,k]
			self.reorder(vm=value_matrix, take_max=take_max, flip_order=flip_order)
