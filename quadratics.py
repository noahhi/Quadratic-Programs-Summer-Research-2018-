import numpy as np
import random
import sys

class Quadratic: #base class for all quadratic problem types
	def __init__(self, seed_=0, n_=10, m_=1, density_=100, symmetric_=False, k_item_=False):
		"""
		:param seed: The seed used for random number generation
		:param n: The number of items in the knapsack
		:param m: For multiple knapsack constraints
		:param density: The density of non-zero value items in the knapsack, value between 1-100
		:param symmetric: Determines if coefficient matrix should be symmetric or upper triangular
		:param k_item: Turns into a k-item Quadratic Knapsack Problem
		"""
		#declare/initialize common attributes
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
		self.C = np.zeros((n_,n_))

	def print_info(self, print_C = False):
		#print all relevant information to a problem instance
		print('\nPROBLEM INSTANCE INFO:')
		print('Type: ' + str(type(self)))
		print('n = ' + str(self.n) + ' (number of items)')
		print('m = ' + str(self.m) + ' (multiple knapsacks)')
		print('b = ' + str(self.b) + ' (capacity constraint(s))')
		print('k = ' + str(self.num_items) + ' (num_items to choose)')
		print('a = ' + str(self.a) + ' (item weights)')
		print('c = ' + str(self.c) + ' (item values)')
		#may not want to print C if too big for console
		if print_C:
			print('quadratic objective coeff vector C = \n' + str(self.C))
		print("")

	def reorder(self): #note this only works for nonnegative coefficients (ie. wont work for UQP atm)
		C = self.C
		n = self.n
		value_matrix = np.absolute(np.copy(self.C))
		i_lower = np.tril_indices(self.n)
		value_matrix[i_lower] = value_matrix.T[i_lower]
		col_sums, new_order = [], []
		for i in range(self.n):
			col_sums.append(sum(value_matrix[i]))
		#TODO this is a bit ineffiecient
		for i in range(self.n):
			low_index=col_sums.index(min(col_sums))
			new_order.append(low_index)
			for j in range(self.n):
				col_sums[j]-=value_matrix[low_index][j]
			col_sums[low_index]=sys.maxsize
		# for new_index,old_index in zip(new_order,range(self.n)):
		# 	for j in range(old_index):

		for i in range(n):
			for j in range(i+1,n):
				C[i,j] =  value_matrix[new_order[i],new_order[j]]
		#TODO possible this could cause issues(make sure c is reordered correctly)
		self.c = [self.c[i] for i in new_order]
		if type(self) == Knapsack:
			for j in range(self.m):
				self.a[j] = [self.a[j][i] for i in new_order]

class Knapsack(Quadratic): #includes QKP, KQKP, QMKP
	def __init__(self, seed=0, n=10, m=1, density=100, symmetric=False, k_item=False, mixed_sign=False):
		super().__init__(seed_=seed, n_=n, m_=m, density_=density, symmetric_=symmetric, k_item_=k_item)

		if k_item:
			self.b.append(-1)
			while(True):
				#item weights
				self.a = np.random.randint(low=1, high=101, size=(m, n))
				self.num_items = (np.random.randint(low=2, high=int(n/4)+1))
				#self.b.append(np.random.randint(low=50, high=np.sum(self.a)))
				self.b[0]=(np.random.randint(low=50, high=(30*self.num_items)+1))
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
					i+=1
					count+=weight
				if self.num_items < kmax:
					break
		else:
			self.a = np.random.randint(low=1, high=101, size=(m, n))
			#knapsack capacity(s)
			for i in range(m):
				#Note: upper bound used in org paper fpor kitem was: high=30*self.num_items
				self.b.append(np.random.randint(low=50, high=np.sum(self.a[i])+1))


		#item values
		for i in range(n):
			if(np.random.randint(1,101) <= density):
				self.c[i] = np.random.randint(low=1, high=101)
				#self.c[i] = np.random.normal(50,10)
				if mixed_sign and np.random.choice([0,1]):
					self.c = self.c * -1
		#pair values
		for i in range(n):
			for j in range(i+1,n):
				if(np.random.randint(1,101) <= density):
					self.C[i,j] = np.random.randint(low=1, high=101)
					#self.C[i,j] = np.random.normal(50,10)
					if mixed_sign and np.random.choice([0,1]):
						self.C = self.C * -1
					if(symmetric):
						self.C[i,j] = self.C[i,j]/2
						self.C[j,i] = self.C[i,j]

class UQP(Quadratic): #unconstrained 0-1 quadratic program
	"""
	:objective: Linear and Quadratic values. Same objective as QKP. C[i,j] Can be negative!
	:constraints: NO CONSTRAINTS (--> trivial if C[i,j] is positive for all i,j)
	"""
	def __init__(self, seed=0, n=10, m=1, density=100, symmetric=False):
		super().__init__(seed_=seed, n_=n, m_=m, density_=density, symmetric_=symmetric)
		# #item values
		# for i in range(n):
		# 	if(np.random.randint(1,101) <= density):
		# 		self.c[i] = np.random.randint(low=-100, high=101)
		# #pair values
		# for i in range(n):
		# 	for j in range(i+1,n):
		# 		if(np.random.randint(1,101) <= density):
		# 			self.C[i,j] = np.random.randint(low=-100, high=101)
		# 			if(symmetric):
		# 				self.C[i,j] = self.C[i,j]/2
		# 				self.C[j,i] = self.C[i,j]

		#Boolean least squares problem generation
		D = np.asmatrix(np.random.normal(0,1, (n,n)))
		y = np.asmatrix(np.random.randint(0,2,n)).transpose()
		d = D@y + np.random.normal(0,1)
		Q = D@D.T
		q = -2*d.T@D
		for i in range(n):
			self.c[i] = -(q.item(i) + Q.item((i,i)))
			for j in range(i+1,n):
				self.C[i,j] = -(2*Q[i,j])
				#TODO symmetric for glovers only
				if symmetric:
					self.C[i,j] = 0.5*self.C[i,j]
					self.C[j,i] = self.C[i,j]

		#print(self.c)
		#print(self.C)

class HSP(Quadratic): #heaviest k-subgraph problem
	"""
	:objective: Only has quadratic values (no linear coefficients). C[i,j] = 0 or 1
	:constraints: K-item constraint. Must take exactly k items
	"""
	def __init__(self, seed=0, n=16, density=50, symmetric=False, k_ratio=0.5):
		super().__init__(seed_=seed, n_=n, density_=density, symmetric_=symmetric)
		self.num_items = int(k_ratio*n)
		#pair values
		for i in range(n):
			for j in range(i+1,n):
				if(np.random.randint(1,101) <= density):
					self.C[i,j] = 1
					if symmetric:
						self.C[i,j] = 0.5
						self.C[j,i] = 0.5

class QSAP: #quadratic semi assignment problem
	def __init__(self, seed=0, n=3, m=10):
		"""
		param n: number of processors
		param m: number of tasks (each task is assigned to exactly one processor)
		"""
		self.seed = seed
		np.random.seed(seed)
		self.n = n
		self.m = m
		self.e = np.zeros((m,n))
		for i in range(m):
			for k in range(n):
				self.e[i,k] = np.random.randint(-50,51)

		self.c = np.zeros((m,n,m,n))
		for i in range(m-1):
			for k in range(n):
				for j in range(i+1,m):
					for l in range(n):
						self.c[i,k,j,l] = np.random.randint(-50,51)
	def reorder(self):
		C = self.c
		n = self.n
		m = self.m
		value_matrix = np.zeros(shape=(m,n,m,n))
		for i in range(m-1):
			for k in range(n):
				for j in range(i+1,m):
					for l in range(n):
						value_matrix[i,k,j,l] = abs(C[i,k,j,l])
						value_matrix[j,l,i,k] = abs(C[i,k,j,l])
		print(value_matrix)
		col_sums = np.zeros((m,n))
		new_order = []
		for i in range(m):
			for k in range(n):
				col_sums[i,k] = sum(sum(value_matrix[i,k,j,l] for j in range(m))for l in range(n))
		print(col_sums)
		for i in range(n*m):
			minv = col_sums[0,0]
			mindex = (0,0)
			for (x,y), v in np.ndenumerate(col_sums):
				if v < minv:
					minv = v
					mindex = (x,y)
			print(mindex)
			new_order.append(mindex)
			for i in range(m):
				for k in range(n):
					col_sums[i,k]-=value_matrix[i,k,mindex[0],mindex[1]]
			#col_sums[mindex[0],mindex[1]]=sys.maxsize
			col_sums[mindex[0],mindex[1]]=5000
			print(col_sums)
		print(new_order)
		# for new_index,old_index in zip(new_order,range(self.n)):
		# 	for j in range(old_index):

		for row in range(m*n):
			for col in range(m*n):
				i = new_order[row][0] #(3,1)
				k = new_order[row][1]
				j = new_order[col][0] #(2,1)
				l = new_order[col][1]
				C[i,k,j,l] =  value_matrix[new_order[i][0],new_order[i][1], new_order[j][0], new_order[j][1]]

		for i in range(m-1):
			for k in range(n):
				for j in range(i+1,m):
					for l in range(n):
						C[i,k,j,l] =  value_matrix[new_order[i+k][0],new_order[i+k][1], new_order[j][0], new_order[l][1]]
		print(C)
		#TODO possible this could cause issues(make sure c is reordered correctly)
		self.c = [self.c[i] for i in new_order]
		if type(self) == Knapsack:
			for j in range(self.m):
				self.a[j] = [self.a[j][i] for i in new_order]

p = QSAP(n=2,m=3)
p.reorder()
