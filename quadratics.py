import numpy as np
import random

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
		self.num_items = []
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

class Knapsack(Quadratic): #includes QKP, KQKP, QMKP
	def __init__(self, seed=0, n=10, m=1, density=100, symmetric=False, k_item=False):
		super().__init__(seed_=seed, n_=n, m_=m, density_=density, symmetric_=symmetric, k_item_=k_item)

		#item weights
		self.a = np.random.randint(low=1, high=51, size=(m, n))
		#knapsack capacity and KQKP num_items constraints (multiple if QMKP)
		if k_item:
			for i in range(m):
				self.num_items.append(np.random.randint(low=2, high=int(n/4)+1))
				self.b.append(np.random.randint(low=50, high=30*self.num_items[i])) #can get low>high errors
		else:
			for i in range(m):
				self.b.append(np.random.randint(low=50, high=np.sum(self.a[i])+1))
		#item values
		for i in range(n):
			if(np.random.randint(1,101) <= density):
				self.c[i] = np.random.randint(low=1, high=101)
		#pair values
		for i in range(n):
			for j in range(i+1,n):
				if(np.random.randint(1,101) <= density):
					self.C[i,j] = np.random.randint(low=1, high=101)
					if(symmetric):
						self.C[i,j] = self.C[i,j]/2
						self.C[j,i] = self.C[i,j]

class UQP(Quadratic): #unconstrained 0-1 quadratic program
	"""
	:objective: Linear and Quadratic values. Same objective as QKP. C[i,j] Can be negative!
	:constraints: NO CONSTRAINTS (--> trivial if C[i,j] is positive for all i,j)
	"""
	def __init__(self, seed=0, n=10, m=1, density=100, symmetric=True):
		super().__init__(seed_=seed, n_=n, m_=m, density_=density, symmetric_=symmetric)
		#item values
		for i in range(n):
			if(np.random.randint(1,101) <= density):
				self.c[i] = np.random.randint(low=-100, high=101)
		#pair values
		for i in range(n):
			for j in range(i+1,n):
				if(np.random.randint(1,101) <= density):
					self.C[i,j] = np.random.randint(low=-100, high=101)
					if(symmetric):
						self.C[i,j] = self.C[i,j]/2
						self.C[j,i] = self.C[i,j]

class HSP(Quadratic): #heaviest k-subgraph problem
	"""
	:objective: Only has quadratic values (no linear coefficients). C[i,j] = 0 or 1
	:constraints: K-item constraint. Must take exactly k items
	"""
	def __init__(self, seed=0, n=16, density=50, symmetric=True, k_ratio=0.5):
		super().__init__(seed_=seed, n_=n, density_=density, symmetric_=symmetric)
		self.num_items = int(k_ratio*n)
		#pair values
		for i in range(n):
			for j in range(i+1,n):
				if(np.random.randint(1,101) <= density):
					self.C[i,j] = 1
					if(symmetric):
						self.C[j,i] = 1
