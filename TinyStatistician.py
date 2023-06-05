import numpy as np

class TinyStatistician:
	"""
		• mean(x), median(x), quartile(x), percentile(x, p), var(x), std(x)
	"""
	def mean(self, x):
		if len(x) == 0:
			return None
		result = 0.0
		for elem in x:
			result += elem
		return result / len(x)

	def median(self, x):
		numbers = sorted(x)
		middle = len(numbers) // 2
		if len(numbers) % 2 == 0:

			return float((numbers[middle - 1] + numbers[middle]) / 2)
		else:
			return float(numbers[middle])

	def quartile(self, x):
		numbers = sorted(x)
		n = len(x)
		# q1 > 1/4 * n
		q1 = self.median(numbers[:(n + 1)//2])
		# q3 > 2/4 * n
		q3 = self.median(numbers[(n + 1)//2:])
		return [q1, q3]

	def percentile(self, x, p):
		if len(x) == 0:
			return None
		sorted_x = sorted(x)
		index = (len(x) - 1) * (p / 100.0)
		if index.is_integer():
			return sorted_x[int(index)]
		else:
			k = int(index) # the index of lower value nearest to the desired percentile
			d = index - k # decimal part of index
			return sorted_x[k] * (1 - d) + sorted_x[k + 1] * d

	def var(self, x):
		mean = self.mean(x)
		suqared_diff_sum = 0.0
		for num in x:
			suqared_diff_sum += (num - mean) ** 2
		#suqared_diff_sum = np.sum((num - mean) ** 2)
		return suqared_diff_sum / (len(x) - 1)

	def std(self, x):
		if len(x) == 0:
			return None
		return self.var(x) ** 0.5
	
	@staticmethod
	def min(x):
		if len(x) == 0:
			return None
		min_value = float('inf')
		for val in x:
			if val < min_value:
				min_value = val
		return min_value

	@staticmethod
	def max(x):
		if len(x) == 0:
			return None
		max_value = float('-inf')
		for val in x:
			if val > max_value:
				max_value = val
		return max_value
