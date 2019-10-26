import numpy as np


class ClassificationTree:

	def gini_index(self, groups, classes):
		"""
		Calculates the gini index for the classes and the groups
		"""
		# Count the total instances in the group
		n_instances = float(sum([len(group) for group in groups]))
		gini = 0.0
		for group in groups:
			size = float(len(group))
			if size == 0:
				continue

			# Calculate the score for this group
			score = 0.0
			for _class in classes:
				p = len(np.where(group[:, -1]==_class)[0]) / size
				score += p * p

			gini += (1.0 - score) * (size / n_instances)
		return gini

	def _split(self, index, value, dataset):
		"""
		Splits the dataset into two groups based on index and value
		"""
		left = dataset[np.where(dataset[:, index] < value)[0]]
		right = dataset[np.where(dataset[:, index] >= value)[0]]
		return left, right

	def get_split(self, dataset):
		"""
		Search exhaustively and greedily for the best split in all the possible splits.
		The split is decided on the basis of the gini index for each split.
		"""
		# Get a list of the class labels
		_labels = np.unique(dataset[:, -1])
		_index, _value, _score, _groups = 999, 999, 999, None # random initialization
		for index in range(len(dataset[0])-1): # -1 because last column is class label
			for row in dataset:
				groups = self._split(index, row[index], dataset)
				gini = self.gini_index(groups, _labels)
				if gini < _score:
					_index = index
					_value = row[index]
					_score = gini
					_groups = groups

		return {'index': _index, 'value': _value, 'groups': _groups}

	def to_terminal(self, group):
		"""
		given a group return the label associated with that group. 
		Common strategy (used here) is to use the most frequent label in the group.
		"""
		outcomes, counts = np.unique(group[:, -1], return_counts=True)
		return outcomes[np.argmax(counts)]

	def split(self, node, max_depth, min_size, depth):
		"""
		Recursively split and build a tree.
		"""
		left, right = node['groups']
		del node['groups']

		# Check if one of the groups is empty
		if len(left) == 0 or len(right) == 0:
			node['left'] = node['right'] = self.to_terminal(np.vstack((left, right)))
			return

		# check if maximum depth exceeded
		if depth > max_depth:
			node['left'] = node['right'] = self.to_terminal(np.vstack((left, right)))
			return

		# Process left child
		if len(left) < min_size:
			node['left'] = self.to_terminal(left)
		else:
			node['left'] = self.get_split(left)

			self.split(node['left'], max_depth, min_size, depth + 1)

		# Process right child
		if len(right) < min_size:
			node['right'] = self.to_terminal(right)
		else:
			node['right'] = self.get_split(right)

			self.split(node['right'], max_depth, min_size, depth + 1)

	def build_tree(self, dataset, max_depth, min_size):
		"""
		wrapper for split
		"""
		root = self.get_split(dataset)
		self.split(root, max_depth, min_size, 1)
		return root

	def predict(self, tree, row):
		"""
		Make prediction for the row
		"""
		if row[tree['index']] < tree['value']:
			if isinstance(tree['left'], dict):
				return self.predict(tree['left'], row)
			else:
				return tree['left']
		else:
			if isinstance(tree['right'], dict):
				return self.predict(tree['right'], row)
			else:
				return tree['right']


if __name__=='__main__':
	c = ClassificationTree()
	