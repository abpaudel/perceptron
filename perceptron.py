import sys
import numpy as np
# import matplotlib.pyplot as plt

class Perceptron():

	def __init__(self, datafile = 'input1.csv', outfile = 'output1.csv'):
		data = np.genfromtxt(datafile, delimiter = ',')
		self.outfile = outfile
		self.x = np.ones(data.shape)
		self.x[:, 1:] = data[:, :2]
		self.y = data[:, 2]
		self.w = np.zeros(3)

	def train(self):
		file = open(self.outfile, 'w')
		for i in range(50):
			w_prev = np.array(self.w)
			for x, y in zip(self.x, self.y):
				pred = self.predict(x)
				error = y - pred
				if error:
					self.w = self.w + error * x
			# self.datavis()
			file.write(str(self.w[1]) + ',' + str(self.w[2]) + ',' + str(self.w[0]) + '\n')
			if np.array_equal(w_prev, self.w):
				return

	def predict(self, x):
		y = np.dot(self.w, np.array(x))
		return self.activation(y)

	def activation(self, x):
		return 1 if x > 0 else -1

	def datavis(self):
		color = ['blue' if l == 1 else 'red' for l in self.y]
		plt.scatter(self.x[:, 1], self.x[:, 2], color = color)
		x1 = np.linspace(np.min(self.x[:,2]), np.max(self.x[:,2]), 100)
		x2 = (-self.w[0] - self.w[1] * x1)/self.w[2]
		plt.plot(x1, x2)
		plt.show()

def main():
	#p = Perceptron(sys.argv[1], sys.argv[2])
	p= Perceptron()
	p.train()


if __name__ == "__main__":
	main()