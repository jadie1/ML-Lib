# Jadie Adams
import DecisionTree

def getData(csvFile):
	data = []
	with open(csvFile, 'r') as f:
		for line in f:
			data.append(line.strip().split(','))
	return data


if __name__ == "__main__":
	train_data = getData("car/train.csv")
	test_data = getData("car/test.csv")