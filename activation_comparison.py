from random import seed
from csv import reader
import numpy as np
import mlp_sigmod
import tanh
import matplotlib.pyplot as plt
#Part 1:Loading dataset and preprocessing methods
# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset
# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup
# Find the min and max values for each column
def dataset_minmax(dataset):	
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats
# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

def load_prepared_dataset(filename):
    seed(1)
    # load and prepare data
    dataset = load_csv(filename)
    for i in range(len(dataset[0])-1):
        str_column_to_float(dataset,i)
    # convert class column to integers
    str_column_to_int(dataset, len(dataset[0])-1)
    dataset = np.array(dataset)
    minmax = dataset_minmax(dataset)
    normalize_dataset(dataset, minmax)
    return dataset


#Part 2:sigmod
def sigmod_activation_results(dataset,test_dataset):
    n_inputs = len(dataset[0]) - 1
    n_outputs = len(set([row[-1] for row in dataset]))
    epoch_num = 500
    prediction_set = []
    label_set = []
    #Struct MLP
    network = mlp_sigmod.mlp_init(n_inputs,5,n_outputs)
    #Train test
    errors = mlp_sigmod.train(network, dataset, 0.3,n_outputs,epoch_num)
    epoches = [n for n in range(epoch_num)]
    mlp_sigmod.plot_training_errors(epoches,errors)
    for layer in network:
        print(layer)
    #Predict
    for row in test_dataset:
        prediction = mlp_sigmod.predict(network, row)
        prediction_set.append(prediction)
        label_set.append(int(row[-1]))
        print('Expected=%d, Got=%d' % (row[-1], prediction))
    #Results/Evaluation
    labels = list(set(label_set))
    class_names = np.array(labels)
    result = mlp_sigmod.plot_confusion_matrix(label_set, prediction_set, classes=class_names, normalize=False)
    plt.show()
    plt.close()
    print('The mean accuracy metric of the MLP is : ' + str(mlp_sigmod.accuracy_metric(network,test_dataset))+' %')
    return errors

#Part 3:tanh activation
def tanh_activation_results(dataset,test_dataset):
    n_inputs = len(dataset[0]) - 1
    n_outputs = len(set([row[-1] for row in dataset]))
    epoch_num = 500
    #Struct MLP
    network = tanh.mlp_init(n_inputs,5,n_outputs)
    #Train test
    errors = tanh.train(network, dataset, 0.3,n_outputs,epoch_num)
    epoches = [n for n in range(epoch_num)]
    tanh.plot_training_errors(epoches,errors)
    for layer in network:
        print(layer)
    #Predict
    prediction_set = []
    label_set = []
    for row in test_dataset:
        prediction = tanh.predict(network, row)
        prediction_set.append(prediction)
        label_set.append(int(row[-1]))
        print('Expected=%d, Got=%d' % (row[-1], prediction))
    #Results/Evaluation
    labels = list(set(label_set))
    class_names = np.array(labels)
    result = tanh.plot_confusion_matrix(label_set, prediction_set, classes=class_names, normalize=False)
    plt.show()
    plt.close()
    print('The mean accuracy metric of the MLP is : ' + str(tanh.accuracy_metric(network,test_dataset))+' %')
    return errors
#Plot both curves
def convergence_comparison(errors1,errors2,epoches):
    x = epoches
    y1 = errors1
    y2 = errors2
    plt.plot(x,y1,ls = '-',lw = 2,label = 'convergence line(sigmod)',color = 'purple')
    plt.plot(x,y2,ls = '-',lw = 2,label = 'convergence line(tanh)',color = 'blue')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('mean error')
    plt.title('Convergence graph comparison')
    plt.show()    

def main():
    epoches = [n for n in range(500)]
    dataset = load_prepared_dataset('wheat-seeds.csv')
    dataset2 = load_prepared_dataset('wheat-seeds.csv')
    errors1 = sigmod_activation_results(dataset,dataset)
    errors2 = tanh_activation_results(dataset2,dataset2)
    convergence_comparison(errors1,errors2,epoches)
    print(dataset)

if __name__=='__main__':
    main()

