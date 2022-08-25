import pandas as pd
import random
import numpy as np
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
def dataset_read(path):
    dataset = pd.read_csv(path)
    return dataset
#Part 1:Struct layers and neurons for mlp
def mlp_init(n_input,n_hidden_1,n_output):
    mlp = []
    hidden_layer_1 = []
    hidden_layer_2 = []
    output_layer = []
    #neurons in hidden layer 1
    for neuron in range(n_hidden_1):
        weights = []
        for weight in range(n_input+1):
            init_random = random.random()
            weights.append(init_random)
        hidden_layer_1.append({'weights':weights})
    mlp.append(hidden_layer_1)
    #hidden layer 2
    # for neuron in range(n_hidden_2):
    #     weights = []
    #     for weight in range(n_hidden_1+1):
    #         init_random = random.random()
    #         weights.append(init_random)
    #     hidden_layer_2.append({'weights':weights})
    # mlp.append(hidden_layer_2)
    #output layer    
    for neuron in range(n_output):
        weights = []
        for weight in range(n_hidden_1+1):
            init_random = random.random()
            weights.append(init_random)
        output_layer.append({'weights':weights})
    mlp.append(output_layer)
    return mlp

#Part 2:Define activation functions for layers of mlp
#Compute the activation of each neuron in layer
def activate(weights,inputs):
    activation = 0
    for n in range(len(weights)-1):
        activation = activation + weights[n] * inputs[n]
    activation = activation + weights[-1]
    return activation
#Transfer activation to true value(activation function)
def sigmod(activation):
    return 1/(1+np.exp(-activation))#sigmod but can also use other activation function....(rectifier transfer function)

#Compute derivative of sigmod for optimazation
def sigmod_derivative(output):
    return output*(1-output)


#Part 3:Forward propagate input through the model and get the outputs
def forward_prop(mlp,data):
    input = data
    for n in range(len(mlp)):
        update_input = []
        if n != len(mlp)-1:
            for neuron in mlp[n]:
                activation = activate(neuron['weights'],input)
                neuron['output'] = sigmod(activation)
                update_input.append(neuron['output'])
        else:
            for neuron in mlp[n]:
                activation = activate(neuron['weights'],input)
                neuron['output'] = sigmod(activation)
                update_input.append(neuron['output'])
        input = update_input
    return input



#Part 4:Back propagate the errors which can be computed by difference between labels and model outputs through the model
# Backpropagate error and store in neurons
def backward_propagate_error(mlp, labels):
	for i in reversed(range(len(mlp))):
		layer = mlp[i]
		errors = list()
		if i != len(mlp)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in mlp[i + 1]:
					error += (neuron['weights'][j] * neuron['del'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(neuron['output'] - labels[j])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['del'] = errors[j] * sigmod_derivative(neuron['output'])



#Part 5:With error back propagating,the weights in the model can be updated according to error and input
#Using stochastic gradient descent(SDG) to update weights(weight = weight - learning_rate * error * input)
def weights_update(mlp,data,lr):#
    for n in range(len(mlp)):
        if n == 0:
            input = data[:-1]#The first layer is input layer,so rows of data can be considered as input/
            for neuron in mlp[n]:
                for t in range(len(input)):
                    neuron['weights'][t] = neuron['weights'][t] - (lr * neuron['del'] * input[t])
                neuron['weights'][-1] = neuron['weights'][-1] - (lr * neuron['del'])
        else:
            input = [neuron['output'] for neuron in mlp[n-1]]#consider the output of neuron from last layer as output
            for neuron in mlp[n]:
                for t in range(len(input)):
                    neuron['weights'][t] = neuron['weights'][t] - (lr * neuron['del'] * input[t])
                neuron['weights'][-1] = neuron['weights'][-1] - (lr * neuron['del'])

#Training process
def train(mlp,dataset,lr,n_outputs,epoches):
    error_set = [0 for n in range(epoches)]
    for n in range(epoches):#each epoch updating the network for each row in the training dataset
        error_s = 0#Error
        for single_data in dataset:
            outputs = forward_prop(mlp,single_data)#forward propagation first and get results for each single data
            results = [0 for x in range(n_outputs)]
            results[int(single_data[-1])] = 1
            for t in range(len(results)):
                error = (results[t] - outputs[t])**2
                error_s = error_s + error
            backward_propagate_error(mlp,results)#back propagate error computed 
            weights_update(mlp,single_data,lr)#update weights in each layer according to error and input in each layer
        error_set[n] = error_s
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (n, lr, error_s))
    return error_set

#Plotting training errors
def plot_training_errors(epoches,errors):
    x = epoches
    y = errors
    plt.plot(x,y,ls = '-',lw = 2,label = 'convergence line',color = 'purple')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('mean error')
    plt.title('Convergence graph with MLP(sigmod)')
    plt.show()

#Predicting process                
def predict(mlp,data):#Prediction according to updated weights in mlp
    result = forward_prop(mlp,data)
    max_prob_loc = result.index(max(result)) 
    return max_prob_loc


#Plotting confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None):

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        pass
        #print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_ylim(len(classes)-0.5, -0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.show()
    return ax


#Add evaluating metrics of trained model
def accuracy_metric(mlp,dataset):
    n = 0
    for single_data in dataset:
        label = single_data[-1]
        prediction = predict(mlp,single_data)
        if np.int(prediction) == np.int(label):
            n = n + 1
    return n/float(len(dataset))*100.0






