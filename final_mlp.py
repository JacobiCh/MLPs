import numpy as np 
import scipy.special as ssp  # sigmod activation function included in ssp and can be used directly



# Multilayers perception class definition(NN)
# define a class refering MLP algorithm and named MLP_NN. 
# The functions included in this class like train or test can be used.
class MLP_NN:
    # initialise the neural network
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate): # parameters of NN class
        # set number of nodes in each input, hidden, output layer）
        self.inodes = input_nodes # input_layer
        self.hnodes = hidden_nodes # hidden_layer
        self.onodes = output_nodes # output_layer
        
        # link weight matrices, wih and who
        # Initialize the weights between input_layer and hidden_layer named wih,hidden_layer and output_layer named who
        # Randomly initialize using normal distribution and the mean value is 0 and deviation is -0.5 power of the node number of hidden layer   
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)) #矩阵大小为隐藏层节点数×输入层节点数
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes)) #矩阵大小为输出层节点数×隐藏层节点数
        
        # learning rate
        self.lr = learning_rate
        
        # activation function is the sigmoid function
        # define sigmoid as self.activation_function
        self.activation_function = lambda x: ssp.expit(x) 
        # lambda x:generate f(x) and named self.activation_function
        pass

    # train the model
    def train(self, inputs_list, targets_list):
        # convert inputs list and labels list to 2d array
        inputs = np.array(inputs_list, ndmin = 2).T # ndmin=2 represents 2d
        targets = np.array(targets_list, ndmin = 2).T 
        
        # Forward propagation
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signal into final output layer
        final_inputs = np.dot(self.who, hidden_outputs) 
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        # Finish FP
        
        # Backward propagation
        # output layer error is the (target - actual)
        # Calculate the errors between results and actual values
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors) # dot of weights and errors of output_layer 
        
        # update the weights for the weights(links) betwwen the hidden and output layers
        self.who = self.who + self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),np.transpose(hidden_outputs))
        # update the weights for the links between the input and hidden layers
        self.wih = self.wih + self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        pass
    
    def predict(self,input_list):
        # FW propagation process to predict(Need normalization)
        inputs = np.array(input_list, ndmin = 2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        predicted_label = np.argmax(final_outputs) # argmax()to find the maximum of labels among labels
        return predicted_label
    #Finish defining the class of MLP and next to train and test



