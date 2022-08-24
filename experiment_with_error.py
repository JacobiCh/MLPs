import final_mlp
import utils
import numpy as np
import matplotlib.pyplot as plt
#Define parameters of MLP
input_nodes = 784 # 784 inputs since pictures are all 28×28 pix
hidden_nodes = 200  # The larger number of hidden nodes is,the preciser the model is.
# But will remain the same when the number is increasing.
output_nodes = 10 

# learning rate
learning_rate = 0.1 

# create instance of multilayer perception
mlp = final_mlp.MLP_NN(input_nodes, hidden_nodes, output_nodes, learning_rate)    

# Load mnist training dataset
training_dataset_h = utils.load_dataset('mnist_train.csv')
training_dataset = training_dataset_h[:10000]

# load mnist test data and construct dataset for evaluation
test_dataset_h = utils.load_dataset("mnist_test.csv")
test_dataset =test_dataset_h[:5000]


#Plot densities of train and test sets
# utils.distribution_visualization('mnist_train.csv')
# utils.distribution_visualization("mnist_test.csv")

#define input error and multiplier
error_times = 16
error_term = utils.error_generation(training_dataset,test_dataset)*error_times
# utils.plot_distribution_with_error(training_dataset,error_term)

# Training process
epochs = 20 #Iteration number
epochs_list = [n for n in range(epochs)]
training_error_set = [0 for x in range(epochs)]
training_accuracy_set = [0 for x in range(epochs)]
testing_accuracy_set = [0 for x in range(epochs)]

predicted_results = []
true_results = []




for n in range(epochs):
    training_correct_num = 0
    testing_correct_num = 0
    error_s = 0 #sum of errors
    # train all records in the training data set
    for data in training_dataset:
        # split the record by the ',' commas
        values = data.split(',')
        # normalization and scale and shift the inputs
        inputs = (np.asfarray(values[:-1]) / 255.0 * 0.99) + 0.01 - error_term
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = np.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(values[-1])] = 0.99
        outputs = mlp.predict(inputs)
        outputs_vec = np.zeros(output_nodes) + 0.01
        outputs_vec[int(outputs)] = 0.99
        for t in range(len(targets)):
                error = (targets[t] - outputs_vec[t])**2
                error_s = error_s + error
        mlp.train(inputs, targets) 
        pass

    for data in training_dataset:
        # split the record by the ',' commas
        values = data.split(',')
        # label is the last value
        true_label = int(values[-1])
        # normalization
        inputs = (np.asfarray(values[:-1]) / 255.0 * 0.99) + 0.01 - error_term
        # predict
        prediction = mlp.predict(inputs)
        if (int(prediction) == true_label):
            training_correct_num += 1
        pass

    for data in test_dataset:
        # split each data into values
        values = data.split(',')
        # label is the last value
        true_label = int(values[-1])
        # normalization
        inputs = (np.asfarray(values[:-1]) / 255.0 * 0.99) + 0.01
        # predict
        prediction = mlp.predict(inputs)
        if (int(prediction) == true_label):
            testing_correct_num += 1
        pass

    testing_accuracy_set[n] = testing_correct_num/len(test_dataset)
    training_accuracy_set[n] = training_correct_num/len(training_dataset)
    print('epoch '+str(n)+':the train accuracy is '+str(training_accuracy_set[n])+',the test accuracy is '+str(testing_accuracy_set[n]))
    training_error_set[n] = error_s
    pass
#Accuracy and loss graphs when training for test and train sets.

# np.save('2_times_error',arr=np.array(training_accuracy_set))
# np.save('2_times_error_test',arr=np.array(testing_accuracy_set))
# utils.plot_training_accuracy(epochs_list,training_accuracy_set)
# utils.plot_testing_accuracy(epochs_list,testing_accuracy_set)
# utils.plot_training_errors(epochs_list,training_error_set)
# utils.plot_training_testing_accuracy(epochs_list,training_accuracy_set,testing_accuracy_set)

#Confusion matrix for trained model
for data in test_dataset:
    # split each data into values
    values = data.split(',')
    # label is the last value
    true_label = int(values[-1])
    true_results.append(true_label)
    # normalization
    inputs = (np.asfarray(values[:-1]) / 255.0 * 0.99) + 0.01
    # predict
    prediction = mlp.predict(inputs) 
    predicted_results.append(int(prediction))
    pass
labels = list(set(true_results))
class_names = np.array(labels)
utils.plot_confusion_matrix(true_results,predicted_results,classes=class_names, normalize=False)
plt.show()
print('The mean accuracy metric of the MLP is : ' + str(utils.accuracy_metric(true_results,predicted_results))+' %')

