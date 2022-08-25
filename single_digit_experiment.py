import final_mlp
import utils
import numpy as np
import random
from matplotlib import pyplot as plt

 
def main():
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
    training_dataset_whole = utils.load_dataset('mnist_train.csv')
    # training_dataset = training_dataset_h[:10000]

    # load mnist test data and construct dataset for evaluation
    test_dataset_whole = utils.load_dataset("mnist_test.csv")
    # test_dataset =test_dataset_h[:5000]

    training_dataset_sd = []
    testing_dataset_sd = []
    random_digit = random.randint(0,10)
    # random_digit = 6
    for n in range(len(training_dataset_whole)):
        values = training_dataset_whole[n].split(',')
        if int(values[-1]) == random_digit:
            training_dataset_sd.append(training_dataset_whole[n])
    for m in range(len(test_dataset_whole)):
        values = test_dataset_whole[m].split(',')
        if int(values[-1]) == random_digit:
            testing_dataset_sd.append(test_dataset_whole[m])


    #Plot densities of train and test sets
    utils.distribution_visualization_sd(training_dataset_sd)
    utils.distribution_visualization_sd(testing_dataset_sd)

    sum_values = [0 for n in range(input_nodes)]
    for a in range(len(training_dataset_sd)):
        values = training_dataset_sd[a].split(',')
        int_values_wit_label = [int(value) for value in values]
        int_values = int_values_wit_label[:-1]
        for b in range(input_nodes):
            sum_values[b] += int_values[b]
    mean_values = [(pixel/len(training_dataset_sd)) for pixel in sum_values]
    #Ploat single digit
    image_correct = np.asfarray(mean_values).reshape((28, 28))
    plt.imshow(image_correct, cmap = 'Greys', interpolation = 'None')
    plt.show()

    #Generate error for single digit database
    error_term = utils.error_generation(training_dataset_sd,testing_dataset_sd)

    train_first_value = training_dataset_sd[0].split(',')
    train_first_input = np.asfarray(train_first_value[:-1])
    train_sum_data = train_first_input
    for n in range(1,len(training_dataset_sd)):
        # split the record by the ',' commas
        values = training_dataset_sd[n].split(',')
        # normalization and scale and shift the inputs
        inputs = np.asfarray(values[:-1]) 
        train_sum_data += inputs

    train_mean_data = train_sum_data/len(training_dataset_sd)
    test_first_value = testing_dataset_sd[0].split(',')
    test_first_input = np.asfarray(test_first_value[:-1])
    test_sum_data = test_first_input
    for b in range(1,len(testing_dataset_sd)):
        # split the record by the ',' commas
        values = testing_dataset_sd[b].split(',')
        # normalization and scale and shift the inputs
        inputs = np.asfarray(values[:-1])
        test_sum_data += inputs
    test_mean_data = test_sum_data/len(testing_dataset_sd)
    #Define error times number
    error_times = 32
    error_term = (train_mean_data - test_mean_data)*error_times
    #Plotting data set
    sum_values_with_error = [0 for n in range(input_nodes)]
    for z in range(len(training_dataset_sd)):
        values = training_dataset_sd[z].split(',')
        int_values_with_error =  np.asfarray(values[:-1]) - error_term 
        for c in range(input_nodes):
            sum_values_with_error[c] += int_values_with_error[c]
    mean_values_with_error = [(pixel/len(training_dataset_sd)) for pixel in sum_values_with_error]
    #Plot image with error
    image_correct_with_error = np.asfarray(mean_values_with_error).reshape((28, 28))
    plt.imshow(image_correct_with_error, cmap = 'Greys', interpolation = 'None')
    plt.show()
    print(error_term)



if __name__=='__main__':
    main()