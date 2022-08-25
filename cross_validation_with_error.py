import final_mlp
import utils
import numpy as np
def main():
    #Define parameters of MLP
    input_nodes = 784 # 784 inputs since pictures are all 28×28 pix
    hidden_nodes = 200  # The larger number of hidden nodes is,the preciser the model is.
    # But will remain the same when the number is increasing.
    output_nodes = 10 

    # learning rate
    learning_rate = 0.1 

    # create instance of multilayer perception
    mlp_0 = final_mlp.MLP_NN(input_nodes, hidden_nodes, output_nodes, learning_rate)    
    mlp_1 = final_mlp.MLP_NN(input_nodes, hidden_nodes, output_nodes, learning_rate)
    mlp_2 = final_mlp.MLP_NN(input_nodes, hidden_nodes, output_nodes, learning_rate)
    mlp_3 = final_mlp.MLP_NN(input_nodes, hidden_nodes, output_nodes, learning_rate)
    mlp_4 = final_mlp.MLP_NN(input_nodes, hidden_nodes, output_nodes, learning_rate)
    mlp_chosen = final_mlp.MLP_NN(input_nodes, hidden_nodes, output_nodes, learning_rate)
    mlps = [mlp_0,mlp_1,mlp_2,mlp_3,mlp_4]

    # Load mnist training dataset
    training_dataset = utils.load_dataset('mnist_train.csv')


    # load mnist test data and construct dataset for evaluation
    test_dataset = utils.load_dataset("mnist_test.csv")

    #define input error 
    error_times = 32
    error_term = utils.error_generation(training_dataset,test_dataset)**4

    # Training process
    epochs = 20 #Iteration number
    epochs_list = [n for n in range(epochs)]
    training_error_set = [0 for x in range(epochs)]
    training_accuracy_set = [0 for x in range(epochs)]
    training_true_results = []
    training_predicted_results = []

    testing_accuracy_set = [0 for x in range(epochs)]
    testing_predicted_results = []
    testing_true_results = []

    predicted_results = []
    true_results = []

    fold_num = 5
    n_folds = utils.n_fold(training_dataset,fold_num)
    indexs = [n for n in range(fold_num)]

    for n in range(epochs):
        sum_training_accuracies = 0
        error_s = 0
        cross_val_results = []
        for b in indexs:
            validation_set = n_folds[b]      
            training_set = []
            single_fold_results = []
            for a in range(fold_num):
                if a != b:
                    training_set.extend(n_folds[a])
            for data in training_set:
                # split the record by the ',' commas
                values = data.split(',')
                # normalization and scale and shift the inputs
                inputs = (np.asfarray(values[:-1]) / 255.0 * 0.99) + 0.01 - error_term
                # create the target output values (all 0.01, except the desired label which is 0.99)
                targets = np.zeros(output_nodes) + 0.01
                # all_values[0] is the target label for this record
                targets[int(values[-1])] = 0.99
                outputs = mlps[b].predict(inputs)
                outputs_vec = np.zeros(output_nodes) + 0.01
                outputs_vec[int(outputs)] = 0.99
                for t in range(len(targets)):
                        error = (targets[t] - outputs_vec[t])**2
                        error_s = error_s + error
                mlps[b].train(inputs, targets) 
                pass
            n_correct = 0
            for data in validation_set:
                # split the record by the ',' commas
                values = data.split(',')
                # label is the last value
                true_label = int(values[-1])
                # normalization
                inputs = (np.asfarray(values[:-1]) / 255.0 * 0.99) + 0.01 - error_term
                # predict
                prediction = mlps[b].predict(inputs)
                if int(prediction) == true_label:
                    n_correct += 1
                pass
            single_fold_accuracy = n_correct/len(validation_set)
            sum_training_accuracies += single_fold_accuracy
            cross_val_results.append(single_fold_accuracy)
        max_value = max(cross_val_results)
        cross_val_largest_index = cross_val_results.index(max_value)
        training_mean_accuracy = sum_training_accuracies/fold_num
        print('epoch '+str(n)+':the mean accuracy(with error) of 5-fold cross validation accuracies is '+str(training_mean_accuracy)+',the 5 folds accuracies are:')
        print(cross_val_results)

if __name__=='__main__':
    main()