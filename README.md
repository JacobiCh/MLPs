# MLPs

# Data files:
 MNIST.zip needs to be unzipped and two data files named mnist_train.csv and mnist_test.csv are the MNIST database.
 wheat-seeds.csv data file is used in activation_comparison.py which is to compare and choose activation function experiment.

# Coding:
# tanh.py and mlp_sigmod.py are the classes constructing a MLP step by step and these two classes would be imported in activation_comparison.py.They also do not have main methods,would be runnable in activation_comparison.py.
# activation_comparison.py is the first experimenting class for comparing and choosing activation function before formal experiments.Thus,it can be dirctly runned to generate comparison results including plots and printings.
# final_mlp.py is the basic structured class(defination) of constructing MLP for training,no main method to run.Other experimenting classes need importing this class for constructing MLP.
# utils.py is the tool class including all the implemented loading,plotting,dividing methods,error generating and etc.This class would also be imported in other experimenting classes like final_mlp.py.

# Experiments:
# With basic constructor(defination) of MLP in final_mlp and tool class named utils.py.All experimenting classes import both of them.There are main methods in these classes so that they can be runned directly.
# cross_valiadation.py and cross_valiadation.py are classes for getting the results of cross validation(5-fold).The former is training MNIST without error term,the latter is with error term.
# single_digit_experiment.py is the experimenting class for training with partly MNIST(single digit) database.In this method,the random digit image would be plotted and the distribution of the single digit dataset would be plotted to analyze.Training results including confusion matrix,convergence graph,accuracy graph would be plotted too.Besides,the error can be generated and added to the single-digit dataset by controlling the variable named error_times(multiplier) and error_term. Having the results of this class,the research can go deep into whole MNIST dataset.
# Experiment 1 classes:
# experiment_without_error.py is the implemented class which is the comparison experiment controlling no input data error and model error in this experiment.The results includes confusion matrix,accuries with epochs,convergence graph...
# experiment_with_error is the implemented class which is adding different sizes of input errors in MNIST training dataset by controlling the variable named error_times. And the results would be saved by running different times with different error_times(need to change the name of saving file).Example:np.save('2_times_error',arr=np.array(training_accuracy_set)) needed to be changed to np.save('4_times_error',arr=np.array(training_accuracy_set)) when changing to larger error_times(from 2 to 4) 
# Experiment 2 classes:
# experiment_with_model_error.py is the implemented class which construct a MLP with much less nodes in hidden layer(to add model error).Except the number of nodes,the other paramters set are same as experiment_without_error.py.It needs training the MNIST dataset without input error following the controlling-variable principle.Results including training and testing accuracies would be also saved as model_error.npy.(need to run firstly)

# Plotting:
# After running all these experiments for times.There are saved results data for training and testing accuracies of different experiments,for example,'model_error.npy','2_times_error.npy'...These names are changable before runing the corresponding classes.plotting.py is the class that uesd after all these results data being saved and plotting by different comparison.   
