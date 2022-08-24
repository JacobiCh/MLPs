import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
import seaborn as sns
from scipy.stats import norm

#Loading dataset
def load_dataset(filename):
    training_data_file = open(filename, 'r') 
    training_data_list = training_data_file.readlines()
    training_data_file.close()
    return training_data_list
#Divide dataset into n folds
def n_fold(list,n):
    lists = []
    for i in range(0,len(list),n):
        lists.append(list[i:i+n])
    return lists

#Plotting distribution
def distribution_visualization(filename):
    training_dataset = load_dataset(filename)
    dataset = []
    for data in training_dataset:
         print(data)
        # split the record by the ',' commas
         values = data.split(',')
        # normalization and scale and shift the inputs
         inputs = (np.asfarray(values[:-1]) / 255.0 * 0.99)
         dataset.extend(inputs)
    sns.set_palette("hls")
    sns.set_style("white")
    plt.figure(dpi=120)
    g = sns.distplot(dataset,
                 hist=True,
                 kde=True,  # kernel density estimate (KDE)
                 kde_kws={'linestyle': '--', 'linewidth': '1', 'color': 'purple',
                          },
                 fit=norm,
                 color='blue',
                 axlabel='attribute(pixel) values',  # x label
                 )
    plt.show()

#Plotting distribution
def distribution_visualization_sd(training_dataset):
    dataset = []
    for data in training_dataset:
         print(data)
        # split the record by the ',' commas
         values = data.split(',')
        # normalization and scale and shift the inputs
         inputs = (np.asfarray(values[:-1]) / 255.0 * 0.99)
         dataset.extend(inputs)
    sns.set_palette("hls")
    sns.set_style("white")
    plt.figure(dpi=120)
    g = sns.distplot(dataset,
                 hist=True,
                 kde=True,  # kernel density estimate (KDE)
                 kde_kws={'linestyle': '--', 'linewidth': '1', 'color': 'purple',
                          },
                 fit=norm,
                 color='blue',
                 axlabel='attribute(pixel) values',  # x label
                 )
    plt.show()
#Generating error according to distributions
def error_generation(train_set,test_set):
    train_first_value = train_set[0].split(',')
    train_first_input = (np.asfarray(train_first_value[:-1]) / 255.0 * 0.99)
    train_sum_data = train_first_input
    for n in range(1,len(train_set)):
        # split the record by the ',' commas
        values = train_set[n].split(',')
         # normalization and scale and shift the inputs
        inputs = (np.asfarray(values[:-1]) / 255.0 * 0.99) 
        train_sum_data += inputs
    train_mean_data = train_sum_data/len(train_set)
    
    test_first_value = test_set[0].split(',')
    test_first_input = (np.asfarray(test_first_value[:-1]) / 255.0 * 0.99)
    test_sum_data = test_first_input
    for b in range(1,len(test_set)):
        # split the record by the ',' commas
        values = test_set[b].split(',')
         # normalization and scale and shift the inputs
        inputs = (np.asfarray(values[:-1]) / 255.0 * 0.99) 
        test_sum_data += inputs
    test_mean_data = test_sum_data/len(test_set)
    error_data = train_mean_data - test_mean_data
    return error_data
#Plotting distribution of data set with error
def plot_distribution_with_error(dataset,error_term):
    new_data_set_plotting_set = []
    for t in range(len(dataset)):
        # split the record by the ',' commas
         values = dataset[t].split(',')
        # normalization and scale and shift the inputs
         inputs = (np.asfarray(values[:-1]) / 255.0 * 0.99) - error_term
         new_data_set_plotting_set.extend(inputs)
    sns.set_palette("hls")
    sns.set_style("white")
    plt.figure(dpi=120)
    g = sns.distplot(new_data_set_plotting_set,
                 hist=True,
                 kde=True,  # kernel density estimate (KDE)
                 kde_kws={'linestyle': '--', 'color': 'purple'},
                 fit=norm,
                 color='blue',
                 axlabel='attribute(pixel) values',  # x-aix label
                 )
    plt.title("Distribution with error")
    plt.show()    
    
#Plotting training accuracy graph          
def plot_training_accuracy(epoches,accuracy_set):
    x = epoches
    y = accuracy_set
    plt.plot(x,y,ls = '-',lw = 2,label = 'accuracy line',color = 'blue')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Training accuracy graph with epoch')
    plt.show()

#Plotting testing accuracy graph    
def plot_testing_accuracy(epoches,accuracy_set):
    x = epoches
    y = accuracy_set
    plt.plot(x,y,ls = '--',lw = 2,label = 'accuracy line',color = 'black')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Testing accuracy graph with epoch')
    plt.show()

#Plotting training accuracy graph with error
def plot_training_errors(epoches,errors):
    x = epoches
    y = errors
    plt.plot(x,y,ls = '-',lw = 2,label = 'convergence line',color = 'purple')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('mean error')
    plt.title('Convergence graph with MLP(sigmod)')
    plt.show()

#Plotting training and testing accuracy graph(together) 
def plot_training_testing_accuracy(epoches,training_accuracies,testing_accuracies):
    x = epoches
    y1 = training_accuracies
    y2 = testing_accuracies
    plt.plot(x,y1,ls = '-',lw = 2,label = 'train',color = 'blue')
    plt.plot(x,y2,ls = '--',lw = 2,label = 'test',color = 'black')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy graph during training process')
    plt.show()

#Add evaluating metrics of trained model
def accuracy_metric(true_labels,predictions):
    n = 0
    for t in range(len(true_labels)):
        if np.int(true_labels[t]) == np.int(predictions[t]):
            n = n + 1
    return n/np.float16(len(true_labels))*100.0

#Plotting confusion matrix for predictions
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
    # fig.show()
    return ax


