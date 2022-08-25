import numpy as np
from matplotlib import pyplot as plt
# Plotting training graph together 
def training_graph_combination():
    x = [i for i in range(20)]
    times_train2 = np.load('2_times_error.npy')
    times_train4 = np.load('4_times_error.npy')
    times_train8 = np.load('8_times_error.npy')
    times_train16 = np.load('16_times_error.npy')

    plt.plot(x,times_train2,ls = '-',lw = 2,label = 'error*2',color = 'purple')
    plt.plot(x,times_train4,ls = '-',lw = 2,label = 'error*4',color = 'blue')
    plt.plot(x,times_train8,ls = '-',lw = 2,label = 'error*8',color = 'yellow')
    plt.plot(x,times_train16,ls = '-',lw = 2,label = 'error*16',color = 'r')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('Training accuracy')
    plt.title('Training accuracy graph with different sizes of errors')
    plt.show()

# Plotting training graph together with model error
def training_graph_combination_with_model_e():
    x = [i for i in range(20)]
    times_train2 = np.load('2_times_error.npy')
    times_train4 = np.load('4_times_error.npy')
    times_train8 = np.load('8_times_error.npy')
    times_train16 = np.load('16_times_error.npy')
    model_error = np.load('model_error.npy')

    plt.plot(model_error,ls = '-',lw = 2,label = 'model error',color = 'black')
    plt.plot(x,times_train2,ls = '-',lw = 2,label = 'error*2',color = 'purple')
    plt.plot(x,times_train4,ls = '-',lw = 2,label = 'error*4',color = 'blue')
    plt.plot(x,times_train8,ls = '-',lw = 2,label = 'error*8',color = 'yellow')
    plt.plot(x,times_train16,ls = '-',lw = 2,label = 'error*16',color = 'r')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('Training accuracy')
    plt.title('Training accuracy graph with different sizes of errors')
    plt.show()

# Plotting testing graph together 
def testing_graph_combination():
    x = [i for i in range(20)]
    times_test2 = np.load('2_times_error_test.npy')
    times_test4 = np.load('4_times_error_test.npy')
    times_test8 = np.load('8_times_error_test.npy')
    times_test16 = np.load('16_times_error_test.npy')

    plt.plot(x,times_test2,ls = '-',lw = 2,label = 'error*2',color = 'purple')
    plt.plot(x,times_test4,ls = '-',lw = 2,label = 'error*4',color = 'blue')
    plt.plot(x,times_test8,ls = '-',lw = 2,label = 'error*8',color = 'yellow')
    plt.plot(x,times_test16,ls = '-',lw = 2,label = 'error*16',color = 'r')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('Testing accuracy')
    plt.title('Testing accuracy graph with different sizes of errors')
    plt.show()    

# Plotting testing graph together with model error
def testing_graph_combination_with_model_e():
    x = [i for i in range(20)]
    times_test2 = np.load('2_times_error_test.npy')
    times_test4 = np.load('4_times_error_test.npy')
    times_test8 = np.load('8_times_error_test.npy')
    times_test16 = np.load('16_times_error_test.npy')
    model_error = np.load('model_error_test.npy')

    plt.plot(model_error,ls = '-',lw = 2,label = 'model error',color = 'black')
    plt.plot(x,times_test2,ls = '-',lw = 2,label = 'error*2',color = 'purple')
    plt.plot(x,times_test4,ls = '-',lw = 2,label = 'error*4',color = 'blue')
    plt.plot(x,times_test8,ls = '-',lw = 2,label = 'error*8',color = 'yellow')
    plt.plot(x,times_test16,ls = '-',lw = 2,label = 'error*16',color = 'r')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('Testing accuracy')
    plt.title('Testing accuracy graph with different sizes of errors')
    plt.show()  

def main():
    # Main running methods
    training_graph_combination()
    training_graph_combination_with_model_e()
    testing_graph_combination()
    testing_graph_combination_with_model_e()

if __name__=='__main__':
    main()