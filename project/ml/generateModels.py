from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.datasets import make_classification
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.callbacks import Callback, EarlyStopping
import tensorflow as tf
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

TRAINING_DATA = 'ml/data/zbior_treningowy.csv'

def prepare_Data(directory, b_size_1, epochs1, activation1, activation2, output_dims1, b_size2, epochs2, activation3, activation4, activation5, activation6, output_dims2, output_dims3, output_dims4):
    maindirectory = TRAINING_DATA
    datasetmain = pd.read_csv(maindirectory, sep=';')
    print(f'file: {directory}')
    datasettest =  pd.read_csv(directory, sep=';')

    dataset=datasetmain.drop(columns="NAZWISKO")
    datasettest=datasettest.drop(columns="NAZWISKO")

    #ustawienie X od kolumny 1 do 12
    X_train = dataset.iloc[:, 1:12].values
	#ustawienie Y od 12 do konca
    y_train = dataset.iloc[:, 12:].values
	
    X_test = datasettest.iloc[:, 1:12].values
    y_test = datasettest.iloc[:, 12:].values
	
	

	#from sklearn.model_selection import train_test_split
	#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testDataSize, random_state = 0)
	
	# standaryzacja zbioru

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    def auc_roc(y_true, y_pred):
        # any tensorflow metric
        value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

        # find all variables created for this metric
        metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

        # Add metric variables to GLOBAL_VARIABLES collection.
        # They will be initialized for new session.
        for v in metric_vars:
            tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

        # force to update metric values
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
            return value

    my_callbacks = [EarlyStopping(monitor='val_accuracy', patience=300, verbose=1, mode='max')]

    def create_first_model(X_train1, y_train1, X_test, y_test, b_size_1, epoch1, activation1, activation2, output_dims1):
        classifier1 = Sequential()
        classifier1.add(Dense( output_dims1, kernel_initializer = 'uniform', activation = activation1, input_dim = 11))
        classifier1.add(Dense( 3, kernel_initializer = 'uniform', activation = activation2))
        classifier1.compile(optimizer = "adam", loss = 'categorical_crossentropy', metrics = ['accuracy'])
        history =classifier1.fit(X_train1, y_train1, batch_size = b_size_1 ,epochs = epoch1, 
                                callbacks=my_callbacks, verbose=1, validation_split=0.1)
        history_test =classifier1.fit(X_test, y_test, batch_size = b_size_1, epochs = epoch1,
                                    callbacks=my_callbacks, verbose=1, validation_split=0.1)

        pred_prob = classifier1.predict(X_test)
        
        
        return classifier1, history, history_test, pred_prob

    def create_third_model(X_train, y_train, X_test, y_test, b_size_1, epoch1, activation1, activation2, activation3, activation4, output_dims1, output_dims2, output_dims3):
        classifier3 = Sequential()
        classifier3.add(Dense( output_dims1, kernel_initializer = 'uniform', activation = activation1, input_dim = 11))
        classifier3.add(Dense( output_dims2, kernel_initializer = 'uniform', activation = activation2))
        classifier3.add(Dense( output_dims3, kernel_initializer = 'uniform', activation = activation3))
        classifier3.add(Dense( 3, kernel_initializer = 'uniform', activation = activation4))
        classifier3.compile(optimizer = "adam", loss = 'categorical_crossentropy', metrics = ['accuracy'])
        history3 =classifier3.fit(X_train, y_train, batch_size = b_size_1, epochs = epoch1,
                                    callbacks=my_callbacks, validation_split=0.1, verbose=1)

        history3_test =classifier3.fit(X_test, y_test, batch_size = b_size_1, epochs = epoch1,
                                    callbacks=my_callbacks, validation_split=0.1, verbose=1)
                                    
        pred_prob = classifier1.predict(X_test)					 
                                    
        return classifier3, history3, history3_test, pred_prob


    def generate_plot_model_one(X_test, y_test, X_train, y_train, classifier1, history, history_test, X_prob):
        final1_test=classifier1.evaluate(X_test,y_test)
        final1=classifier1.evaluate(X_train,y_train)

        fpr1, tpr1, thresh1 = roc_curve(y_test.argmax(axis=1), X_prob[:,1], pos_label=1)
        random_probs = [0 for i in range(len(y_test))]
        p_fpr, p_tpr, _ = roc_curve(y_test.argmax(axis=1), random_probs, pos_label=1)

        plt.plot(history.history['accuracy'])
        plt.plot(history_test.history['accuracy'])
        plt.title('Dokładnosc predykcji dla każdej epoki ')
        plt.ylabel('Dokładnosc')
        plt.xlabel('Epoka')
        plt.legend(['Zbior treningowy', 'Zbior testowy'])
        plt.show()


        plt.plot(history.history['val_accuracy'])
        plt.plot(history_test.history['val_accuracy'])

        plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Model pierwszy')
        
        plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
        # title
        plt.title('ROC curve')
        # x label
        plt.xlabel('False Positive Rate')
        # y label
        plt.ylabel('True Positive rate')

        plt.legend(loc='best')
        plt.savefig('ROC',dpi=300)
        plt.show()
        



        
    def generate_plot_model_two(X_test, y_test, X_train, y_train, classifier2, history, history_test, X_prob):
        final1_test=classifier2.evaluate(X_test,y_test)
        final1=classifier2.evaluate(X_train,y_train)

        fpr1, tpr1, thresh1 = roc_curve(y_test.argmax(axis=1), X_prob[:,1], pos_label=1)
        random_probs = [0 for i in range(len(y_test))]
        p_fpr, p_tpr, _ = roc_curve(y_test.argmax(axis=1), random_probs, pos_label=1)


        plt.plot(history.history['accuracy'])
        plt.plot(history_test.history['accuracy'])
        plt.title('Dokładnosc predykcji dla każdej epoki ')
        plt.ylabel('Dokładnosc')
        plt.xlabel('Epoka')
        plt.legend(['Zbior treningowy', 'Zbior testowy'])
        plt.show()

        plt.plot(history.history['val_accuracy'])
        plt.plot(history_test.history['val_accuracy'])

        plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Model pierwszy')
        
        plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
        # title
        plt.title('ROC curve')
        # x label
        plt.xlabel('False Positive Rate')
        # y label
        plt.ylabel('True Positive rate')

        plt.legend(loc='best')
        plt.savefig('ROC',dpi=300)
        plt.show()



    def final_score_data_table(b_size_1, epochs1, b_size_2, epochs2, classifier1, classifier2, X_test1, X_train1, y_train1,  y_test1 ):
        cls1 = KerasClassifier(build_fn = classifier1, batch_size = b_size_1, epochs = epochs1)
        # accuracies1 = cross_val_score(estimator = cls1, X = X_train1, y = y_train1, cv = 5, n_jobs = 1)

        #Średnia precyzja dopasowania modelu
        # mean_1 = accuracies1.mean()
        
        #Odchylenie standardowe
        # variance_1 = accuracies1.std()
        from sklearn import metrics
        y_pred1 = classifier1.predict(X_test1)
        #Wartosc funkcji starty
        score1 = metrics.log_loss(y_test1, y_pred1)
        
        #dokladnosc dopasowania zbioru testowego
        final1_test=classifier1.evaluate(X_test1,y_test1)
        #treningowego
        final1=classifier1.evaluate(X_train1,y_train1)
        
        cls2 = KerasClassifier(build_fn = classifier2, batch_size = b_size_2, epochs = epochs2)
        #accuracies2 = cross_val_score(estimator = cls2, X = X_train1, y = y_train1, cv = 5, n_jobs = 1, 
        #                      verbose=1)
        #Średnia precyzja dopasowania modelu
        #mean_2 = accuracies2.mean()
        
        #Odchylenie standardowe
        #variance_2 = accuracies2.std()
        
        #dokladnosc dopasowania zbioru testowego
        final2_test=classifier2.evaluate(X_test,y_test)
        #treningowego
        final2=classifier2.evaluate(X_train,y_train)

        from sklearn import metrics
        y_pred2 = classifier2.predict(X_test)
        #Wartosc funkcji starty
        score2 = metrics.log_loss(y_test, y_pred2)

        return final1_test, final2_test, final1, final2, score1, score2

    classifier1, history, history_test, pred_prob= create_first_model(X_train, y_train, X_test, y_test, b_size_1, epochs1, activation1, activation2, output_dims1)
    generate_plot_model_one(X_test, y_test, X_train, y_train, classifier1, history, history_test, pred_prob)
    classifier2, history2, history_test2, pred_prob2=create_third_model(X_train, y_train, X_test, y_test, b_size2, epochs2, activation3, activation4, activation5, activation6, output_dims2, output_dims3, output_dims4)
    generate_plot_model_two(X_test, y_test, X_train, y_train, classifier2, history2, history_test2, pred_prob2)
    final1_test, final2_test, final1, final2, score1, score2 =final_score_data_table(b_size_1, b_size2, epochs1, epochs2, classifier1, classifier2, X_test, X_train, y_train,  y_test )    
    return final1_test, final2_test, final1, final2, score1, score2

# try:
#print(prepare_Data('/home/adam/projects/python/suml/pro/ml/data/zbior_testowy.csv', 8,5,'relu', 'relu', 3, 3,5,'relu', 'relu', 'elu', 'softmax', 14, 14, 13))
# except:
#     print('Error')


