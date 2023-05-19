import matplotlib.pyplot as plt
import mlrose_hiive as mlrose
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from util import splitData, getFullFilePath
from sklearn.model_selection import learning_curve, cross_validate
from time import time
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd
import numpy as np
simplefilter("ignore", category=ConvergenceWarning)

PRESET_HIDDEN_LAYERS = [20, 20]
PRESET_MAX_ATTEMPTS = 200
PRESET_MAX_ITER = 5000
PRESET_MAX_ITER_GA = 500
PRESET_SOLVER = 'adam'
PRESENT_TRAIN_SIZE = 90
RANDOM_SEED = 100

class NeuralNetworks:

    numeric_fields = []
    title = ''

    def __init__(self):
        self.title = self.__class__.__name__

    def analyzeSimulatedAnnealing(self, data):

        accuracy_test = []
        accuracy_train = []
        f1_score_test = []
        precision_score_test = []
        recall_score_test = []
        curve_list = []

        X_train, X_test, y_train, y_test = splitData(data, PRESENT_TRAIN_SIZE)
        decay_list = [0.5, 0.7, 0.85, 0.99]

        fig = plt.figure()
        ax = fig.gca()
        i = 0

        for decay in decay_list:
            schedule = mlrose.GeomDecay(decay=decay)
            nn_model = mlrose.NeuralNetwork(hidden_nodes = PRESET_HIDDEN_LAYERS, activation = 'relu',
                        algorithm = 'simulated_annealing', max_iters = PRESET_MAX_ITER,
                        bias=True,is_classifier=True, learning_rate=0.01,early_stopping = True, clip_max = 5,
                        restarts=0, max_attempts = PRESET_MAX_ATTEMPTS, random_state = RANDOM_SEED, curve=False,
                        schedule=schedule)

            nn_model.fit(X_train, y_train)
            predict_test = nn_model.predict(X_test)
            accuracy_test.append(accuracy_score(y_test, predict_test))
            predict_train = nn_model.predict(X_train)
            accuracy_train.append(accuracy_score(y_train, predict_train))
            f1_score_test.append(f1_score(y_test, predict_test, average='weighted'))
            precision_score_test.append(precision_score(y_test, predict_test, average='weighted'))
            recall_score_test.append(recall_score(y_test, predict_test, average='weighted'))

        fig, ax = plt.subplots()
        plt.title('Neutral Network Weight Optimization - Simulated Annealing - Model Complexity Curve')
        plt.xlabel('Decay Rate')
        plt.ylabel("Accuracy")
        ax.plot(decay_list, accuracy_test, color="blue", label="Testing Accuracy")
        ax.plot(decay_list, accuracy_train, color="red", label="Training Accuracy")
        plt.legend()
        filename = 'NN-Compare-SA-Parameters-Curve.png'
        plt.savefig(getFullFilePath(filename), bbox_inches='tight')
        plt.close()

        df = pd.DataFrame(dict(graph=decay_list,
            accuracy_test=accuracy_test, f1_score_test=f1_score_test, precision_score_test=precision_score_test, recall_score_test=recall_score_test))

        ind = np.arange(len(df))
        width = 0.2
        ind = ind + 0.1
        fig, ax = plt.subplots()
        ax.barh(ind , df.accuracy_test, width, color='red', label='Accuracy')
        ax.barh(ind + width, df.f1_score_test, width, color='green', label='F1 Score')
        ax.barh(ind + 2*width, df.precision_score_test, width, color='blue', label='Precision Score')
        ax.barh(ind + 3*width, df.recall_score_test, width, color='yellow', label='Recall Score')

        ax.set(yticks=ind + width, yticklabels=df.graph, ylim=[2*width - 1, len(df)])
        ax.legend()
        plt.title('Neutral Network Weight Optimization - Simulated Annealing')
        plt.xlabel('Metrics Score')
        plt.ylabel("Decay Rate")
        filename = 'NN-Compare-SA-Parameters.png'
        plt.savefig(getFullFilePath(filename), bbox_inches='tight')
        plt.close()


    def analyzeGeneticAlgoirthm(self, data):

        accuracy_test = []
        accuracy_train = []
        f1_score_test = []
        precision_score_test = []
        recall_score_test = []
        curve_list = []
        X_train, X_test, y_train, y_test = splitData(data, PRESENT_TRAIN_SIZE)

        ####### pop_size
        pop_size_list  = [50, 100, 200, 300, 400]

        fig = plt.figure()
        ax = fig.gca()
        i = 0

        for pop_size in pop_size_list:

            nn_model = mlrose.NeuralNetwork(hidden_nodes = PRESET_HIDDEN_LAYERS, activation = 'relu',
                        algorithm = 'simulated_annealing', max_iters = PRESET_MAX_ITER,
                        bias=True,is_classifier=True, learning_rate=0.01,early_stopping = True, clip_max = 5,
                        restarts=0, max_attempts = PRESET_MAX_ATTEMPTS, random_state = RANDOM_SEED, curve=False,
                        pop_size=pop_size)

            nn_model.fit(X_train, y_train)
            predict_test = nn_model.predict(X_test)
            accuracy_test.append(accuracy_score(y_test, predict_test))
            predict_train = nn_model.predict(X_train)
            accuracy_train.append(accuracy_score(y_train, predict_train))
            f1_score_test.append(f1_score(y_test, predict_test, average='weighted'))
            precision_score_test.append(precision_score(y_test, predict_test, average='weighted'))
            recall_score_test.append(recall_score(y_test, predict_test, average='weighted'))

        fig, ax = plt.subplots()
        plt.title('Neutral Network Weight Optimization - Genetic Algorithm - PopSize Comparison')
        plt.xlabel('Population Size')
        plt.ylabel("Accuracy")
        ax.plot(pop_size_list, accuracy_test, color="blue", label="Testing Accuracy")
        ax.plot(pop_size_list, accuracy_train, color="red", label="Training Accuracy")
        plt.legend()
        filename = 'NN-Compare-GA-Parameters-PopSize-Curve.png'
        plt.savefig(getFullFilePath(filename), bbox_inches='tight')
        plt.close()

        df = pd.DataFrame(dict(graph=pop_size_list,
            accuracy_test=accuracy_test, f1_score_test=f1_score_test, precision_score_test=precision_score_test, recall_score_test=recall_score_test))

        ind = np.arange(len(df))
        width = 0.2
        ind = ind + 0.1
        fig, ax = plt.subplots()
        ax.barh(ind , df.accuracy_test, width, color='red', label='Accuracy')
        ax.barh(ind + width, df.f1_score_test, width, color='green', label='F1 Score')
        ax.barh(ind + 2*width, df.precision_score_test, width, color='blue', label='Precision Score')
        ax.barh(ind + 3*width, df.recall_score_test, width, color='yellow', label='Recall Score')

        ax.set(yticks=ind + width, yticklabels=df.graph, ylim=[2*width - 1, len(df)])
        ax.legend()
        plt.title('Neutral Network Weight Optimization - Genetic Algorithm - PopSize variance')
        plt.xlabel('Metrics Score')
        plt.ylabel("Population Size")
        filename = 'NN-Compare-GA-Parameters-PopSize.png'
        plt.savefig(getFullFilePath(filename), bbox_inches='tight')
        plt.close()

        mutation_prob_list = [0.01, 0.05, 0.1, 0.2, 0.3]

        fig = plt.figure()
        ax = fig.gca()
        i = 0


        accuracy_test = []
        accuracy_train = []
        f1_score_test = []
        precision_score_test = []
        recall_score_test = []
        curve_list = []
        for mutation_prob in mutation_prob_list:
            nn_model = mlrose.NeuralNetwork(hidden_nodes = PRESET_HIDDEN_LAYERS, activation = 'relu',
                        algorithm = 'simulated_annealing', max_iters = PRESET_MAX_ITER,
                        bias=True,is_classifier=True, learning_rate=0.01,early_stopping = True, clip_max = 5,
                        restarts=0, max_attempts = PRESET_MAX_ATTEMPTS, random_state = RANDOM_SEED, curve=False,
                        mutation_prob=mutation_prob)

            nn_model.fit(X_train, y_train)
            predict_test = nn_model.predict(X_test)
            accuracy_test.append(accuracy_score(y_test, predict_test))
            predict_train = nn_model.predict(X_train)
            accuracy_train.append(accuracy_score(y_train, predict_train))
            f1_score_test.append(f1_score(y_test, predict_test))
            precision_score_test.append(precision_score(y_test, predict_test))
            recall_score_test.append(recall_score(y_test, predict_test))

        fig, ax = plt.subplots()
        plt.title('Neutral Network Weight Optimization - Genetic Algorithm - Mutation Prob Curve')
        plt.xlabel('Population Size')
        plt.ylabel("Accuracy")
        ax.plot(pop_size_list, accuracy_test, color="blue", label="Testing Accuracy")
        ax.plot(pop_size_list, accuracy_train, color="red", label="Training Accuracy")
        plt.legend()
        filename = 'NN-Compare-GA-Parameters-MutationProb-Curve.png'
        plt.savefig(getFullFilePath(filename), bbox_inches='tight')
        plt.close()

        df = pd.DataFrame(dict(graph=mutation_prob_list,
            accuracy_test=accuracy_test, f1_score_test=f1_score_test, precision_score_test=precision_score_test, recall_score_test=recall_score_test))

        ind = np.arange(len(df))
        width = 0.2
        ind = ind + 0.1
        fig, ax = plt.subplots()
        ax.barh(ind , df.accuracy_test, width, color='red', label='Accuracy')
        ax.barh(ind + width, df.f1_score_test, width, color='green', label='F1 Score')
        ax.barh(ind + 2*width, df.precision_score_test, width, color='blue', label='Precision Score')
        ax.barh(ind + 3*width, df.recall_score_test, width, color='yellow', label='Recall Score')

        ax.set(yticks=ind + width, yticklabels=df.graph, ylim=[2*width - 1, len(df)])
        ax.legend()
        plt.title('Neutral Network Weight Optimization - Genetic Algorithm - Mutation Probability variance')
        plt.xlabel('Metrics Score')
        plt.ylabel("Mutation Probability")
        filename = 'NN-Compare-GA-Parameters-MutationProb.png'
        plt.savefig(getFullFilePath(filename), bbox_inches='tight')
        plt.close()


    def analyzeAlgorithms(self, data):

        accuracy_train = []
        accuracy_test = []

        f1_score_train = []
        f1_score_test = []

        precision_score_train = []
        precision_score_test = []

        recall_score_train = []
        recall_score_test = []

        time_train = []
        time_test = []

        curves = []

        seed_list = np.arange(10, 110, 10).tolist()

        X_train, X_test, y_train, y_test = splitData(data, PRESENT_TRAIN_SIZE)

        print('running MLPClassifier...')
        a1_list=[]; a2_list=[]; f1_list=[]; f2_list=[]; p1_list=[]; p2_list=[]; r1_list=[]; r2_list=[]; t1_list=[]; t2_list=[];
        for s in seed_list:
            mlp = MLPClassifier(activation='relu', momentum=0.9,
                hidden_layer_sizes=PRESET_HIDDEN_LAYERS, learning_rate='constant',
                max_iter=PRESET_MAX_ITER, solver=PRESET_SOLVER, random_state=s,
                early_stopping = True)
            time_before_training = time()
            mlp.fit(X_train, y_train)
            time_after_training = time()
            predict_test = mlp.predict(X_test)
            time_after_predict = time()
            predict_train = mlp.predict(X_train)

            a1_list.append(accuracy_score(y_train, predict_train))
            a2_list.append(accuracy_score(y_test, predict_test))
            f1_list.append(f1_score(y_train, predict_train, average='weighted'))
            f2_list.append(f1_score(y_test, predict_test, average='weighted'))
            p1_list.append(precision_score(y_train, predict_train, average='weighted'))
            p2_list.append(precision_score(y_test, predict_test, average='weighted'))
            r1_list.append(recall_score(y_train, predict_train, average='weighted'))
            r2_list.append(recall_score(y_test, predict_test, average='weighted'))
            t1_list.append(time_after_training - time_before_training)
            t2_list.append(time_after_predict - time_after_training)

        accuracy_train.append(np.mean(a1_list))
        accuracy_test.append(np.mean(a2_list))
        f1_score_train.append(np.mean(f1_list))
        f1_score_test.append(np.mean(f2_list))
        precision_score_train.append(np.mean(p1_list))
        precision_score_test.append(np.mean(p2_list))
        recall_score_train.append(np.mean(r1_list))
        recall_score_test.append(np.mean(r2_list))
        time_train.append(np.mean(t1_list))
        time_test.append(np.mean(t2_list))
        # print(confusion_matrix(y_test, predict_test))
        # print(classification_report(y_test, predict_test))


        print('running random_hill_climb...')
        a1_list=[]; a2_list=[]; f1_list=[]; f2_list=[]; p1_list=[]; p2_list=[]; r1_list=[]; r2_list=[]; t1_list=[]; t2_list=[]; _curves=[]
        for s in seed_list:
            nn_model = mlrose.NeuralNetwork(hidden_nodes = PRESET_HIDDEN_LAYERS, activation = 'relu', early_stopping=True,
                    algorithm = 'random_hill_climb', max_iters = PRESET_MAX_ITER,
                    bias=True, is_classifier=True, learning_rate=0.01,  clip_max = 5,
                    restarts=0, max_attempts = PRESET_MAX_ATTEMPTS, random_state = s, curve=True)
            time_before_training = time()
            nn_model.fit(X_train, y_train)
            time_after_training = time()
            predict_test = nn_model.predict(X_test)
            time_after_predict = time()
            predict_train = nn_model.predict(X_train)

            a1_list.append(accuracy_score(y_train, predict_train))
            a2_list.append(accuracy_score(y_test, predict_test))
            f1_list.append(f1_score(y_train, predict_train, average='weighted'))
            f2_list.append(f1_score(y_test, predict_test, average='weighted'))
            p1_list.append(precision_score(y_train, predict_train, average='weighted'))
            p2_list.append(precision_score(y_test, predict_test, average='weighted'))
            r1_list.append(recall_score(y_train, predict_train, average='weighted'))
            r2_list.append(recall_score(y_test, predict_test, average='weighted'))
            t1_list.append(time_after_training - time_before_training)
            t2_list.append(time_after_predict - time_after_training)
            _curves.append(nn_model.fitness_curve)

        accuracy_train.append(np.mean(a1_list))
        accuracy_test.append(np.mean(a2_list))
        f1_score_train.append(np.mean(f1_list))
        f1_score_test.append(np.mean(f2_list))
        precision_score_train.append(np.mean(p1_list))
        precision_score_test.append(np.mean(p2_list))
        recall_score_train.append(np.mean(r1_list))
        recall_score_test.append(np.mean(r2_list))
        time_train.append(np.mean(t1_list))
        time_test.append(np.mean(t2_list))
        curves.append(np.mean(_curves, axis=0))
        # print(confusion_matrix(y_test, predict_test))
        # print(classification_report(y_test, predict_test))

        print('running simulated_annealing...')
        a1_list=[]; a2_list=[]; f1_list=[]; f2_list=[]; p1_list=[]; p2_list=[]; r1_list=[]; r2_list=[]; t1_list=[]; t2_list=[]; _curves=[]
        for s in seed_list:
            nn_model = mlrose.NeuralNetwork(hidden_nodes = PRESET_HIDDEN_LAYERS, activation = 'relu', early_stopping=True,
                    algorithm = 'simulated_annealing', max_iters = PRESET_MAX_ITER,
                    bias=True,is_classifier=True, learning_rate=0.01, clip_max = 5,
                    restarts=0, max_attempts = PRESET_MAX_ATTEMPTS, random_state = s, curve=True)
            time_before_training = time()
            nn_model.fit(X_train, y_train)
            time_after_training = time()
            predict_test = nn_model.predict(X_test)
            time_after_predict = time()
            predict_train = nn_model.predict(X_train)

            a1_list.append(accuracy_score(y_train, predict_train))
            a2_list.append(accuracy_score(y_test, predict_test))
            f1_list.append(f1_score(y_train, predict_train, average='weighted'))
            f2_list.append(f1_score(y_test, predict_test, average='weighted'))
            p1_list.append(precision_score(y_train, predict_train, average='weighted'))
            p2_list.append(precision_score(y_test, predict_test, average='weighted'))
            r1_list.append(recall_score(y_train, predict_train, average='weighted'))
            r2_list.append(recall_score(y_test, predict_test, average='weighted'))
            t1_list.append(time_after_training - time_before_training)
            t2_list.append(time_after_predict - time_after_training)
            _curves.append(nn_model.fitness_curve)

        accuracy_train.append(np.mean(a1_list))
        accuracy_test.append(np.mean(a2_list))
        f1_score_train.append(np.mean(f1_list))
        f1_score_test.append(np.mean(f2_list))
        precision_score_train.append(np.mean(p1_list))
        precision_score_test.append(np.mean(p2_list))
        recall_score_train.append(np.mean(r1_list))
        recall_score_test.append(np.mean(r2_list))
        time_train.append(np.mean(t1_list))
        time_test.append(np.mean(t2_list))
        curves.append(np.mean(_curves, axis=0))
        # print(confusion_matrix(y_test, predict_test))
        # print(classification_report(y_test, predict_test))

        print('running genetic_alg...')
        a1_list=[]; a2_list=[]; f1_list=[]; f2_list=[]; p1_list=[]; p2_list=[]; r1_list=[]; r2_list=[]; t1_list=[]; t2_list=[]; _curves=[]
        for s in seed_list:
            nn_model = mlrose.NeuralNetwork(hidden_nodes = PRESET_HIDDEN_LAYERS, activation = 'relu', early_stopping=True,
                    algorithm = 'genetic_alg', max_iters = PRESET_MAX_ITER_GA,
                    bias=True,is_classifier=True, learning_rate=0.01, clip_max = 5,
                    restarts=0, max_attempts = PRESET_MAX_ATTEMPTS, random_state = s, curve=True)
            time_before_training = time()
            nn_model.fit(X_train, y_train)
            time_after_training = time()
            predict_test = nn_model.predict(X_test)
            time_after_predict = time()
            predict_train = nn_model.predict(X_train)

            a1_list.append(accuracy_score(y_train, predict_train))
            a2_list.append(accuracy_score(y_test, predict_test))
            f1_list.append(f1_score(y_train, predict_train, average='weighted'))
            f2_list.append(f1_score(y_test, predict_test, average='weighted'))
            p1_list.append(precision_score(y_train, predict_train, average='weighted'))
            p2_list.append(precision_score(y_test, predict_test, average='weighted'))
            r1_list.append(recall_score(y_train, predict_train, average='weighted'))
            r2_list.append(recall_score(y_test, predict_test, average='weighted'))
            t1_list.append(time_after_training - time_before_training)
            t2_list.append(time_after_predict - time_after_training)
            _curves.append(nn_model.fitness_curve)

        accuracy_train.append(np.mean(a1_list))
        accuracy_test.append(np.mean(a2_list))
        f1_score_train.append(np.mean(f1_list))
        f1_score_test.append(np.mean(f2_list))
        precision_score_train.append(np.mean(p1_list))
        precision_score_test.append(np.mean(p2_list))
        recall_score_train.append(np.mean(r1_list))
        recall_score_test.append(np.mean(r2_list))
        time_train.append(np.mean(t1_list))
        time_test.append(np.mean(t2_list))
        curves.append(np.mean(_curves, axis=0))
        # print(confusion_matrix(y_test, predict_test))
        # print(classification_report(y_test, predict_test))

        # accuracy comparison
        df = pd.DataFrame(dict(graph=['MLPClassifier', 'RHC', 'SA', 'GA'],
            accuracy_train=accuracy_train, accuracy_test=accuracy_test))

        ind = np.arange(len(df))
        width = 0.2
        ind = ind + 0.1
        fig, ax = plt.subplots()
        ax.barh(ind , df.accuracy_train, width, color='red', label='training accuracy')
        ax.barh(ind + width, df.accuracy_test, width, color='green', label='testing accuracy')

        ax.set(yticks=ind + width, yticklabels=df.graph, ylim=[2*width - 1, len(df)])
        ax.legend()
        plt.title('Neutral Network Weight Optimization - Accuracy of different algorithms')
        plt.xlabel('Accuracy')
        plt.ylabel("Algorithms")

        filename = 'NN-Compare-Accuracy.png'
        plt.savefig(getFullFilePath(filename), bbox_inches='tight')
        plt.close()

        # time comparison
        df = pd.DataFrame(dict(graph=['MLPClassifier', 'RHC', 'SA', 'GA'],
            time_train=time_train, time_test=time_test))

        ind = np.arange(len(df))
        width = 0.2
        ind = ind + 0.1
        fig, ax = plt.subplots()
        ax.barh(ind , df.time_train, width, color='red', label='training time')
        ax.barh(ind + width, df.time_test, width, color='green', label='query time')

        ax.set(yticks=ind + width, yticklabels=df.graph, ylim=[2*width - 1, len(df)])
        ax.legend()
        plt.xscale("log")
        plt.title('Neutral Network Weight Optimization - Runtime of different algorithms')
        plt.xlabel('Time (s)')
        plt.ylabel("Algorithms")
        filename = 'NN-Compare-RunTime.png'
        plt.savefig(getFullFilePath(filename), bbox_inches='tight')
        plt.close()

        # all scores
        algorithm_list = ['MLPClassifier','RHC', 'SA', 'GA']
        df = pd.DataFrame(dict(graph=algorithm_list,
            accuracy_test=accuracy_test, f1_score_test=f1_score_test, precision_score_test=precision_score_test, recall_score_test=recall_score_test))

        ind = np.arange(len(df))
        width = 0.2
        ind = ind + 0.1
        fig, ax = plt.subplots()
        ax.barh(ind , df.accuracy_test, width, color='red', label='Accuracy')
        ax.barh(ind + width, df.f1_score_test, width, color='green', label='F1 Score')
        ax.barh(ind + 2*width, df.precision_score_test, width, color='blue', label='Precision Score')
        ax.barh(ind + 3*width, df.recall_score_test, width, color='yellow', label='Recall Score')

        ax.set(yticks=ind + width, yticklabels=df.graph, ylim=[2*width - 1, len(df)])
        ax.legend()
        plt.title('Neutral Network Weight Optimization - Metrics Score Comparison')
        plt.xlabel('Metrics Score')
        plt.ylabel("Algorithms")
        filename = 'NN-Compare-MetricsScore.png'
        plt.savefig(getFullFilePath(filename), bbox_inches='tight')
        plt.close()

        # fitness curve
        algorithm_list = ['RHC', 'SA', 'GA']
        color_list = ['black', 'green', 'red']
        i = 0
        for curve in curves:
            plt.plot(curve, label=algorithm_list[i], alpha=0.5, color=color_list[i])
            i = i + 1
        plt.legend()
        plt.xlabel("Iterations")
        plt.ylabel("Fitness")
        plt.title("Neural Network Weight Optimization - Fitness Comparison")
        filename = 'NN-Compare-Fitness.png'
        plt.savefig(getFullFilePath(filename), bbox_inches='tight')
        plt.close()


    def analyzeLearningCurve(self, data):

        X_train, X_test, y_train, y_test = splitData(data, PRESENT_TRAIN_SIZE)

        print('running MLPClassifier learning curve...')
        mlp = MLPClassifier(activation='relu', momentum=0.9,
            hidden_layer_sizes=PRESET_HIDDEN_LAYERS, learning_rate='constant',
            max_iter=PRESET_MAX_ITER, solver=PRESET_SOLVER, random_state=RANDOM_SEED,
            early_stopping = True)
        train_sizes, train_scores, test_scores = learning_curve(mlp, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 10))
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        plt.plot(train_sizes, train_mean, label="MLP Training Learning Curve", color='red', alpha=0.7)
        plt.plot(train_sizes, test_mean, label="MLP Testing Learning Curve", color='blue', alpha=0.7)
        plt.legend()
        plt.title('Neutral Network Weight Optimization - Learning Curves of MLPClassifier')
        plt.xlabel('Training Size')
        plt.ylabel("Score")
        filename = 'NN-LearningCurves-MLPClassifier.png'
        plt.savefig(getFullFilePath(filename), bbox_inches='tight')
        plt.close()

        print('running random_hill_climb learning curve...')
        nn_model = mlrose.NeuralNetwork(hidden_nodes = PRESET_HIDDEN_LAYERS, activation = 'relu', early_stopping=True,
            algorithm = 'random_hill_climb', max_iters = PRESET_MAX_ITER,
            bias=True, is_classifier=True, learning_rate=0.01,  clip_max = 5,
            restarts=0, max_attempts = PRESET_MAX_ATTEMPTS, random_state = RANDOM_SEED, curve=True)
        train_sizes, train_scores, test_scores = learning_curve(nn_model, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 10))
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        plt.plot(train_sizes, train_mean, label="RHC Training Learning Curve", color='red', alpha=0.7)
        plt.plot(train_sizes, test_mean, label="RHC Testing Learning Curve", color='blue', alpha=0.7)
        plt.legend()
        plt.title('Neutral Network Weight Optimization - Learning Curves of random_hill_climb')
        plt.xlabel('Training Size')
        plt.ylabel("Score")
        filename = 'NN-LearningCurves-rhc.png'
        plt.savefig(getFullFilePath(filename), bbox_inches='tight')
        plt.close()

        print('running simulated_annealing learning curve...')
        nn_model = mlrose.NeuralNetwork(hidden_nodes = PRESET_HIDDEN_LAYERS, activation = 'relu', early_stopping=True,
            algorithm = 'simulated_annealing', max_iters = PRESET_MAX_ITER,
            bias=True, is_classifier=True, learning_rate=0.01,  clip_max = 5,
            restarts=0, max_attempts = PRESET_MAX_ATTEMPTS, random_state = RANDOM_SEED, curve=True)
        train_sizes, train_scores, test_scores = learning_curve(nn_model, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 10))
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        plt.plot(train_sizes, train_mean, label="SA Training Learning Curve", color='red', alpha=0.7)
        plt.plot(train_sizes, test_mean, label="SA Testing Learning Curve", color='blue', alpha=0.7)
        plt.legend()
        plt.title('Neutral Network Weight Optimization - Learning Curves of simulated_annealing')
        plt.xlabel('Training Size')
        plt.ylabel("Score")
        filename = 'NN-LearningCurves-sa.png'
        plt.savefig(getFullFilePath(filename), bbox_inches='tight')
        plt.close()

        print('running genetic_alg learning curve...')
        nn_model = mlrose.NeuralNetwork(hidden_nodes = PRESET_HIDDEN_LAYERS, activation = 'relu', early_stopping=True,
            algorithm = 'genetic_alg', max_iters = PRESET_MAX_ITER_GA,
            bias=True, is_classifier=True, learning_rate=0.01,  clip_max = 5,
            restarts=0, max_attempts = PRESET_MAX_ATTEMPTS, random_state = RANDOM_SEED, curve=True)
        train_sizes, train_scores, test_scores = learning_curve(nn_model, X_train, y_train, cv=5, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 10))
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        plt.plot(train_sizes, train_mean, label="GA Training Learning Curve", color='red', alpha=0.7)
        plt.plot(train_sizes, test_mean, label="GA Testing Learning Curve", color='blue', alpha=0.7)
        plt.legend()
        plt.title('Neutral Network Weight Optimization - Learning Curves of genetic_alg')
        plt.xlabel('Training Size')
        plt.ylabel("Score")
        filename = 'NN-LearningCurves-ga.png'
        plt.savefig(getFullFilePath(filename), bbox_inches='tight')
        plt.close()


    def analyze(self, data):

        self.analyzeAlgorithms(data)
        #self.analyzeLearningCurve(data)
        #self.analyzeSimulatedAnnealing(data)
        #self.analyzeGeneticAlgoirthm(data)

