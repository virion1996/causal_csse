import os
os.system("pip install lingam")
os.system("pip install fsspec")
os.system("pip install s3fs")

import random as rnd
from scipy.spatial import distance
import pandas as pd
from sklearn import preprocessing
import warnings
import sys
import argparse
import ast
import time
import json

import boto3

#German
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import train_test_split

import pickle

import numpy as np
import pandas as pd
import graphviz
import lingam
from lingam.utils import print_causal_directions, print_dagc, make_dot

import random as rnd

# from IPython.display import display

import warnings

warnings.filterwarnings('ignore')

s3 = boto3.client('s3')

#Used for ordering evaluations
class individual:
    def __init__(self, index, score, distance, num_changes, aval_norm, dist_norm, predict_proba):
        self.index = index #Indicates the instance's position in the dataframe
        self.score = score #Indicates the score in relation to the proximity of the class boundary
        self.distance = distance #Indicates the distance from the original instance
        self.num_changes = num_changes #Indicates the number of changes for class change
        self.aval_norm = aval_norm #Indicates the final fitness with standardized metrics
        self.dist_norm = dist_norm #Indicates the normalized distance (distance and number of changes)
        self.predict_proba = predict_proba #Indicates de individual's class
    def __repr__(self):
        return repr((self.index, self.score, self.distance, self.num_changes, self.aval_norm, self.dist_norm, self.predict_proba))

class counter_change:
    def __init__(self, column, value):
        self.column = column 
        self.value = value
    def __eq__(self, other):
        if self.column == other.column and self.value == other.value:
            return True
        else:
            return False
    def __repr__(self):
        return repr((self.column, self.value))    

#Used to generate a random value in the mutation operation
class feature_range:
    def __init__(self, column, col_type, min_value, max_value):
        self.column = column 
        self.col_type = col_type
        self.min_value = min_value
        self.max_value = max_value

    #Returns a random value to perform mutation operation
    def get_random_value(self):
        if self.col_type == 'int64' or self.col_type == 'int' or self.col_type == 'int16' or self.col_type == 'int8' or (self.col_type == 'uint8'):
            value = rnd.randint(self.min_value, self.max_value)
        else:  
            value = round(rnd.uniform(self.min_value, self.max_value), 2)
        return value
    
    #Checks if the attribute has only one value.
    def unique_value(self):
        if self.min_value != self.max_value:
            return False
        else:  
            return True    

    def __repr__(self):
        return repr((self.column, self.col_type, self.min_value, self.max_value)) 
        
class CSSE(object):
    
    def __init__(self, input_dataset, model, static_list = [], K = 3, num_gen = 30, pop_size = 100, per_elit = 0.1, cros_proba = 0.8, mutation_proba = 0.1, L1 = 1, L2 = 1):
        #User Options
        self.static_list = static_list #List of static features
        self.K = K #Number of counterfactuals desired
        #Model
        self.input_dataset = input_dataset
        self.model = model
        #GA Parameters
        self.num_gen = num_gen
        self.pop_size = pop_size
        self.per_elit = per_elit
        self.cros_proba = cros_proba
        self.mutation_proba = mutation_proba
        #Objective function parameters
        self.L1 = L1 #weight assigned the distance to the original instance
        self.L2 = L2 #weight assigned the number of changes needed in the original instance   
    
    #Get which index in the SHAP corresponding to the current class
    def getBadClass(self):   
        if self.current_class == self.model.classes_[0]:
            ind_cur_class = 0
        else:
            ind_cur_class = 1
        
        return ind_cur_class
    
    #Gets the valid values range for each feature
    def getFeaturesRange(self):
        features_range = []
       
        for i in range (0, self.input_dataset.columns.size):
            col_name = self.input_dataset.columns[i]
            col_type = self.input_dataset[col_name].dtype
            min_value = min(self.input_dataset[col_name])
            max_value = max(self.input_dataset[col_name])
            
            feature_range_ind = feature_range(col_name, col_type, min_value, max_value)
            features_range.append(feature_range_ind)
        
        return features_range
       
    def getMutationValue(self, currentValue, index, ind_feature_range):
        new_value = ind_feature_range.get_random_value()
        
        while currentValue == new_value:
            new_value = ind_feature_range.get_random_value()
        
        return new_value
    
    def equal(self, individual, population):
        aux = 0
        for i in range ( 1, len(population)):
            c = population.loc[i].copy()
            dst = distance.euclidean(individual, c)
            if dst == 0:
                aux = 1
        
        return aux

    def getPopInicial (self, df, features_range): 
        #The reference individual will always be in the 0 position of the df - so that it is normalized as well (it will be used later in the distance function)
        df.loc[0] = self.original_ind.copy()
        
        #Counting numbers of repeated individuals
        number_repetitions = 0
        
        #One more position is used because the zero position was reserved for the reference individual
        while len(df) < self.pop_size + 1:
            #Draw a feature to change
            index_a = rnd.randint( 0, self.input_dataset.columns.size - 1 )
            while df.columns[index_a] in self.static_list:
                index_a = rnd.randint( 0, self.input_dataset.columns.size - 1 )
                
            if not features_range[index_a].unique_value():
                #Mutation
                mutant = self.original_ind.copy()

                new_value =  self.getMutationValue(mutant.iloc[index_a], index_a, features_range[index_a])
                mutant.iloc[index_a] = new_value

                ni = self.equal(mutant, df)
                if ni == 0:
                    df.loc[len(df)] = mutant.copy()
                else:
                    #Assesses whether the GA is producing too many repeated individuals.
                    number_repetitions = number_repetitions + 1
                    if number_repetitions == 2*self.pop_size:
                        self.pop_size = round(self.pop_size - self.pop_size*0.1)
                        self.mutation_proba = self.mutation_proba + 0.1
                        #print('Adjusting population size...', self.pop_size)
                        number_repetitions = 0
    
    #Complete the standardized proximity and similarity assessments for each individual
    def getNormalEvaluation(self, evaluation, aval_norma):
        scaler2 = preprocessing.MinMaxScaler()
        aval_norma2 = scaler2.fit_transform(aval_norma)
    
        i = 0
        while i < len(evaluation):
            evaluation[i].aval_norm = self.L1*aval_norma2[i,0] + self.L2*aval_norma2[i,1] + aval_norma2[i,2]
            evaluation[i].dist_norm = self.L1*aval_norma2[i,0] + self.L2*aval_norma2[i,1]
        
            i = i + 1
    
    def numChanges(self, ind_con):
        num = 0
        for i in range(len(self.original_ind)):
            if self.original_ind[i] != ind_con[i]:
                num = num + 1
        
        return num
        
    def fitness(self, population, evaluation, ind_cur_class):
        def getProximityEvaluation (proba):
            #Penalizes the individual who is in the negative class
            if proba < 0.5:
                predict_score = 0
            else:
                predict_score= proba
             
            return predict_score
               
        #Calculates similarity to the original instance
        def getEvaluationDist (ind, X_train_minmax):
            #Normalizes the data so that the different scales do not bias the distance
            a = X_train_minmax[0]
            b = X_train_minmax[ind]
            dst = distance.euclidean(a, b)
  
            return dst
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
        
            predict_proba = self.model.predict_proba(population)
                    
        #Calculating the distance between instances
        scaler = preprocessing.MinMaxScaler()
        X_train_minmax = scaler.fit_transform(population)
    
        i = 0
        aval_norma = [] 
        while i < len(population):
            proximityEvaluation = getProximityEvaluation(predict_proba[i, ind_cur_class])
            evaldist = getEvaluationDist(i, X_train_minmax)
            #The original individual is in the 1st position
            numChanges = self.numChanges(population.loc[i])
        
            ind = individual(i, proximityEvaluation, evaldist, numChanges, 0, 0, predict_proba[i, ind_cur_class])
            aval_norma.append([evaldist, numChanges, proximityEvaluation])
            evaluation.append(ind)
            i = i + 1

        self.getNormalEvaluation(evaluation, aval_norma)
       
    #Given a counterfactual solution returns the list of modified columns
    def getColumns(self, counter_solution):
        colums = []
        for j in range (0, len(counter_solution)):
            colums.append(counter_solution[j].column)
        
        return colums      
             
    #Checks if the new solution is contained in the solutions already found
    def contained_solution(self, original_instance, current_list, current_column_list, new_solution, new_column_solution):
        contained = False
        for i in range (0, len(current_list)):              
            if set(current_column_list[i]).issubset(new_column_solution):
                for j in range (0, len(current_list[i])):
                    pos = new_column_solution.index(current_list[i][j].column)
                    distancia_a = abs(original_instance[current_list[i][j].column] - current_list[i][j].value)
                    distancia_b = abs(original_instance[current_list[i][j].column] - new_solution[pos].value)
                    if distancia_b >= distancia_a:
                        contained = True

        return contained
      
    def elitism(self, evaluation, df, parents):
         
        num_elit = round(self.per_elit*self.pop_size)
        
        aval = []
        aval = evaluation.copy()
        aval.sort(key=lambda individual: individual.aval_norm)
        
        #contrafactual_ind = pd.DataFrame(columns=self.input_dataset.columns)
        solution_list = []
        solution_colums_list = []
        
        i = 0
        numContraf = 0
        while i < len(aval) and numContraf <= num_elit + 1:
            #Checks if the example belongs to the counterfactual class
            if aval[i].predict_proba < 0.5:
                ind_changes = []
                ind_colums_change = []
         
                #Gets counterfactual example change list
                ind_changes = self.getChanges(aval[i].index, parents)
                #Generates the list of columns modified in the counterfactual to check if there is already a solution with that set of columns
                ind_colums_change = self.getColumns(ind_changes)
                
                if ind_colums_change not in solution_colums_list:
                    #Check if one solution is a subset of the other
                    if not self.contained_solution(self.original_ind, solution_list, solution_colums_list, ind_changes, ind_colums_change):
                        #Include counterfactual in the list of examples of the final solution                    
                        df.loc[len(df)] = parents.iloc[aval[i].index].copy()                     
                                
                        #Add to the list of solutions (changes only)       
                        solution_list.append(ind_changes)
                        #Used to compare with the next counterfactuals (to ensure diversity)
                        solution_colums_list.append(ind_colums_change)
                                        
                        numContraf = numContraf + 1
                      
            i = i + 1
        return solution_list
    
    def roulette_wheel(self, evaluation):
        summation = 0
        #Performs roulette wheel to select parents who will undergo genetic operations
        for i in range (1, len(evaluation)): 
            summation = summation + 1/evaluation[i].aval_norm
    
        roulette = rnd.uniform( 0, summation )
    
        roulette_score = 1/evaluation[1].aval_norm
        i = 1
        while roulette_score < roulette:
            i += 1
            roulette_score += 1/evaluation[i].aval_norm
        
        return i
            
    def crossover (self, df, parents, evaluation, number_cross_repetitions):
        child = []
            
        corte = rnd.randint( 0, self.input_dataset.columns.size - 1 )
            
        index1 = self.roulette_wheel(evaluation)
        index2 = self.roulette_wheel(evaluation)
        
        ind_a = parents.iloc[index1].copy()
        ind_b = parents.iloc[index2].copy()
            
        crossover_op = rnd.random()
        if crossover_op <= self.cros_proba:
            child[ :corte ] = ind_a[ :corte ].copy()
            child[ corte: ] = ind_b[ corte: ].copy()
        else:
            child = ind_a.copy()
        
        ni = self.equal(child, df)
        if ni == 0:
            df.loc[len(df)] = child.copy()
        else:
            #Assesses whether the GA is producing too many repeated individuals.
            number_cross_repetitions = number_cross_repetitions + 1
            if number_cross_repetitions == self.pop_size:
                self.pop_size = round(self.pop_size - self.pop_size*0.1)
                self.mutation_proba = self.mutation_proba + 0.1
                #print('Adjusting population size...', self.pop_size)
                number_cross_repetitions = 0
        #    print('repeated')
        return number_cross_repetitions
                       
    def mutation (self, df, individual_pos, features_range):
        ni = 1
        #Does not allow repeated individual
        while ni == 1:
            #Draw a feature to change
            index_a = rnd.randint( 0, self.input_dataset.columns.size - 1 )
            while df.columns[index_a] in self.static_list:
                index_a = rnd.randint( 0, self.input_dataset.columns.size - 1 )
            
            if not features_range[index_a].unique_value():
                #Mutation
                mutant = df.iloc[individual_pos].copy()
            
                #Draw the value to be changed
                new_value =  self.getMutationValue(mutant.iloc[index_a], index_a, features_range[index_a])  
                mutant.iloc[index_a] = new_value

                ni = self.equal(mutant, df)
                if ni == 0:
                    df.loc[individual_pos] = mutant.copy()
                #else:
                #    print('repeated')
     
    def getChanges(self, ind, dfComp):
        changes = []
        
        for i in range (len(dfComp.iloc[ind])):
            if self.original_ind[i] != dfComp.loc[ind][i]:
                counter_change_ind = counter_change(dfComp.columns[i], dfComp.loc[ind][i])
                changes.append(counter_change_ind)

        return changes
    
    #Generates the solution from the final population
    def getContrafactual(self, df, aval):
        
        contrafactual_ind = pd.DataFrame(columns=self.input_dataset.columns)
        solution_list = []
        solution_colums_list = []
        
        i = 0
        numContraf = 0
        while i < len(aval) and numContraf < self.K:
            #Checks if the example belongs to the counterfactual class
            if aval[i].predict_proba < 0.5:
                ind_changes = []
                ind_colums_change = []
         
                #Gets counterfactual example change list
                ind_changes = self.getChanges(aval[i].index, df)
                #Generates the list of columns modified in the counterfactual to check if there is already a solution with that set of columns
                ind_colums_change = self.getColumns(ind_changes)
                
                if ind_colums_change not in solution_colums_list:
                    #Check if one solution is a subset of the other
                    if not self.contained_solution(self.original_ind, solution_list, solution_colums_list, ind_changes, ind_colums_change):
                        #Include counterfactual in the list of examples of the final solution
                        contrafactual_ind.loc[len(contrafactual_ind)] = df.iloc[aval[i].index].copy()
                                
                        #Add to the list of solutions (changes only)       
                        solution_list.append(ind_changes)
                        #Used to compare with the next counterfactuals (to ensure diversity)
                        solution_colums_list.append(ind_colums_change)
                                        
                        numContraf = numContraf + 1
                        #print('solution_list ', solution_list)
                    #else:
                        #print('is contained ', ind_changes)
                #else:
                    #print('repeated ', ind_changes)
                      
            i = i + 1

        return contrafactual_ind, solution_list   
    
    def printResults(self, solution):
        print("Result obtained")
        if len(solution) != 0:
            for i in range(0, len(solution)): 
                print("\n")
                print(f"{'Counterfactual ' + str(i + 1):^34}")
                for j in range(0, len(solution[i])): 
                    print(f"{str(solution[i][j].column):<29} {str(solution[i][j].value):>5}")
        else:
            print('Solution not found. It may be necessary to adjust the parameters for this instance.')
                                                 
    def explain(self, original_ind, current_class):
        self.original_ind = original_ind #Original instance
        #self.ind_cur_class = ind_cur_class #Index in the shap corresponds to the original instance class
        self.current_class = current_class #Original instance class
        
        ind_cur_class = self.getBadClass()
    
        #Gets the valid values range of each feature
        features_range = []
        features_range = self.getFeaturesRange()

        #The DataFrame df will have the current population
        df = pd.DataFrame(columns=self.input_dataset.columns)
        
        #Generates the initial population with popinitial mutants        
        self.getPopInicial(df, features_range)
        for g in range(self.num_gen):
            #To use on the parents of each generation
            parents = pd.DataFrame(columns=self.input_dataset.columns)
    
            #Copy parents to the next generation
            parents = df.copy()
            #df will contain the new population
            df = pd.DataFrame(columns=self.input_dataset.columns)
            
            evaluation = []                         
                   
            #Assessing generation counterfactuals
            self.fitness(parents, evaluation, ind_cur_class)
            #The original individual will always be in the 0 position of the df - So that it is normalized too (it will be used later in the distance function)
            df.loc[0] = self.original_ind.copy()
            
            #Copies to the next generation the per_elit best individuals
            self.elitism(evaluation, df, parents)
            number_cross_repetitions = 0
            while len(df) < self.pop_size + 1: #+1, as the 1st position is used to store the reference individual
                number_cross_repetitions = self.crossover(df, parents, evaluation, number_cross_repetitions)
                
                mutation_op = rnd.random()
                if mutation_op <= self.mutation_proba:
                    self.mutation(df, len(df) - 1, features_range)
            
            print()
                 
        evaluation = []
    
        #Evaluating the latest generation
        self.fitness(df, evaluation, ind_cur_class)
    
        #Order the last generation by distance to the original instance     
        evaluation.sort(key=lambda individual: individual.aval_norm)     
        
        #Getting the counterfactual set
        contrafactual_set = pd.DataFrame(columns=self.input_dataset.columns)
        contrafactual_set, solution_list = self.getContrafactual(df, evaluation)       
                 
        return contrafactual_set, solution_list
    
    
    




class CCSSE:
    def __init__(self, dataset, bb_model, samples = None, K = 5, generation = 10, dataset_size = None):
        self.df_datasets = self.load_df_dataset()
        self.dataset = dataset
        self.samples = samples
        self.K = K
        self.generation = generation
        self.dataset_size = dataset_size
        
#         self.x_train, self.x_test, self.y_train, self.y_test, self.dfx_full, self.dfy_full = self.get_datasets_train_test()
        self.x_train, self.x_test, self.y_train, self.y_test, self.dfx_full, self.dfy_full = self.get_dataset()

        self.bb_model, self.p = self.get_bb_model(bb_model)
        self.explainerCSSE = self.get_model_contrafactual()
        self.model_causal, self.df_causal_effects, self.df_error, self.causal_order = self.get_model_causality()
    
        self.run_dict = {}
        self.run_non_causal_dict = {}
        
    def load_df_dataset(self):
        def convert_to_list(val):
            try:
                return ast.literal_eval(val) if isinstance(val, str) and val.startswith('[') and val.endswith(']') else val
            except (ValueError, SyntaxError):
                return val
            
        df = pd.read_parquet("s3://omar-testes-gerais/artigos/artifacts/dfm_use.parquet")
        df['path'] = df['path'].apply(convert_to_list)
        
        return df


    def get_dataset(self):
        dataset_dict = self.df_datasets[self.df_datasets['name'] == self.dataset].iloc[0].to_dict()
        
        if isinstance(dataset_dict['path'], list):
            if 'Column' in dataset_dict['classe']:
                df_train = pd.read_csv(f"s3://omar-testes-gerais/artigos/artifacts/datasets/{dataset_dict['path'][0]}", header = None, nrows=self.dataset_size)
                df_train.columns = df_train.columns.astype(str)
                df_test = pd.read_csv(f"s3://omar-testes-gerais/artigos/artifacts/datasets/{dataset_dict['path'][1]}", header = None, nrows=self.dataset_size)
                df_test.columns = df_test.columns.astype(str)
                class_name = str(int(dataset_dict['class'].split('Column')[1]) - 1)
            else:
                df_train = pd.read_csv(f"s3://omar-testes-gerais/artigos/artifacts/datasets/{dataset_dict['path'][0]}", nrows=self.dataset_size)
                df_test = pd.read_csv(f"s3://omar-testes-gerais/artigos/artifacts/datasets/{dataset_dict['path'][1]}", nrows=self.dataset_size)
                class_name = dataset_dict['classe']
            
            x_train = df_train.drop(columns=[class_name])
            y_train = df_train[class_name]

            # Dividindo o df_test
            x_test = df_test.drop(columns=[class_name])
            y_test = df_test[class_name]
            
            dfx_full = pd.concat([x_train, x_test])
            dfy_full = pd.concat([y_train, y_test])
            
        
        else:
            if 'Column' in dataset_dict['classe']:
                df_main = pd.read_csv(f"s3://omar-testes-gerais/artigos/artifacts/datasets/{dataset_dict['path']}", header = None, nrows=self.dataset_size)
                df_main.columns = df_main.columns.astype(str)
                class_name = str(int(dataset_dict['classe'].split('Column')[1]) - 1)
            else:
                df_main = pd.read_csv(f"s3://omar-testes-gerais/artigos/artifacts/datasets/{dataset_dict['path']}", nrows=self.dataset_size)
                class_name = dataset_dict['classe']
            
            columns = df_main.columns
            columns_tmp = list(columns)
            columns_tmp.remove(class_name)

            x_train, x_test, y_train, y_test = train_test_split(df_main[columns_tmp], df_main[class_name], test_size=0.1)

            dfx_full = pd.concat([x_train, x_test])
            dfy_full = pd.concat([y_train, y_test])
            
        return x_train, x_test, y_train, y_test, dfx_full, dfy_full
    
    def get_datasets_train_test(self):
        if self.dataset == "german_short":
            map_columns = {
                'Unnamed: 0': 'index',
                'x0': 'Sex',
                'x1': 'Age',
                'x2': 'Credit',
                'x3': 'LoanDuration',
            }
            x_train = pd.read_csv("data/algrec_german/X_train_short.csv").rename(columns = map_columns)
            x_test = pd.read_csv("data/algrec_german/X_test_short.csv").rename(columns = map_columns)
            y_train = pd.read_csv("data/algrec_german/y_train_short.csv").rename(columns={'Unnamed: 0': 'index'})
            y_test = pd.read_csv("data/algrec_german/y_test_short.csv").rename(columns={'Unnamed: 0': 'index'})
            x_train = x_train.set_index('index')
            x_test = x_test.set_index('index')
            y_train = y_train.set_index('index')
            y_test = y_test.set_index('index')
            dfx_full = pd.concat([x_train, x_test])
            dfy_full = pd.concat([y_train, y_test])

        elif self.dataset == "german_medium":
            map_columns = {
                'Unnamed: 0': 'index',
                'x0': 'Sex',
                'x1': 'Age',
                'x2': 'Credit',
                'x3': 'LoanDuration',
                'x4': 'CheckingAccountBalance',
                'x5':'SavingsAccountBalance',
                'x6':'HousingStatus'
            }
            x_train = pd.read_csv("data/algrec_german/X_train.csv").rename(columns = map_columns)
            x_test = pd.read_csv("data/algrec_german/X_test.csv").rename(columns = map_columns)
            y_train = pd.read_csv("data/algrec_german/y_train.csv").rename(columns={'Unnamed: 0': 'index'})
            y_test = pd.read_csv("data/algrec_german/y_test.csv").rename(columns={'Unnamed: 0': 'index'})
            x_train = x_train.set_index('index')
            x_test = x_test.set_index('index')
            y_train = y_train.set_index('index')
            y_test = y_test.set_index('index')
            dfx_full = pd.concat([x_train, x_test])
            dfy_full = pd.concat([y_train, y_test])

        elif self.dataset == "german_full":
            df_main = prepare_german_dataset("german_credit.csv", "data/")
            columns = df_main.columns
            class_name = 'default' # default = 0 = "Good class" / default = 1 = "Bad class" 
            columns_tmp = list(columns)
            columns_tmp.remove(class_name)

            x_train, x_test, y_train, y_test = train_test_split(df_main[columns_tmp], df_main[class_name], test_size=0.1)

            dfx_full = pd.concat([x_train, x_test])
            dfy_full = pd.concat([y_train, y_test])
            
        elif self.dataset == 'np':
            df_main = pd.read_csv("data/breast_coimbra_np.csv")
            columns = df_main.columns
            class_name = 'Classification' # default = 0 = "Good class" / default = 1 = "Bad class" 
            columns_tmp = list(columns)
            columns_tmp.remove(class_name)

            x_train, x_test, y_train, y_test = train_test_split(df_main[columns_tmp], df_main[class_name], test_size=0.1)

            dfx_full = pd.concat([x_train, x_test])
            dfy_full = pd.concat([y_train, y_test])
            
        elif self.dataset == 'nm':
            df_main = pd.read_csv("data/heloc_dataset_v1_nm.csv")
            columns = df_main.columns
            class_name = 'RiskPerformance' # default = 0 = "Good class" / default = 1 = "Bad class" 
            columns_tmp = list(columns)
            columns_tmp.remove(class_name)

            x_train, x_test, y_train, y_test = train_test_split(df_main[columns_tmp], df_main[class_name], test_size=0.1)

            dfx_full = pd.concat([x_train, x_test])
            dfy_full = pd.concat([y_train, y_test])
            
        elif self.dataset == 'cm':
            df_main = pd.read_csv("data/house_votes_84_processada_cm.csv")
            columns = df_main.columns
            class_name = 'Class Name' # default = 0 = "Good class" / default = 1 = "Bad class" 
            columns_tmp = list(columns)
            columns_tmp.remove(class_name)

            x_train, x_test, y_train, y_test = train_test_split(df_main[columns_tmp], df_main[class_name], test_size=0.1)

            dfx_full = pd.concat([x_train, x_test])
            dfy_full = pd.concat([y_train, y_test])
            
        elif self.dataset == 'ng':
            df_main = pd.read_csv("data/ionosphere_ng.csv")
            columns = df_main.columns
            class_name = 'target' # default = 0 = "Good class" / default = 1 = "Bad class" 
            columns_tmp = list(columns)
            columns_tmp.remove(class_name)

            x_train, x_test, y_train, y_test = train_test_split(df_main[columns_tmp], df_main[class_name], test_size=0.1)

            dfx_full = pd.concat([x_train, x_test])
            dfy_full = pd.concat([y_train, y_test])
            
        elif self.dataset == 'tokyo_ng':
            df_main = pd.read_csv("data/Tokyo_ng.csv").rename(columns = {'Unnamed: 44': 'class'})
            columns = df_main.columns
            class_name = 'class' # default = 0 = "Good class" / default = 1 = "Bad class" 
            columns_tmp = list(columns)
            columns_tmp.remove(class_name)

            x_train, x_test, y_train, y_test = train_test_split(df_main[columns_tmp], df_main[class_name], test_size=0.1)

            dfx_full = pd.concat([x_train, x_test])
            dfy_full = pd.concat([y_train, y_test])
            
        elif self.dataset == 'breast-cancer_ng':
            df_main = pd.read_csv("data/breast-cancer_ng.csv")
            columns = df_main.columns
            class_name = 'diagnosis' # default = 0 = "Good class" / default = 1 = "Bad class" 
            columns_tmp = list(columns)
            columns_tmp.remove(class_name)

            x_train, x_test, y_train, y_test = train_test_split(df_main[columns_tmp], df_main[class_name], test_size=0.1)

            dfx_full = pd.concat([x_train, x_test])
            dfy_full = pd.concat([y_train, y_test])
        
        elif self.dataset == 'cp':
            df_train = pd.read_csv("data/monks-1_train_cp.csv")

            #Get the input features
            columns = df_train.columns
            class_name = 'Class' # default = 0 = "Good class" / default = 1 = "Bad class" 
            columns_tmp = list(columns)
            columns_tmp.remove(class_name)
            columns_tmp.remove('Id')

            x_train = df_train[columns_tmp]
            y_train = df_train[['Class']]

            # x_train, x_test, y_train, y_test = train_test_split(df_main[columns_tmp], df_main[class_name], test_size=0.1)

            model = RandomForestClassifier()  
            model.fit(x_train, y_train)

            df_test = pd.read_csv('data/monks-1_test_cp.csv')

            #Get the input features
            columns = df_test.columns
            class_name = 'Class' # default = 0 = "Good class" / default = 1 = "Bad class" 
            columns_tmp = list(columns)
            columns_tmp.remove(class_name)
            columns_tmp.remove('Id')

            x_test = df_test[columns_tmp]
            y_test = df_test[['Class']]

            dfx_full = pd.concat([x_train, x_test])
            dfy_full = pd.concat([y_train, y_test])
            
        elif self.dataset == 'cg':
            df_main = pd.read_csv("data/mushrooms_processada_cg.csv")[:1000]
            columns = df_main.columns
            class_name = 'class' # default = 0 = "Good class" / default = 1 = "Bad class" 
            columns_tmp = list(columns)
            columns_tmp.remove(class_name)

            x_train, x_test, y_train, y_test = train_test_split(df_main[columns_tmp], df_main[class_name], test_size=0.1)

            dfx_full = pd.concat([x_train, x_test])
            dfy_full = pd.concat([y_train, y_test])
            
        elif self.dataset == 'Phishing_cg':
            df_main = pd.read_csv("data/Phishing_cg.csv")[2:1000].astype(float)
            columns = df_main.columns
            class_name = 'Class' # default = 0 = "Good class" / default = 1 = "Bad class" 
            columns_tmp = list(columns)
            columns_tmp.remove(class_name)

            x_train, x_test, y_train, y_test = train_test_split(df_main[columns_tmp], df_main[class_name], test_size=0.1)

            dfx_full = pd.concat([x_train, x_test])
            dfy_full = pd.concat([y_train, y_test])

        else:
            x_train = pd.DataFrame()
            x_test = pd.DataFrame()
            y_train = pd.DataFrame()
            y_test = pd.DataFrame()
            dfx_full = pd.DataFrame()
            dfy_full = pd.DataFrame()

        return x_train, x_test, y_train, y_test, dfx_full, dfy_full
    

    def get_bb_model(self, bb_model_name):
        
        if bb_model_name == 'rf':
            bb_model = RandomForestClassifier()  
            bb_model.fit(self.x_train, self.y_train)

            p = bb_model.predict(self.x_test)

            print(classification_report(self.y_test, p))

            return bb_model, p
        elif bb_model_name == 'rn':
            bb_model = MLPClassifier()  
            bb_model.fit(self.x_train, self.y_train)

            p = bb_model.predict(self.x_test)

            print(classification_report(self.y_test, p))

            return bb_model, p

    def get_model_contrafactual(self):
        return CSSE(self.dfx_full, self.bb_model, K = self.K, num_gen = self.generation)

    def get_model_causality(self):
        model_causal = lingam.DirectLiNGAM()
        model_causal.fit(self.dfx_full)
            
        labels = [f'{i}' for i in self.dfx_full.columns]
        causal_order = [labels[x] for x in model_causal.causal_order_]
        
        matrix = model_causal.adjacency_matrix_
        from_list = []
        to_list = []
        effect_list = []

        # Iteração sobre a matriz para extrair os valores e suas posições
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] != 0:
                    from_list.append(j)
                    to_list.append(i)
                    effect_list.append(matrix[i][j])

        # Criando o DataFrame
        df_causal_effects = pd.DataFrame({'from': from_list, 'to': to_list, 'effect': effect_list})
        labels = [f'{i}' for i in self.dfx_full.columns]
        df_causal_effects['from'] = df_causal_effects['from'].apply(lambda x : labels[x])
        df_causal_effects['to'] = df_causal_effects['to'].apply(lambda x : labels[x])


        matrix_error = model_causal.get_error_independence_p_values(self.dfx_full)
        from_list = []
        to_list = []
        effect_list = []

        # Iteração sobre a matriz para extrair os valores e suas posições
        for i in range(len(matrix_error)):
            for j in range(i + 1, len(matrix_error[i])):
                if matrix_error[i][j] != 0:
                    from_list.append(j)
                    to_list.append(i)
                    effect_list.append(matrix_error[i][j])

        # Criando o DataFrame
        df_error = pd.DataFrame({'from': from_list, 'to': to_list, 'effect': effect_list})
        labels = [f'{i}' for i in self.dfx_full.columns]
        df_error['from'] = df_error['from'].apply(lambda x : labels[x])
        df_error['to'] = df_error['to'].apply(lambda x : labels[x])
        df_error.fillna(0, inplace = True)

        return model_causal, df_causal_effects, df_error, causal_order
        
    
    def print_causal_graph(self):
        make_dot(self.model_causal.adjacency_matrix_)

    def run_non_causal(self):
        self.run_non_causal_dict = {}

        if isinstance(self.samples, list):
            self.create_run_dict(self)
            for sample in self.samples:
                self.run_non_causal_sample(sample)
                
        elif isinstance(self.samples, int):
            for sample in range(self.samples):
                self.run_non_causal_sample(sample)
        
        else: 
            for sample in range(10):
                self.run_non_causal_sample(sample)
                
    def run_non_causal_sample(self, sample):
        self.run_non_causal_dict[sample] = {}
        original_instance = self.x_test.iloc[sample].copy()
        contrafactual_set, solution = self.explainerCSSE.explain(original_instance, self.p[sample]) #Method returns the list of counterfactuals and the explanations generated from them

        self.run_non_causal_dict[sample]['solution'] = solution

    def run_causal(self):
        start_time = time.time()
        self.run_dict = {}
        self.run_dict['global_numbers'] = {
                    "global_quant_changes": 0,
                    "global_quant_causal_changes": 0,
                    "global_quant_causal_rules": 0,
                    "global_quant_zeros_causal": 0,
                    "global_quant_full_causal": 0,
                    "global_quant_causal_contrafac": 0,
                    "global_quant_maioria_causal_satisfeita": 0,
                    "global_quant_contrafac_unico": 0,
            }
        self.global_quant_contrafac_max = 0
        if isinstance(self.samples, list):
            for sample in self.samples:
                self.run_causal_sample(sample)
                
        elif isinstance(self.samples, int):
            for sample in range(self.samples):
                try:
                    self.run_causal_sample(sample)
                except Exception as e:
                    print(f"DEBUG ERRO: {e}")
        
        else: 
            for sample in range(10):
                self.run_causal_sample(sample)
        
        self.global_quant_contrafac_max = self.K * len(self.run_dict)
        self.run_dict['global_numbers']['global_timing_run_causal'] = time.time() - start_time


    def run_causal_sample(self, sample):
        if isinstance(self.samples, list):
            original_instance = self.dfx_full.iloc[sample]
        else:
            original_instance = self.x_test.iloc[sample]
        self.run_dict[sample] = {}
        self.run_dict[sample]['original_instance'] = original_instance

#         print(f'Running original instance:\n {display(original_instance)}')
        print(f'Start to Running samples')

        causal_explain = self.get_causal_explain(sample)
        self.run_dict[sample]['causal_explain'] = causal_explain

        list_analyse = []
        for contrafactual in causal_explain[0]:
            list_analyse.append(self.analyse_contrafac(contrafactual, causal_explain[1], causal_explain[2]))

        self.run_dict[sample]['list_analyse'] = list_analyse
        self.analyse_explaination(sample)

    def analyse_contrafac(self, contrafac, df, original_ind):
        columns = [x.column for x in contrafac]
        condicao = (df['to'].isin(columns)) & (df['from'].isin(columns))
        ind = original_ind[columns]
        return [contrafac, df[condicao], ind]

    def get_causal_explain(self, sample):
        if isinstance(self.samples, list):
            original_ind = self.dfx_full.iloc[sample].copy()
        else:
            original_ind = self.x_test.iloc[sample].copy() #Original instance
        #self.ind_cur_class = ind_cur_class #Index in the shap corresponds to the original instance class
        self.explainerCSSE.current_class = self.p[sample] #Original instance class
        self.explainerCSSE.original_ind = original_ind
        
        ind_cur_class = self.explainerCSSE.getBadClass()

        #Gets the valid values range of each feature
        features_range = []
        features_range = self.explainerCSSE.getFeaturesRange()

        #The DataFrame df will have the current population
        df = pd.DataFrame(columns=self.explainerCSSE.input_dataset.columns)

        #Generates the initial population with popinitial mutants        
        self.explainerCSSE.getPopInicial(df, features_range)
        df_causal = df.copy()
        dict_dfs = {}

        # for g in tqdm(range(self.explainerCSSE.num_gen), desc= "Processing..."):
        for g in range(self.generation):

            #To use on the parents of each generation
            old_parents = pd.DataFrame(columns=self.explainerCSSE.input_dataset.columns)

            #Copy parents to the next generation
            old_parents = df_causal.copy()
            dict_dfs[g] = {}

            parents_causal = self.apply_causality(old_parents)
            dict_dfs[g]['causal_parents'] = parents_causal
            #df will contain the new population
            df_causal = pd.DataFrame(columns=self.explainerCSSE.input_dataset.columns)
            evaluation_causal = []

            #Assessing generation counterfactuals
            self.explainerCSSE.fitness(dict_dfs[g]['causal_parents'], evaluation_causal, ind_cur_class)

            #The original individual will always be in the 0 position of the df - So that it is normalized too (it will be used later in the distance function)
            df_causal.loc[0] = original_ind.copy()

            #Copies to the next generation the per_elit best individuals
            self.explainerCSSE.elitism(evaluation_causal, df_causal, parents_causal)
            number_cross_repetitions = 0
            while len(df_causal) < self.explainerCSSE.pop_size + 1: #+1, as the 1st position is used to store the reference individual
                number_cross_repetitions_causal = self.explainerCSSE.crossover(df_causal, parents_causal, evaluation_causal, number_cross_repetitions)

                mutation_op = rnd.random()
                if mutation_op <= self.explainerCSSE.mutation_proba:
                    self.explainerCSSE.mutation(df_causal, len(df_causal) - 1, features_range)


        evaluation = []
        evaluation_causal = []

        #Evaluating the latest generation
        self.explainerCSSE.fitness(df_causal, evaluation_causal, ind_cur_class)

        #Order the last generation by distance to the original instance     
        evaluation_causal.sort(key=lambda individual: individual.aval_norm) 

        #Getting the counterfactual CAUSAL set
        contrafactual_set_causal, solution_list_causal = self.explainerCSSE.getContrafactual(df_causal, evaluation_causal) 

        dict_dfs['contrafactual_set_causal'] = contrafactual_set_causal
        dict_dfs['solution_list_causal'] = solution_list_causal
        
        df_contrafac_causal = self.get_contrafac_df_causal(solution_list_causal)
        return [solution_list_causal, df_contrafac_causal, original_ind]
    

    def apply_causality(self, df):
        df_apply_causal = pd.DataFrame(columns = df.columns)
        original = df.iloc[0]
        df_apply_causal.loc[0] = original
        for index, df_row in df.iloc[1:].iterrows():
            causal_ind = df_row.copy()
            for column in self.causal_order:
                value_diff = causal_ind[column] - original[column]
                if value_diff != 0:
                    tmp_effects = self.df_causal_effects[self.df_causal_effects['from'] == column]
                    for index, row in tmp_effects.iterrows():
    #                     prob = rnd.random()
    #                     if row['probability'] <= prob:
                        tmp_error = self.df_error[self.df_error['from'].isin([column, row['to']]) | self.df_error['to'].isin([column, row['to']])]
                        error_value = tmp_error['effect'].iloc[0]
    #                     print(f'error value = {error_value}')
                        causal_ind[row['to']] = causal_ind[row['to']] + (value_diff * row['effect']) + tmp_error['effect'].iloc[0]
            df_apply_causal.loc[len(df_apply_causal)] = causal_ind
        return df_apply_causal


    def get_contrafac_df_causal(self, solution_list_causal):
        lista_solution_causal = [[t.column for t in sublist] for sublist in solution_list_causal]

        # Inicializa uma lista para armazenar os resultados
        resultados = []

        # Loop sobre os valores na lista
        for lista_valores in lista_solution_causal:
            if len(lista_valores) > 1:
                for v1 in lista_valores:
                    for v2 in lista_valores:
                        if v1 != v2:
                            # Cria uma condição para cada par de valores diferentes na lista
                            condicao = (self.df_causal_effects['to'].isin([v1, v2])) & (self.df_causal_effects['from'].isin([v1, v2]))
                            # Realiza a busca no DataFrame usando a condição e armazena os resultados
                            resultados.append(self.df_causal_effects[condicao])

        # Concatena os resultados em um único DataFrame
        if resultados:
            resultado_final = pd.concat(resultados)
            resultado_final = resultado_final.drop_duplicates()
        else:
            resultado_final = pd.DataFrame(columns = self.df_causal_effects.columns)
            
        return resultado_final
    

    def analyse_explaination(self, sample):
        self.run_dict[sample]['data_analysis'] = []
        for i, content in enumerate(self.run_dict[sample]['list_analyse']):
            self.global_quant_contrafac_max += 1
            controle = {}
            causal = content[0]
            df = content[1]
            ori = content[2]
            
            
            num_changes = len(causal)
            self.run_dict['global_numbers']['global_quant_changes'] += num_changes
            
            num_causal_rules = len(df)
            self.run_dict['global_numbers']['global_quant_causal_rules'] += num_causal_rules
            
            for attr in causal:
                key = attr.column
                if attr.value > ori[key]:
                    controle[key] = 'mais'
                else:
                    controle[key] = 'menos'

            df_temp = df.copy()
            df_temp['from'] = df['from'].map(controle)
            df_temp['to'] = df['to'].map(controle)
            if len(df_temp) > 0:
                df_temp['causal'] = df_temp.apply(lambda row: self.verificar_condicoes(row), axis = 1)
                causal_finds = df_temp['causal'].sum()
            else:
                causal_finds = 0
                
            data_dict = {}

            data_dict['df_respeita_causal'] = df_temp
            data_dict['contrafactual_causal'] = causal
            data_dict['df_causal_effects'] = df
            
            self.run_dict[sample]['data_analysis'].append(data_dict)

            self.run_dict['global_numbers']['global_quant_causal_changes'] += causal_finds
            
            # print(f'causal = \n{causal}\n')
            # print(f'original = \n{ori}\n')
            # print(f'df_temp = \n{display(df_temp)}\n')
            
            if len(df_temp) > 0:
                if causal_finds > 0:
                    self.run_dict['global_numbers']['global_quant_causal_contrafac'] += 1
                else:
                    # print(f'nenhuma relaçao causal satisfeita')
                    self.run_dict['global_numbers']['global_quant_zeros_causal'] += 1
    #                 display(df_temp)
    #                 print(f"original = {ori}")
    #                 print(f"causal = {causal}")

                if causal_finds == num_causal_rules:
                    self.run_dict['global_numbers']['global_quant_full_causal'] += 1
                    # if causal_finds > 2:
                        # print(f'todas > 2 relaçoes causais satisfeitas')
    #                     display(df_temp)
    #                     print(f"original = {ori}")
    #                     print(f"causal = {causal}")
                    # elif causal_finds == 1:
                        # print(f'todas = 1 relaçoes causais satisfeitas')
                
                if causal_finds >= (len(df_temp)/2):
                    self.run_dict['global_numbers']['global_quant_maioria_causal_satisfeita'] += 1
            else:
    #             if len(causal) > 0:
                self.run_dict['global_numbers']['global_quant_contrafac_unico'] += 1
        
    def verificar_condicoes(self, row):
        if (row['from'] == 'mais' and row['to'] == 'mais' and row['effect'] > 0):
            return True
        elif row['from'] == 'menos' and row['to'] == 'menos' and row['effect'] > 0:
            return True
        elif row['from'] == 'mais' and row['to'] == 'menos' and row['effect'] < 0:
            return True
        elif row['from'] == 'menos' and row['to'] == 'mais' and row['effect'] < 0:
            return True
        else:
            return False
    
            

    def show_metrics(self, get_output = False):
        
        print(f"Quantidade de instâncias contrafactuais = {self.global_quant_contrafac_max}")
        print(f'Quantidade de relações causais na base de dados = {len(self.df_causal_effects)}')
        print(f"Quantidade de atributos modificados = {self.run_dict['global_numbers']['global_quant_changes']}")
        print(f"Quantidade de instâncias contrafactuais causais = {self.run_dict['global_numbers']['global_quant_contrafac_unico'] + self.run_dict['global_numbers']['global_quant_causal_contrafac']}")
        print(f"Quantidade de relações causais analisadas = {self.run_dict['global_numbers']['global_quant_causal_rules']}")
        print(f"Quantidade de relações causais satisfeitas = {self.run_dict['global_numbers']['global_quant_causal_changes']}")
        print(f"Quantidade de instâncias contrafactuais com um único atributo modificado = {self.run_dict['global_numbers']['global_quant_contrafac_unico']}")
        print(f"Tempo de execução = {self.run_dict['global_numbers']['global_timing_run_causal']}")
        
        if get_output:
            metrics_dict = {
                "Quantidade de instâncias contrafactuais": self.global_quant_contrafac_max,
                "Quantidade de relações causais na base de dados": len(self.df_causal_effects),
                "Quantidade de atributos modificados": self.run_dict['global_numbers']['global_quant_changes'],
                "Quantidade de instâncias contrafactuais causais": self.run_dict['global_numbers']['global_quant_contrafac_unico'] + self.run_dict['global_numbers']['global_quant_causal_contrafac'],
                "Quantidade de relações causais analisadas": self.run_dict['global_numbers']['global_quant_causal_rules'],
                "Quantidade de relações causais satisfeitas": self.run_dict['global_numbers']['global_quant_causal_changes'],
                "Quantidade de instâncias contrafactuais com um único atributo modificado": self.run_dict['global_numbers']['global_quant_contrafac_unico'],
                "Tempo de execução": self.run_dict['global_numbers']['global_timing_run_causal']
            }
        
            return metrics_dict

def get_causal_metrics(row, bb_model_name):
#     try:
    print(row)
    ccsse = CCSSE(row['name'], samples = 10, K = 10, generation= 10, bb_model = bb_model_name)
    print(f"criou o modelo")
    ccsse.run_causal()
    print(f"run_causal completo")
    dict_metricas = ccsse.show_metrics(get_output = True)
    converted_dict_metricas = convert_np_types(dict_metricas)
    json_data = json.dumps(converted_dict_metricas, indent=4)

    s3.put_object(Bucket='omar-testes-gerais', Key=f'artigos/causal_csse/bateria_metricas/outputs/metricas_1/{bb_model_name}/{row["name"]}.json', Body=json_data)
    print(f"Execução completa para {row['name']}")
#     except Exception as e:
#         print(f'Execução falhou - Nome da base de dados: {row["name"]}')
#         print(e)

def convert_np_types(data):
    """Converte tipos de dados NumPy em tipos nativos do Python."""
    if isinstance(data, dict):
        return {key: convert_np_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_np_types(item) for item in data]
    elif isinstance(data, np.int64):
        return int(data)  # Converte int64 para int
    elif isinstance(data, np.float64):
        return float(data)  # Converte float64 para float
    else:
        return data

#PJ LOCAL
# def handler(dataset, model_name):
    
#     args = {
#         "list_dataset_name": dataset,
#         'bb_model_name': model_name
#     }
    
#     df_map_inference_datasets = pd.read_parquet(f"s3://omar-testes-gerais/artigos/artifacts/df_map_inference_datasets.parquet")
#     df_dataset = df_map_inference_datasets[df_map_inference_datasets['name'].isin(args["list_dataset_name"])]
#     df_dataset.apply(lambda x: get_causal_metrics(x, args["bb_model_name"]), axis = 1)
#     return df_dataset
    

#PJ REAL
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    # Adicionando os argumentos para input e output
    parser.add_argument('--list_dataset_name', type=str)
    parser.add_argument('--bb_model_name', type=str)
    
    args = parser.parse_args()
    list_dataset_name = ast.literal_eval(args.list_dataset_name)

    df_map_inference_datasets = pd.read_parquet(f"s3://omar-testes-gerais/artigos/artifacts/dfm_use.parquet")
    df_dataset = df_map_inference_datasets[df_map_inference_datasets['name'].isin(list_dataset_name)]
    df_dataset.apply(lambda x: get_causal_metrics(x, args.bb_model_name), axis = 1)
