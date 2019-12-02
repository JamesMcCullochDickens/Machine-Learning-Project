import warnings
warnings.filterwarnings("ignore")
from sklearn import tree
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree.export import export_text
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import os
from skrules import SkopeRules
from scipy import stats
import matplotlib.pyplot as plt
import time
from sklearn.utils import shuffle
start = time.time()

train_dath_path = 'census-income.data/train_data'
test_data_path = 'census-income.test/test_data'
feature_names = ["Age", "ClassNotEmployed", "ClassPrivate", "ClassSelfEmployed", "ClassLocalGovernment",
                 "ClassStateGovernment", "ClassFederalGovernment", "IndustryCode",
                 "OccupationCode", "LessThanHighSchool:", "College", "Bachelors", "Masters", "ProfDegree", "Doctorate",
                 "Wage", "EnrolledEducation - Not in School", "EnrolledHighSchool", "EnrolledCollegeOrUniversity",
                 "IsNotMarried", "IsMarried", "IsDivorced",
                 "IsWidowed", "RaceAsian", "RaceWhite", "RaceOther",
                 "RaceAmericanIndian", "RaceBlack", "SexMale", "EmploymentStatusNotEmployed",
                 "EmploymentStatusPartTime", "EmploymentStatusFullTime",
                 "CapitalGains", "CapitalLosses", "StockDividends", "HeadOfHousehold", "JointFiler", "SingleFiler",
                 "NonFiler", "HasAmericanParent", "IsAmericanBorn", "WeeksWorked"]


# Read the data of a file into an array
def read_file_into_array(file_path):
    file_array = []
    file = open(file_path)
    lines = file.read().split("\n")
    for line in lines:
        if len(line) > 1:
            temp = []
            string_vals = line.split(', ')
            for i in range(0, len(string_vals)):
                temp.append(float(string_vals[i]))    # Convert the string values to floats for processing
            file_array.append(temp)
    file.close()
    return file_array  # remove the last entry as it is just white-space


# Read the pre-encoded data into arrays
train_data = read_file_into_array(train_dath_path)
test_data = read_file_into_array(test_data_path)


# Remove a feature at the specified index
def remove_feature(data_array, index):
    return_array = []
    for i in range(0, len(data_array)):
        temp = []
        for j in range(0, len(data_array[0])):
            if j != index:
                temp.append(data_array[i][j])
        return_array.append(temp)
    return return_array


# Remove the weights from the training and data set
train_data = remove_feature(train_data, 39)
test_data = remove_feature(test_data, 39)


# Separate the class from the features
def separate_features_from_class(data_array):
    feature_array = []
    class_array = []
    for i in range(0, len(data_array)):
        feature_temp = []
        class_temp = []
        for j in range(0, len(data_array[0])):
            if j != (len(data_array[0]) - 1):
                feature_temp.append(data_array[i][j])
            else:
                class_temp.append(data_array[i][j])
        feature_array.append(feature_temp)
        class_array.append(class_temp)
    return feature_array, class_array


# Print the number of majority and minority classes in a data set
def print_class_distribution(data, data_type):
    num_minority = 0
    num_majority = 0
    for i in range(0, len(data)):
        if data[i] == 0.0:
            num_majority = num_majority + 1
        else:
            num_minority = num_minority + 1
    print("The number of majority class instances in the " + data_type + " set is " + str(num_majority))
    print("The number of minority class instances in the " + data_type + " set is " + str(num_minority))
    print("The total is " + str(num_minority + num_majority))
    return num_majority, num_minority


# Get the training data and convert it to the appropriate format
X_train, Y_train = separate_features_from_class(train_data)
X_train = np.array(X_train)
Y_train = np.array(Y_train)
Y_train = Y_train.ravel()

# Get the test data and convert it to the appropriate format
X_test, Y_test = separate_features_from_class(test_data)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
Y_test = Y_test.ravel()

# Print the class distributions of the training and testing set before oversampling
print_class_distribution(Y_train, "training")
print_class_distribution(Y_test, "testing")

# Oversample the Minority Class
ros = RandomOverSampler(random_state=0, ratio={1: 20000, 0: 140529})
X_train, Y_train = ros.fit_resample(X_train, Y_train)
# Undersample the majority Class
ros = RandomUnderSampler(random_state=0, ratio={1: 20000, 0: 50000})
X_train, Y_train = ros.fit_resample(X_train, Y_train)
X_train, Y_train = shuffle(X_train, Y_train)

# Get validation data which is roughly 10% the size of the original training data
X_val = X_train[0:int(0.8 * len(X_train))]
Y_val = Y_train[0:int(0.8 * len(X_train))]
X_test_val = X_train[int(0.8 * len(X_train)):]
Y_test_val = Y_train[int(0.8 * len(X_train)):]

# Print the class distributions of the training and testing set after oversampling
print("After resampling")
print_class_distribution(Y_train, "training")
print_class_distribution(Y_test_val, "validation")


# Helper function that prints statistics for predictions from the models
# Prints accuracy, recall, f1 score, precision, and the confusion matrix
def print_statistics(model_name, y_pred, y_true):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    print("\nThe accuracy for the model " + model_name + " is {0:.2%}".format(accuracy))
    print("The minority class recall for the model " + model_name + " is {0:.2%}".format(recall))
    print("The minority class f1 score for the model " + model_name + " is {0:.2%}".format(f1))
    print("The minority class precision for the model " + model_name + " is {0:.2%}".format(precision))
    print("The confusion matrix is given by: ")
    print(confusion_matrix(y_true, y_pred))
    return accuracy


# Compute and print the results from cross validation
def print_cross_validation(model, model_name, data, target):
    scoring = {'accuracy': 'accuracy',
               'recall': 'recall',
               'precision': 'precision',
               'roc_auc': 'roc_auc'}
    scores = cross_validate(model, data, target, cv=10, scoring=scoring)
    print("\nPrinting the results per fold for the " + model_name + " model")
    for i in range(0, 10):
        print("\nFold " + str(i + 1) + ": Accuracy " + " {0:.2%}".format(scores['test_accuracy'][i]))
        print("Fold " + str(i + 1) + ": Recall " + " {0:.2%}".format(scores['test_recall'][i]))
        print("Fold " + str(i + 1) + ": Precision " + " {0:.2%}".format(scores['test_precision'][i]))
        print("Fold " + str(i + 1) + ": ROC area " + " {0:.2%}".format(scores['test_roc_auc'][i]))
    return scores


# Helper function to write the decision tree model to a file
def write_rules_to_file(data_array, file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        with open(file_path, 'w') as f:
            f.write(str(data_array))
            f.close()


# **************** Decision Tree Section ******************************
# Create three decision trees for validation and ultimately select 1
# First decision tree model
clf_val_gini = tree.DecisionTreeClassifier(max_depth=8, criterion="gini", min_samples_split=2)
clf_val_gini.fit(X_val, Y_val)
clf_val_gini_predict = clf_val_gini.predict(X_test_val)
print_statistics("Decision Tree - Gini criterion", clf_val_gini_predict, Y_test_val)
# Second decision tree model
clf_val_entropy = tree.DecisionTreeClassifier(max_depth=20, criterion="entropy", min_samples_split=2)
clf_val_entropy.fit(X_val, Y_val)
clf_val_entropy_predict = clf_val_entropy.predict(X_test_val)
print_statistics("Decision Tree - Entropy criterion", clf_val_entropy_predict, Y_test_val)
# Third decision tree model
clf_val_gini2 = tree.DecisionTreeClassifier(criterion="gini")
clf_val_gini2.fit(X_val, Y_val)
clf_val_gini2_predict = clf_val_gini2.predict(X_test_val)
print_statistics("Decision Tree - Gini criterion variant", clf_val_gini2_predict, Y_test_val)


print("In light of these results, the entropy decision tree will be chosen with max depth = 8, and minimum samples to split = 2")

# Next we fit the chosen decision tree classifier with the training data
dt_clf = tree.DecisionTreeClassifier(max_depth=8, criterion="gini", min_samples_split=2)

# Compute the cross validated accuracy scores
dt_clf_cross_validation_scores = print_cross_validation(dt_clf, "Decision Tree - Entropy criterion", X_train, Y_train)

# Train the decision tree and predict the results on the test set, and print the results
dt_clf.fit(X_train, Y_train)
dt_clf_pred = dt_clf.predict(X_test)
print_statistics("Decision Tree - Entropy criterion", dt_clf_pred, Y_test)

# Next we print the tree's rules and export them to a text file
tree_rules = export_text(dt_clf, feature_names=feature_names)
write_rules_to_file(tree_rules, "Tree Rules")

# **************** Nearest Neighbours Section *********************
# Next we will compare 5 nearest neighbour models on the validation set
# First knn model
knn_val_1 = KNeighborsClassifier(n_neighbors=5, weights="distance")
knn_val_1.fit(X_val, Y_val)
knn_val_1_predict = knn_val_1.predict(X_test_val)
print_statistics("Nearest Neighbours - Model 1 ", knn_val_1_predict, Y_test_val)
# Second knn model
knn_val_2 = KNeighborsClassifier(n_neighbors=5, metric="manhattan")
knn_val_2.fit(X_val, Y_val)
knn_val_2_predict = knn_val_2.predict(X_test_val)
print_statistics("Nearest Neighbours - Model 2 ", knn_val_2_predict,  Y_test_val)
# Third knn model
knn_val_3 = KNeighborsClassifier(n_neighbors=5, weights="distance", metric="manhattan")
knn_val_3.fit(X_val, Y_val)
knn_val_3_predict = knn_val_3.predict(X_test_val)
print_statistics("Nearest Neighbours - Model 3 ", knn_val_3_predict, Y_test_val)
# Fourth knn model
knn_val_4 = KNeighborsClassifier(n_neighbors=3, weights="distance")
knn_val_4.fit(X_val, Y_val)
knn_val_4_predict = knn_val_4.predict(X_test_val)
print_statistics("Nearest Neighbours - Model 4 ", knn_val_4_predict, Y_test_val)
# Fifth knn model
knn_val_5 = KNeighborsClassifier(n_neighbors=1, metric="manhattan")
knn_val_5.fit(X_val, Y_val)
knn_val_5_predict = knn_val_5.predict(X_test_val)
print_statistics("Nearest Neighbours - Model 5 ", knn_val_5_predict, Y_test_val)
print("\nAs per the results, the first nearest neighbours model with parameters 5 neighbours at a weighted distance, will be chosen")

# Fit the chosen nearest neighbours model to the training data
knn = KNeighborsClassifier(n_neighbors=5, weights="distance")

# Compute cross validation on the nearest neighbour model
knn_cross_validation_scores = print_cross_validation(knn, "Nearest Neighbours", X_train, Y_train)

# Run the nearest neighbour model on the test set and print the stats
knn.fit(X_train, Y_train)
knn_pred = knn.predict(X_test)
print_statistics("Nearest Neighbours", knn_pred, Y_test)

# ************* Linear Model: Neural Network with Semi-Supervised Learning! *******

# Neural network for semi-supervised learning
neural_network = Sequential()
neural_network.add(Dense(8, input_dim=42, activation='relu'))
neural_network.add(Dense(16, activation='relu'))
neural_network.add(Dense(8, activation='relu'))
neural_network.add(Dense(4, activation='relu'))
neural_network.add(Dense(1, activation='sigmoid'))
neural_network.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Get the indices which have a high probability for training the network
# only on high probability unsupervised labellings
def get_indices_of_high_probability(pred_array):
    indices = []
    count = 0
    for i in range(0, len(pred_array)):
        if pred_array[i] > 0.625:
            count = count + 1
            indices.append(i)
        elif pred_array[i] < 0.175:
            indices.append(i)
    return indices


# Helper function to split a list in two
def split_list(a_list):
    half = len(a_list) // 2
    return a_list[:half], a_list[half:]


# Compute Semi Supervised learning on the network
def compute_semi_supervised_learning(neural_net, training_features, training_classes):
    # First we split the array up as needed into 10 parts
    feature_array = []
    class_array = []
    num_partitions = 4
    increment_val = int(len(training_features) / num_partitions)
    counter = 0
    # Split the data into the appropriate number of partitions
    for i in range(0, num_partitions):
        if i != num_partitions - 1:
            feature_array.append(np.array(training_features[counter:counter + increment_val]))
            class_array.append(np.array(training_classes[counter:counter + increment_val]))
        else:
            feature_array.append(np.array(training_features[counter:len(training_features)]))
            class_array.append(np.array(training_classes[counter:len(training_features)]))
        counter = counter + increment_val
    # Train with the first partition (supervised)
    neural_net.fit(feature_array[0], class_array[0], epochs=10, batch_size=1000)
    for i in range(1, num_partitions):
        # Split the arrays sub-arrays dedicated for unsupervised and supervised learning
        feature_array_sl, feature_array_ul = split_list(feature_array[i])
        class_array_sl, class_array_ul = split_list(class_array[i])
        # Train the network on the supervised sub-array
        neural_net.fit(feature_array_sl, class_array_sl, epochs=10, batch_size=1000)

        # Get the predicted probabilities for the unclassified data
        pred_probabilities = np.array(neural_net.predict(feature_array_ul))
        pred_indices = get_indices_of_high_probability(pred_probabilities)
        pred_labels = neural_net.predict_classes(feature_array_ul)
        # Train the network only on unsupervised data that is predicted as 0 or 1 with high probability
        neural_net.fit(feature_array_ul[pred_indices], pred_labels[pred_indices], epochs=10, batch_size=1000)
    return neural_net


# Train the network, then make predictions on the test set and print the results
neural_network = compute_semi_supervised_learning(neural_network, X_train, Y_train)
neural_network_pred = np.array(neural_network.predict_classes(np.array(X_test)))
print_statistics("Semi-Supervised Neural Network", neural_network_pred, Y_test)


# ************* Rule Model:  ************************
# Here we compare 3 nearest neighbour models on the validation set
# First skope rules model
rule_clf1 = SkopeRules(
                       n_estimators=50,
                       precision_min=0.2,
                       recall_min=0.2,
                       feature_names=feature_names)
rule_clf1.fit(X_val, Y_val)
rule_clf1_ypred = rule_clf1.predict(X_test_val)
print_statistics("Rule Classifier - Skope Rules - Model 1", rule_clf1_ypred, Y_test_val)
# Second skope rules model
rule_clf2 = SkopeRules(n_estimators=50,
                       precision_min=0.2,
                       recall_min=0.2,
                       feature_names=feature_names)
rule_clf2.fit(X_val, Y_val)
rule_clf2_ypred = rule_clf2.predict(X_test_val)
print_statistics("Rule Classifier - Skope Rules - Model 2", rule_clf2_ypred, Y_test_val)
# Third skope rules model
rule_clf3 = SkopeRules(n_estimators=25,
                       precision_min=0.2,
                       recall_min=0.2,
                       feature_names=feature_names)
rule_clf3.fit(X_val, Y_val)
rule_clf3_ypred = rule_clf3.predict(X_test_val)
print_statistics("Rule Classifier - Skope Rules - Model 3", rule_clf3_ypred, Y_test_val)

print("\nAs per the results, Skope Rules model 1 will be chosen for training")
rule_clf = SkopeRules(n_estimators=50,
                      precision_min=0.2,
                      recall_min=0.2,
                      feature_names=feature_names)

# Run 10-fold cross validation on the
rule_clf_cross_validation_scores = print_cross_validation(rule_clf, "Skope Rules", X_train, Y_train)

# Train the skope rules model on the training data and print the results on the test data
rule_clf.fit(X_train, Y_train)
rule_clf_pred = rule_clf.predict(X_test)
print_statistics("Skope Rules", rule_clf_pred, Y_test)
rules = rule_clf.rules_
for rule in rules:
    print(rule)

# ************* Ensemble Model: AdaBoostClassifier **********************
# Here we compare three AdaBoostClassifiers on the validation set
# First ada boost model
ada_boost_clf1 = AdaBoostClassifier(n_estimators=50, random_state=0)
ada_boost_clf1.fit(X_val, Y_val)
ada_boost_clf1_ypred = ada_boost_clf1.predict(X_test_val)
print_statistics("Ada Boost Classifier - Model 1", ada_boost_clf1_ypred, Y_test_val)
# Second ada boost model
ada_boost_clf2 = AdaBoostClassifier(n_estimators=50, random_state=0, learning_rate=0.5)
ada_boost_clf2.fit(X_val, Y_val)
ada_boost_clf2_ypred = ada_boost_clf2.predict(X_test_val)
print_statistics("Ada Boost Classifier - Model 2", ada_boost_clf2_ypred, Y_test_val)
# Third ada boost model
ada_boost_clf3 = AdaBoostClassifier(n_estimators=100, random_state=0, learning_rate=0.2)
ada_boost_clf3.fit(X_val, Y_val)
ada_boost_clf3_ypred = ada_boost_clf3.predict(X_test_val)
print_statistics("Ada Boost Classifier - Model 3", ada_boost_clf3_ypred, Y_test_val)

print("\nAs per the results, Model 1 will be chosen for training and testing")
ada_boost_clf = AdaBoostClassifier(n_estimators=100, random_state=0, learning_rate=0.2)

# Compute 10-fold cross validation on the Ada Boost Classifier
ada_boost_cross_validation_scores = print_cross_validation(ada_boost_clf, "Ada Boost Classifier", X_train, Y_train)

# Train the ada boost classifier model on the training data and print the results from testing on the test data
ada_boost_clf.fit(X_train, Y_train)
ada_boost_clf_pred = ada_boost_clf.predict(X_test)
print_statistics("Ada Boost Classifier", ada_boost_clf_pred, Y_test)


# Compute the Paired t signed rank test for a subset of pairs
# of the models on the accuracy and recall per fold respectively
def paired_t_test(data1, data2, model1name, model2name):
    accuracy_t, accuracy_p_value = stats.ttest_ind(data1["test_accuracy"], data2["test_accuracy"])
    recall_t, recall_p_value = stats.ttest_ind(data1["test_recall"], data2["test_recall"])
    print("\nPrinting stats for models " + model1name + " and " + model2name)
    print("The T statistic for accuracy is given by " + str(accuracy_t))
    if accuracy_p_value < 0.05:
        print("We reject the null hypothesis that the difference in accuracy of the models is not significantly different")
    else:
        print("The accuracy of the models is not significantly different")
    print("The T statistic for recall is given by " + str(recall_t))
    if recall_p_value < 0.05:
        print("We reject the null hypothesis that the difference in recall of the models is not significantly different")
    else:
        print("The recall of the models belongs is not significantly different")


# Compute the pairwise signed Wilcoxon's test (not for the neural network or the overall model)

paired_t_test(dt_clf_cross_validation_scores, knn_cross_validation_scores, "Decision Tree",
                           "Nearest Neighbours")
paired_t_test(dt_clf_cross_validation_scores, rule_clf_cross_validation_scores, "Decision Tree",
                           "Skope Rules")
paired_t_test(dt_clf_cross_validation_scores, ada_boost_cross_validation_scores, "Decision Tree",
                           "Ada Boost Classifier")
paired_t_test(knn_cross_validation_scores, rule_clf_cross_validation_scores, "Nearest Neighbours",
                           "Skope Rules")
paired_t_test(knn_cross_validation_scores, ada_boost_cross_validation_scores, "Nearest Neighbours",
                           "Ada Boost Classifier")
paired_t_test(rule_clf_cross_validation_scores, ada_boost_cross_validation_scores, "Skope Rule",
                           "Ada Boost Classifier")


# Here we take a majority vote of all the models developed and check out the results!
def compute_majority_vote(pred_array):
    predictions = []
    for i in range(0, len(pred_array[0])):
        votes_for_minority = 0
        votes_for_majority = 0
        for j in range(0, len(pred_array)):
            if pred_array[j][i] == 1:
                votes_for_minority = votes_for_minority + 1
            else:
                votes_for_majority = votes_for_majority + 1
        if votes_for_minority > votes_for_majority:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions


# Print the results of majority voting on the test data using all models
model_array = [dt_clf_pred, knn_pred, neural_network_pred, rule_clf_pred, ada_boost_clf_pred]
overall_pred = compute_majority_vote(model_array)
print_statistics("Overall Predictions", overall_pred, Y_test)

# Print the ROC Curves of each model
plt.figure(0).clf()

dt_clf_auc = roc_auc_score(dt_clf_pred, Y_test)
dt_clf_fpr, dt_clf_tpr, _ = roc_curve(dt_clf_pred, Y_test, pos_label=1)
plt.plot(dt_clf_fpr, dt_clf_tpr, label="dt_clf, auc = " + str(dt_clf_auc))

knn_auc = roc_auc_score(knn_pred, Y_test)
knn_fpr, knn_tpr, _ = roc_curve(knn_pred, Y_test)
plt.plot(knn_fpr, knn_tpr, label="knn_clf, auc = " + str(knn_auc))

neural_network_auc = roc_auc_score(neural_network_pred, Y_test)
neural_network_fpr, neural_network_tpr, _ = roc_curve(neural_network_pred, Y_test, pos_label=1)
plt.plot(neural_network_fpr, neural_network_tpr, label="neural net, auc = " + str(neural_network_auc))

rule_clf_auc = roc_auc_score(rule_clf_pred, Y_test)
rule_clf_fpr, rule_clf_tpr, _ = roc_curve(rule_clf_pred, Y_test, pos_label=1)
plt.plot(rule_clf_fpr, rule_clf_tpr, label="rule clf, auc = " + str(rule_clf_auc))

ada_boost_auc = roc_auc_score(ada_boost_clf_pred, Y_test)
ada_boost_fpr, ada_boost_tpr, _ = roc_curve(ada_boost_clf_pred, Y_test, pos_label=1)
plt.plot(ada_boost_fpr, ada_boost_tpr, label="ada boost clf, auc = " + str(ada_boost_auc))

overall_auc = roc_auc_score(overall_pred, Y_test)
overall_fpr, overall_tpr, _ = roc_curve(overall_pred, Y_test,  pos_label=1)
plt.plot(overall_fpr, overall_tpr, label="overall clf, auc = "+str(overall_auc))
plt.legend(loc=0)
print('The total running time of this script is {0:0.1f} seconds'.format(time.time() - start))
plt.show()
