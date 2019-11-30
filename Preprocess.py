from operator import itemgetter
import os

# Arrays to store raw training and test data
training_data_path = 'census-income.data/census-income.data'
test_data_path = 'census-income.test/census-income.test'
num_attributes = 42


# Read the data of a file into an array
def read_file_into_array(file_path):
    file_array = []
    file = open(file_path)
    lines = file.read().split("\n")
    for line in lines:
        file_array.append(line.split(', '))
    file.close()
    return file_array[:-1]  # remove the last entry as it is just white-space


raw_training_data = read_file_into_array(training_data_path)
raw_test_data = read_file_into_array(test_data_path)

# Convert the age value to an integer for sorting by numeric value for both data sets
for index in range(0, len(raw_training_data)):
    raw_training_data[index][0] = int(raw_training_data[index][0])

print("The total number of instances in the training data is " + str(len(raw_training_data)))

for index in range(0, len(raw_test_data)):
    raw_test_data[index][0] = int(raw_test_data[index][0])

print("The total number of instances in the testing data is " + str(len(raw_test_data)))
print("The number of attributes in the testing data is " + str(len(raw_test_data[1])))

# Sort the data by age of the person
sorted_raw_training_data = sorted(raw_training_data, key=itemgetter(0))
sorted_raw_testing_data = sorted(raw_test_data, key=itemgetter(0))


# Get the possible values for each attribute into a dict
def get_possible_attribute_values(data_array):
    set_array = []
    for i in range(0, num_attributes):
        set_array.append(set())
    for i in range(0, len(data_array)):
        for j in range(0, num_attributes):
            set_array[j].add(data_array[i][j])
    return set_array


# Print the number of different values for each attribute
def print_num_attribute_values(set_array, attribute_index):
    print('There are ' + str(len(set_array[attribute_index])) + ' different values')
    print(set_array[attribute_index])


# Get the possible attribute arrays as sets for the training and testing data
training_data_attribute_set_array = get_possible_attribute_values(sorted_raw_training_data)
testing_data_attribute_set_array = get_possible_attribute_values(sorted_raw_testing_data)


# Get the min and max value for a given numerical attribute
def get_min_and_max_values(data_set, index):
    data_set = set(map(float, data_set[index]))
    return min(data_set), max(data_set)


# Print some stats about the data
print("The min,max value for the capital gains attribute for the data set is " +
      str(get_min_and_max_values(training_data_attribute_set_array, 16)))
print("The min,max value for the capital gains attribute for the test set is " +
      str(get_min_and_max_values(testing_data_attribute_set_array, 16)))
print("The min,max value for the capital losses attribute for the data set is " +
      str(get_min_and_max_values(training_data_attribute_set_array, 17)))
print("The min,max value for the capital losses attribute for the test set is " +
      str(get_min_and_max_values(testing_data_attribute_set_array, 17)))
print("The min,max value for the stock dividends attribute for the data set is " +
      str(get_min_and_max_values(training_data_attribute_set_array, 18)))
print("The min,max value for the stock dividends attribute for the test set is " +
      str(get_min_and_max_values(testing_data_attribute_set_array, 18)))
print("The min,max value for the wage attribute in the data set is " +
      str(get_min_and_max_values(training_data_attribute_set_array, 5)))


# Determine which attributes in both data sets have missing attributes
def print_attributes_with_missing_values(set_array):
    for i in range(0, num_attributes):
        if ('?' or ' ?') in set_array[i]:
            print('The attribute ' + str(i) + ' has missing values')


# Check out which values have missing values
print("\nMissing values in the training set (0 indexed)")
print_attributes_with_missing_values(training_data_attribute_set_array)
print("\nMissing values in the test set (0 indexed)")
print_attributes_with_missing_values(testing_data_attribute_set_array)


# Get the number of missing attributes given by a question mark in the data set
def count_number_of_missing_attribute(data_array):
    num_missing_values = 0
    for i in range(0, len(data_array)):
        for j in range(0, num_attributes):
            if data_array[i][j] == '?' or data_array[i][j] == ' ?':
                num_missing_values = num_missing_values + 1
    return num_missing_values


print("\nThe number of missing values in the training data set is " + str(count_number_of_missing_attribute(sorted_raw_training_data)))
print("The number of missing values in the testing data set is " + str(count_number_of_missing_attribute(sorted_raw_testing_data)))


# Creates a dictionary for each feature with the attribute count per attribute
def get_number_of_instances_with_attribute_value(data_array, data_set_array):
    attribute_count_array = []
    for i in range(0, num_attributes):
        dict = {}
        for x in data_set_array[i]:
            dict[x] = 0
        attribute_count_array.append(dict)
    for i in range(0, len(data_array)):
        for j in range(0, num_attributes):
            (attribute_count_array[j])[data_array[i][j]] = (attribute_count_array[j])[data_array[i][j]] + 1
    return attribute_count_array


# Get the number of instances per feature value in a dict and then
# print the number of each class in the training and testing set
# For example here we get the total number of positive and negative classes in
# the training and testing set
training_attribute_count_array = get_number_of_instances_with_attribute_value(sorted_raw_training_data, training_data_attribute_set_array)
testing_attribute_count_array = get_number_of_instances_with_attribute_value(sorted_raw_testing_data, testing_data_attribute_set_array)
print(training_attribute_count_array[41])
print(testing_attribute_count_array[41])


# Remove any duplicate data, where a duplicate is defined to be data where every entry is exactly the same
def remove_duplicates(data_array, data_type_string):
    num_duplicates = 0
    data_size = len(data_array)
    current_size = data_size  # Current size of the array
    index = 0
    while index < current_size:
        counter = 1
        # Only compare data of the same age for duplicates
        if index + counter < current_size:
            while data_array[index][0] == data_array[index + counter][0]:
                if data_array[index] == data_array[index + counter]:
                    del data_array[index + counter]
                    current_size = current_size - 1
                    num_duplicates = num_duplicates + 1
                else:
                    counter = counter + 1
                if not (index + counter < current_size):  # Make sure not to use indices that are too large
                    break
        index = index + 1
    print('The number of duplicate entries in the ' + str(data_type_string) + ' data is ' + str(num_duplicates))
    return data_array


# Remove duplicates from the training data set
sorted_raw_training_data = remove_duplicates(sorted_raw_training_data, 'training')
print("The number of training data instances after removing duplicate entries is " + str(len(sorted_raw_training_data)))
#print("The number of testing data instances after removing duplicate entries is " + str(len(sorted_raw_testing_data)))


# Convert the age value back to a string for printing/writing for both data sets
for index in range(0, len(sorted_raw_training_data)):
    sorted_raw_training_data[index][0] = str(sorted_raw_training_data[index][0])

for index in range(0, len(sorted_raw_testing_data)):
    sorted_raw_testing_data[index][0] = str(sorted_raw_testing_data[index][0])


# Check if there are conflicts, where a conflict is defined to be two raw instances
# which differ only in the instance weight value or the categorical outcome
# determined by the index parameter
def is_conflict(instance1, instance2, index):
    for i in range(0, num_attributes):
        if instance1[i] != instance2[i] and i != index:
            return False
    return instance1[index] != instance2[index]  # Different instance feature values


# Process weight conflicts in the data
def process_weight_conflicts(data_array, data_type_string):
    num_instance_weight_conflicts = 0
    current_data_size = len(data_array)
    index = 0
    while index < current_data_size:
        counter = 1
        # Only compare data of the same age for conflict issues
        if index + counter < current_data_size:
            while data_array[index][0] == data_array[index + counter][0]:
                if is_conflict(data_array[index], data_array[index + counter], 24):
                    # In case of a conflict, we will take the average of the instance weight values
                    average_instance_weight = str((float(data_array[index][24]) +
                                                   float(data_array[index + counter][24])) / 2)
                    data_array[index][24] = average_instance_weight
                    del data_array[index + counter]
                    num_instance_weight_conflicts = num_instance_weight_conflicts + 1
                    current_data_size = current_data_size - 1
                else:
                    counter = counter + 1
                if not (index + counter < current_data_size):
                    break
        index = index + 1
    print('The number of instance weight conflicts in the ' + data_type_string + ' data without duplicates is ' + str(
        num_instance_weight_conflicts))
    return data_array


sorted_raw_training_data = process_weight_conflicts(sorted_raw_training_data, 'training')
print("\nAfter processing weight conflicts the number of instances in the training set is " + str(len(sorted_raw_training_data)))
print("After processing weight conflicts the number of instances in the testing set is " + str(len(sorted_raw_testing_data)))


# Replace missing values with another value
def replace_missing_value(data_array, index, new_value):
    for i in range(0, len(data_array)):
        if data_array[i][index] == '?' or data_array[i][index] == ' ?':
            data_array[i][index] = new_value
    return data_array


# Replace the missing values with default values
replace_missing_value(sorted_raw_training_data, 21, 'California')
replace_missing_value(sorted_raw_training_data, 25, 'NonMover')
replace_missing_value(sorted_raw_training_data, 26, 'NonMover')
replace_missing_value(sorted_raw_training_data, 27, 'NonMover')
replace_missing_value(sorted_raw_training_data, 29, 'No')
replace_missing_value(sorted_raw_training_data, 32, 'USA')
replace_missing_value(sorted_raw_training_data, 33, 'USA')
replace_missing_value(sorted_raw_training_data, 34, 'USA')
replace_missing_value(sorted_raw_testing_data, 21, 'California')
replace_missing_value(sorted_raw_testing_data, 25, 'NonMover')
replace_missing_value(sorted_raw_testing_data, 26, 'NonMover')
replace_missing_value(sorted_raw_testing_data, 27, 'NonMover')
replace_missing_value(sorted_raw_testing_data, 29, 'No')
replace_missing_value(sorted_raw_testing_data, 32, 'USA')
replace_missing_value(sorted_raw_testing_data, 33, 'USA')
replace_missing_value(sorted_raw_testing_data, 34, 'USA')


# Remove the instance weights from the training data
def assign_duplicates_based_on_weight(data_array):
    array_with_weighted_duplicates = []
    for i in range(len(data_array)):
        instance_weight = data_array[i][24]
        num_duplicates = int((float(instance_weight)/20000) * 10)+1
        for j in range(0, num_duplicates):
            array_with_weighted_duplicates.append(data_array[i])
    return array_with_weighted_duplicates


''''' 
indices_to_add_quotes = [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 37, 41]
# Extra code for Weka
def add_quotation_marks(data_array):
    for i in range(0, len(data_array)):
        for j in range(0, len(data_array[0])):
            if j in indices_to_add_quotes:
                data_array[i][j] = "\"" + data_array[i][j] + "\""
    return data_array

#sorted_raw_training_data = add_quotation_marks(sorted_raw_training_data_weighted_duplicates)
'''''


# Write the pre-processed data to a file for the next stage of processing
def write_array_to_file(data_array, file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, 'w') as f:
        for i in range(0, len(data_array)):
            f.write("%s\n" % ", ".join(data_array[i]))
        f.close()


# Paths for writing
training_data_path_preprocess1 = 'census-income.data/training_data_preprocess1'
testing_data_path_preprocess1 = 'census-income.test/testing_data_preprocess1'

# Write the pre-processed data to test files for the next step in the feature selection/pre-processing stage
write_array_to_file(sorted_raw_training_data, training_data_path_preprocess1)
write_array_to_file(sorted_raw_testing_data, testing_data_path_preprocess1)