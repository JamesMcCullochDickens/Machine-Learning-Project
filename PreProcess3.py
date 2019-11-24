import numpy as np
import os

# Paths for writing
training_data_path = 'census-income.data/training_data_preprocess2'
testing_data_path = 'census-income.test/testing_data_preprocess2'


# Read the data of a file into an array
def read_file_into_array(file_path):
    file_array = []
    file = open(file_path)
    lines = file.read().split("\n")
    for line in lines:
        file_array.append(line.split(', '))
    file.close()
    return file_array[:-1]  # remove the last entry as it is just white-space


# Read the files into arrays for processing the training and testing data
train_data = read_file_into_array(training_data_path)
test_data = read_file_into_array(testing_data_path)

# Declare arrays with the different attribute values
num_attributes = 18
class_of_worker_attributes = ['Not Employed', 'Private', 'Self-Employed', 'Local Government', 'State Government', 'Federal Government']
education_attributes = ['Less than high school', 'College', 'Bachelors', 'Masters', 'Prof Degree', 'Doctorate']
education_enrollment_attributes = ['Not in School', 'High School', 'College or University']
married_attributes = ['Not Married', 'Married', 'Divorced', 'Widowed']
race_attributes = ['Asian or Pacific Islander', 'White', 'Other', 'Amer Indian Aleut or Eskimo', 'Black']
sex_attributes = ['Male, Female']
employment_attributes = ['Not Employed', 'Part Time', 'Full Time']
tax_filer_status_attributes = ['Head of Household', 'Joint', 'Single', 'Nonfiler']
country_of_birth_of_parents_attributes = ["USA", "Both not USA"]
country_of_birth_of_person = ["American", "Not American"]
income = ["- 50000.", "50000+."]


# A normalizing function for age that divides by 100
def normalize_age_values(data_array, index):
    for i in range(0, len(data_array)):
        data_array[i][index] = float(data_array[i][index])/float(90)
    return data_array


# Normalizes a real valued feature by dividing by max-min
def normalize_values(data_array, index, max_value, min_value):
    for i in range(0, len(data_array)):
        data_array[i][index] = (float(data_array[i][index])-min_value)/float(max_value-min_value)
    return data_array


# Turns a categorical variable with two values two a single binary digit
def binarization(data_array, index, value_array):
    for i in range(0, len(data_array)):
        if data_array[i][index] == value_array[0]:
            data_array[i][index] = float(0)
        else:
            data_array[i][index] = float(1)
    return data_array


# Normalize the data values
train_data = normalize_age_values(train_data, 0)   # Normalize the age values
test_data = normalize_age_values(test_data, 0)
train_data = normalize_values(train_data, 5, 10000, 0)  # Normalize the wage data
test_data = normalize_values(test_data, 5, 10000, 0)
train_data = normalize_values(train_data, 11, 100000, 0)  # Normalize the Capital Gains
test_data = normalize_values(test_data, 11, 100000, 0)
train_data = normalize_values(train_data, 12, 4608, 0)  # Normalize the Capital Losses
test_data = normalize_values(test_data, 12, 4608, 0)
train_data = normalize_values(train_data, 13, 100000, 0)  # Normalize the Stock Dividends
test_data = normalize_values(test_data, 13, 100000, 0)
train_data = normalize_values(train_data, 18, 52, 0)  # Normalize the number of weeks works
test_data = normalize_values(test_data, 18, 52, 0)


# Convert the Industry Code and Occupation Code to floats
for i in range(0, len(train_data)):
    train_data[i][2] = float(train_data[i][2])
    train_data[i][3] = float(train_data[i][3])

for i in range(0, len(test_data)):
    test_data[i][2] = float(test_data[i][2])
    test_data[i][3] = float(test_data[i][3])


# One hot encode a categorical value
def one_hot_encode(value_array, value):
    one_hot_encoding = []
    for i in range(0, len(value_array)):
        if value_array[i] == value:
            one_hot_encoding.append(float(1))
        else:
            one_hot_encoding.append(float(0))
    return one_hot_encoding


# One-hot encode an entire array
def one_hot_encode_feature(data_array, index, value_array):
    for i in range(0, len(data_array)):
        data_array[i][index] = one_hot_encode(value_array, data_array[i][index])
    return data_array


train_data = one_hot_encode_feature(train_data, 1, class_of_worker_attributes)      # One-hot encode the class of worker attribute
test_data = one_hot_encode_feature(test_data, 1, class_of_worker_attributes)
train_data = one_hot_encode_feature(train_data, 4, education_attributes)            # One-hot encode the educational attribute
test_data = one_hot_encode_feature(test_data, 4, education_attributes)
train_data = one_hot_encode_feature(train_data, 6, education_enrollment_attributes) # One-hot encode the enrollment attribute
test_data = one_hot_encode_feature(test_data, 6, education_enrollment_attributes)
train_data = one_hot_encode_feature(train_data, 7, married_attributes)              # One-hot encode the married attribute
test_data = one_hot_encode_feature(test_data, 7, married_attributes)
train_data = one_hot_encode_feature(train_data, 8, race_attributes)                 # One-hot encode the race attribute
test_data = one_hot_encode_feature(test_data, 8, race_attributes)
train_data = binarization(train_data, 9, sex_attributes)                            # Binarize the sex attribute
test_data = binarization(test_data, 9, sex_attributes)
train_data = one_hot_encode_feature(train_data, 10, employment_attributes)          # One-hot encode the emplyment attributs
test_data = one_hot_encode_feature(test_data, 10, employment_attributes)
train_data = one_hot_encode_feature(train_data, 14, tax_filer_status_attributes)    # One-hot encode the tax filer attribute
test_data = one_hot_encode_feature(test_data, 14, tax_filer_status_attributes)
train_data = binarization(train_data, 16, country_of_birth_of_parents_attributes)   # Binarize the country of the parents attrobite
test_data = binarization(test_data, 16, country_of_birth_of_parents_attributes)
train_data = binarization(train_data, 17, country_of_birth_of_person)               # Binarize the country of birth of the person attribute
test_data = binarization(test_data, 17, country_of_birth_of_person)
train_data = binarization(train_data, 19, income)                                   # Binarize the income data
test_data = binarization(test_data, 19, income)


# The indices of elements in the data that are lists for flattening
list_indices = [1, 4, 6, 7, 8, 10, 14]


# Custom flattening function
def flatten_data(data_array):
    return_array = []
    for i in range(0, len(data_array)):
        temp = []
        for j in range(0, len(data_array[0])):
            if j in list_indices:
                for t in range(0, len(data_array[i][j])):
                    temp.append(data_array[i][j][t])
            else:
                temp.append(data_array[i][j])
        return_array.append(temp)
    return return_array


# Flatten the arrays into a 1-d vector for training and testing
train_data = np.array(flatten_data(train_data)).astype(float)
test_data = np.array(flatten_data(test_data)).astype(float)


def get_possible_attribute_values(data_array, index):
    set_values = set()
    for i in range(0, len(data_array)):
        set_values.add(data_array[i][index])
    return set_values


industry_code_set = get_possible_attribute_values(train_data, 7) # get the possible values for the industry code attribute
occupation_code_set = get_possible_attribute_values(train_data, 8) # get the possible values for the occupation code attribute


def get_attribute_counts(data_array, attribute_set, index):
    count_dict = {}  # The number of values of a given attribute
    count_pos_dict = {}  # The number of values of a given attribute that are positive
    for x in attribute_set:
        count_dict[x] = 0
        count_pos_dict[x] = 0
        for i in range(0, len(data_array)):
            if data_array[i][index] == x:
                count_dict[x] = count_dict[x]+1
                if data_array[i][len(data_array[0]) - 1] == 1:
                    count_pos_dict[x] = count_pos_dict[x]+1
    return count_dict, count_pos_dict


industry_code_counts, industry_code_pos_counts = get_attribute_counts(train_data, industry_code_set, 7)
occupation_code_counts, occupation_code_pos_counts = get_attribute_counts(train_data, occupation_code_set, 8)


# Compute the calibration using the Laplace correction
def compute_calibration(total_count, num_pos):
    prior_odds = float(0.3504/(1-0.3504))  # The prior odds of an American making more than 50 K a year
    numerator = float(num_pos + 1)
    denominator = float(num_pos + 1 + prior_odds*(total_count - num_pos + 1))
    return float(numerator/denominator)


def calibrate_features(data_array, counts, pos_counts, index):
    for i in range(0, len(data_array)):
        data_array[i][index] = compute_calibration(counts[data_array[i][index]], pos_counts[data_array[i][index]])
    return data_array


# Calibrate the industry code and occupation code
train_data = calibrate_features(train_data, industry_code_counts, industry_code_pos_counts, 7)
test_data = calibrate_features(test_data, industry_code_counts, industry_code_pos_counts, 7)
train_data = calibrate_features(train_data, occupation_code_counts, occupation_code_pos_counts, 8)
test_data = calibrate_features(test_data, occupation_code_counts, occupation_code_pos_counts, 8)

np.random.shuffle(train_data)
np.random.shuffle(test_data)


# Convert the array to a string values for writing to a file
def convert_floats_to_string(data_array):
    return_array = []
    for i in range(0, len(data_array)):
        temp = []
        for j in range(0, len(data_array[0])):
            temp.append(str(data_array[i][j]))
        return_array.append(temp)
    return return_array


# Convert the float values to the string for writing to a file
train_data = convert_floats_to_string(train_data)
test_data = convert_floats_to_string(test_data)


# Write the pre-processed data to a file for the next stage of processing
def write_array_to_file(data_array, file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, 'w') as f:
        for i in range(0, len(data_array)):
            f.write("%s\n" % ", ".join(data_array[i]))
        f.close()


# Paths for writing
train_dath_path = 'census-income.data/train_data'
test_data_path = 'census-income.test/test_data'


# Write the pre-processed data to test files for the next step in the feature selection/pre-processing stage
write_array_to_file(train_data, train_dath_path)
write_array_to_file(test_data, test_data_path)