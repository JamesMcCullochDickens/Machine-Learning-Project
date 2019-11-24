import os

# Paths for writing
training_data_path_weighted_duplicates = 'census-income.data/training_data_preprocess1'
testing_data_path_without_weights = 'census-income.test/testing_data_preprocess1'


# Read the data of a file into an array
def read_file_into_array(file_path):
    file_array = []
    file = open(file_path)
    lines = file.read().split("\n")
    for line in lines:
        file_array.append(line.split(', '))
    file.close()
    return file_array[:-1]  # remove the last entry as it is just white-space


train_data = read_file_into_array(training_data_path_weighted_duplicates)
test_data = read_file_into_array(testing_data_path_without_weights)
num_attributes = 42
features_to_keep = [0, 1, 2, 3, 4, 5, 6, 7, 10, 12, 15, 16, 17, 18, 19, 24, 32, 33, 34, 39, 41]


# Remove the weight feature (and year completed) from the data
def remove_features(data, features_indices):
    array_without_feature = []
    for i in range(0, len(data)):
        temp = []
        for j in range(0, num_attributes):
            if j in features_indices:
                temp.append(data[i][j])
        array_without_feature.append(temp)
    return array_without_feature


# Remove the features not in the array features_to_keep
train_data = remove_features(train_data, features_to_keep)
test_data = remove_features(test_data, features_to_keep)


# Replace missing values with another value
def replace_feature_value(data_array, index, func):
    for i in range(0, len(data_array)):
        data_array[i][index] = func(data_array[i][index])
    return data_array


# Simplify the class of worker
def class_of_worker_replacement(data_value):
    if data_value == "Not in universe" or data_value == "Never worked" or data_value == "Without pay":
        return "Not Employed"
    elif data_value == "Self-employed-incorporated" or data_value == "Self-employed-not incorporated":
        return "Self-Employed"
    return data_value


# Simplify the education level, in particular all the uknnecessary values below high school
def simplify_education_level_replace(data_value):
    if data_value == "Children" or data_value == "Less than 1st grade" or data_value == "1st 2nd 3rd or 4th grade" \
        or data_value == "5th or 6th grade" or data_value == "7th and 8th grade" or data_value == "5th or 6th grade" \
        or data_value == "7th and 8th grade" or data_value == "9th grade" or data_value == "10th grade" or data_value == "11th grade" \
        or data_value == "12th grade no diploma":
             return "Less than High School"
    elif data_value == "Some college but no degree" or data_value == "Associates degree-academic program" or data_value == "Associates degree-occup /vocational":
            return "College"
    elif data_value == "Bachelors degree(BA AB BS)":
            return "Bachelors"
    elif data_value == "Masters degree(MA MS MEng MEd MSW MBA)":
            return "Masters"
    elif data_value == "Prof school degree (MD DDS DVM LLB JD)":
            return "Prof Degree"
    elif data_value == "Doctorate degree(PhD EdD)":
            return "Doctorate"
    else:
        return data_value


# Simplify the enrollment status
def simplify_enrollment_status(data_value):
     if data_value == "Not in universe":
         return "Not in School"
     else:
         return data_value


# Simplify the marriage status
def simplify_marital_status(data_value):
    if data_value == "Married-civilian spouse present" or data_value == "Married-spouse absent" or data_value == "Married-A F spouse present"\
        or data_value == "Married-civilian spouse present":
        return "Married"
    elif data_value == "Never married":
        return "Not Married"
    elif data_value == "Separated":
        return 'Divorced'
    return data_value


# Simplify the employment status by making it binary.
def simplify_employment_status(data_value):
    if data_value == "PT for non-econ reasons usually FT" or data_value == "Unemployed part- time" or data_value == "PT for econ reasons usually FT" or \
        data_value == "PT for econ reasons usually PT":
        return "Part-Time"
    elif data_value == "Unemployed full-time" or data_value == "Not in labor force":
        return "Unemployed"
    elif data_value == "Full-time schedules":
        return "Full-Time"
    return data_value


def simplify_tax_filer_status(data_value):
    if data_value == 'Joint both under 65' or data_value == "Joint one under 65 & one 65+" or data_value == "Joint both 65+":
        return "Joint"
    else:
        return data_value


# Simplify the birth country of the parents to be American or
# not American (American if at least one parent is born in America)
def simplify_birth_country_of_parents(data_array):
    for i in range(0, len(data_array)):
        if data_array[i][16] == "United-States" or data_array[i][16] == "USA" or  data_array[i][17] == "United-States" or data_array[i][17] == "USA":
            data_array[i][16] = "Has American Parent"
        else:
            data_array[i][16] = "Does not have American Parent"
    return data_array


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


# Simplify the features according to the function return values
train_data = replace_feature_value(train_data, 1, class_of_worker_replacement)
test_data = replace_feature_value(test_data, 1, class_of_worker_replacement)
train_data = replace_feature_value(train_data, 4, simplify_education_level_replace)
test_data = replace_feature_value(test_data, 4, simplify_education_level_replace)
train_data = replace_feature_value(train_data, 6, simplify_enrollment_status)
test_data = replace_feature_value(test_data, 6, simplify_enrollment_status)
train_data = replace_feature_value(train_data, 7, simplify_marital_status)
test_data = replace_feature_value(test_data, 7, simplify_marital_status)
train_data = replace_feature_value(train_data, 10, simplify_employment_status)
test_data = replace_feature_value(test_data, 10, simplify_employment_status)
train_data = replace_feature_value(train_data, 14, simplify_tax_filer_status)
test_data = replace_feature_value(test_data, 14, simplify_tax_filer_status)
train_data = simplify_birth_country_of_parents(train_data)
test_data = simplify_birth_country_of_parents(test_data)
train_data = remove_feature(train_data, 17)   # Get rid of one birthcountry feature
test_data = remove_feature(test_data, 17)


# Simplify the birthplace to a binary if they are or are not born in the USA
def simplify_birthplace(data_value):
    if data_value == "United-States":
        return "Born in United-States"
    else:
        return "Not born in United-States"


train_data = replace_feature_value(train_data, 17, simplify_birthplace)
test_data = replace_feature_value(test_data, 17, simplify_birthplace)

#Extra code for printing to a file friendly for Weka
''''
real_values = [0,2,3,5,11,12,13,18]

for i in range(0, len(train_data)):
    for j in range(0, len(train_data[0])):
        if j not in real_values:
            train_data[i][j] = "\""+train_data[i][j]+"\""
'''''


# Write the pre-processed data to a file for the next stage of processing
def write_array_to_file(data_array, file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, 'w') as f:
        for i in range(0, len(data_array)):
            f.write("%s\n" % ", ".join(data_array[i]))


# Paths for writing
simplified_training_data_path = 'census-income.data/training_data_preprocess2'
simplified_testing_data_path = 'census-income.test/testing_data_preprocess2'

# Write the pre-processed data to test files for the next step in the feature selection/pre-processing stage
write_array_to_file(train_data, simplified_training_data_path)
write_array_to_file(test_data, simplified_testing_data_path)