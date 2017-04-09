from features import extract_features, features2matrix
from utils.dataset import Dataset
from utils.preprocessing import preprocess_dataframe, extract_labels, decipher_labels

if __name__ == "__main__":
    # Extract and preprocess dataset
    dataset = Dataset(path='data')
    training_dataset, testing_dataset = dataset.generate_hold_out_split()
    raw_training_data = training_dataset.data
    raw_testing_data = testing_dataset.data
    training_data = preprocess_dataframe(raw_training_data, 'training_data')
    testing_data = preprocess_dataframe(raw_testing_data, 'testing_data')
    training_labels = extract_labels(training_data)
    testing_labels = extract_labels(testing_data)

    # Extract features
    training_features = extract_features(training_data, raw_training_data, 'training')
    testing_features = extract_features(testing_data, raw_testing_data, 'testing')
    np_training_features = features2matrix(training_features)
    np_testing_features = features2matrix(testing_features)