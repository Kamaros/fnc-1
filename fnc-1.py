from classifier import train_classifier
from features import extract_features, flatten_features, scale_features, decompose_features, select_features
from utils.dataset import read_dataset, generate_hold_out_split
from utils.preprocessing import preprocess_dataframe, extract_labels, decipher_labels
from utils.scoring import evaluate_submission

if __name__ == "__main__":
    # Extract and preprocess dataset
    data = read_dataset('data')
    raw_training_data, raw_testing_data = generate_hold_out_split(data)
    del data

    training_data = preprocess_dataframe(raw_training_data, 'training_data')
    testing_data = preprocess_dataframe(raw_testing_data, 'testing_data')

    training_labels = extract_labels(training_data)
    testing_labels = extract_labels(testing_data)

    # Extract and process features
    print('Extracting features...')
    training_features = extract_features(training_data, raw_training_data, 'training')
    testing_features = extract_features(testing_data, raw_testing_data, 'testing')

    print('Flattening features...')
    training_features = flatten_features(training_features)
    testing_features = flatten_features(testing_features)

    # print('Scaling features...')
    # training_features = scale_features(training_features)
    # testing_features = scale_features(testing_features)

    # print('Decomposing features...')
    # training_features = decompose_features(training_features)
    # testing_features = decompose_features(testing_features)

    # print('Selecting features...')
    # training_features = select_features(training_features, training_labels)
    # testing_features = select_features(testing_features)

    print('Training classifier...')
    classifier = train_classifier(training_features, training_labels)
    predictions = raw_testing_data.copy()
    predictions['Stance'] = decipher_labels(classifier.predict(testing_features), index=testing_features.index)
    evaluate_submission(raw_testing_data, predictions)