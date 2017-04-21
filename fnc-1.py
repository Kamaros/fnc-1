from classifier import train_classifier
from features import extract_features, flatten_features
from utils.dataset import read_dataset
from utils.preprocessing import preprocess_dataframe, extract_labels, decipher_labels, oversample_minority_classes
from utils.scoring import print_cv_score, evaluate_submission
from utils.splits import generate_hold_out_split

if __name__ == "__main__":
    print('Reading data...')
    raw_data = read_dataset('data')
    data = preprocess_dataframe(raw_data, 'raw_data')
    labels = extract_labels(data)

    print('Extracting features...')
    features = extract_features(data, raw_data)

    print('Flattening features...')
    flattened_features = flatten_features(features)

    print('Generating hold-out split...')
    training_data, testing_data, unused_data = generate_hold_out_split(raw_data)
    training_features, testing_features = flattened_features.iloc[training_data.index], flattened_features.iloc[testing_data.index]
    training_labels, testing_labels = labels.iloc[training_data.index], labels.iloc[testing_data.index]

    print('Oversampling minority classes...')
    oversampled_training_features, oversampled_training_labels = oversample_minority_classes(training_features, training_labels)

    print('Training classifier...')
    classifier = train_classifier(oversampled_training_features, oversampled_training_labels)

    print('Cross-validating...')
    print_cv_score(classifier, training_features, training_labels, cv=5)

    print('Generating predictions...')
    predictions = testing_data.copy()
    predictions['Stance'] = decipher_labels(classifier.predict(testing_features), index=testing_features.index)
    evaluate_submission(testing_data, predictions)