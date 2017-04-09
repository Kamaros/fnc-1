"""
Scorer for the Fake News Challenge. 
Adapted from https://github.com/FakeNewsChallenge/fnc-1/blob/master/scorer.py.
"""

LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
RELATED = LABELS[0:3]

ERROR_MISMATCH = """
ERROR: Entry mismatch at line {}
 [expected] Headline: {} // Body ID: {}
 [got] Headline: {} // Body ID: {}
"""

SCORE_REPORT = """
MAX  - the best possible score (100% accuracy)
NULL - score as if all predicted stances were unrelated
TEST - score based on the provided predictions

||    MAX    ||    NULL   ||    TEST   ||\n||{:^11}||{:^11}||{:^11}||
"""

class FNCException(Exception):
    pass

def score_submission(gold_labels, test_labels):
    """Calculates the score and confusion matrix for a submission.

    Scoring is as follows:
        +0.25 for each correct unrelated
        +0.25 for each correct related (label is any of agree, disagree, discuss)
        +0.75 for each correct agree, disagree, discuss

    Parameters
    ----------
    gold_labels : Pandas DataFrame
        DataFrame with reference GOLD stance labels.
    test_labels : Pandas DataFrame
        DataFrame with predicted stance labels.

    Returns
    -------
    score : float
        The numeric score for the submission.
    cm : list, shape
    evaluation : list, length = 2
        List where the first entry is the numeric score for the submission (float) and the second entry contains a representation of the confusion matrix (2-D array).
    """
    score = 0.0
    cm = [[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]

    if len(gold_labels) != len(test_labels):
        raise FNCException('Reference and prediction datasets are not of the same length.')
    
    # Named tuples require single-word keys, so rename "Body ID" to "BodyID".
    gold_labels = gold_labels.rename(columns={'Body ID': 'BodyID'})
    test_labels = test_labels.rename(columns={'Body ID': 'BodyID'})

    for i, (g, t) in enumerate(zip(gold_labels.itertuples(), test_labels.itertuples())):
        if g.Headline != t.Headline or g.BodyID != t.BodyID:
            error = ERROR_MISMATCH.format(i+2,
                                          g.Headline, g.BodyID,
                                          t.Headline, t.BodyID)
            raise FNCException(error)
        else:
            g_stance, t_stance = g.Stance, t.Stance
            if g_stance == t_stance:
                score += 0.25
                if g_stance != 'unrelated':
                    score += 0.50
            if g_stance in RELATED and t_stance in RELATED:
                score += 0.25

        cm[LABELS.index(g_stance)][LABELS.index(t_stance)] += 1

    return score, cm

def score_defaults(gold_labels):
    """Calculates the "all false" baseline (all labels as unrelated) and the max possible score.

    Parameters
    ----------
    gold_labels : Pandas DataFrame
        DataFrame with the reference GOLD stance labels.

    Returns
    -------
    null_score : float
        The score for the "all false" baseline.
    max_score : float
        The max possible score.
    """
    unrelated = [g for g in gold_labels.itertuples() if g.Stance == 'unrelated']
    null_score = 0.25 * len(unrelated)
    max_score = null_score + (len(gold_labels) - len(unrelated))
    return null_score, max_score

def print_confusion_matrix(cm):
    """Prints a representation of the confusion matrix to the console."""
    lines = ['CONFUSION MATRIX:']
    header = "|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format('', *LABELS)
    line_len = len(header)
    lines.append("-"*line_len)
    lines.append(header)
    lines.append("-"*line_len)

    hit = 0
    total = 0
    for i, row in enumerate(cm):
        hit += row[i]
        total += sum(row)
        lines.append("|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format(LABELS[i], *row))
        lines.append("-"*line_len)
    lines.append("ACCURACY: {:.3f}".format(hit / total))
    print('\n'.join(lines))

def evaluate_submission(gold_labels, test_labels):
    """Prints the confusion matrix and score report for a submission.

    Parameters
    ----------
    gold_labels : Pandas DataFrame
        DataFrame with reference GOLD stance labels.
    test_labels : Pandas DataFrame
        DataFrame with predicted stance labels.
    """
    try:
        test_score, cm = score_submission(gold_labels, test_labels)
        null_score, max_score = score_defaults(gold_labels)
        print_confusion_matrix(cm)
        print(SCORE_REPORT.format(max_score, null_score, test_score))
    except FNCException as e:
        print(e)