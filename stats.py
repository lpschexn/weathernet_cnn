import torch

# Reference formulas https://towardsdatascience.com/precision-and-recall-88a3776c8007

def calc_total_correct(outputs, targets):
    pass

def calc_total_false_positives(outputs, targets):
    pass

def calc_total_true_positives(outputs, targets):
    pass

def calc_true_sum(targets):
    pass

def calc_false_sum(targets):
    pass

def calc_true_positive_rate(outputs, targets):
    # Calculate True Positive Rate (a.k.a. Recall)
    # Inputs:
    #   - outputs: tensor structure output from CNN
    #   - targets: tensor structure with skitti answers
    # Outputs:
    #   - true positive rate (ratio of correctly guessed true positives and anything that should have been positive)

    # Get the indices that show the best guess of no snow/snow (indices 0 and 1)
    output_guesses = torch.argmax(outputs, 1)
    target_answers = torch.argmax(targets, 1)

    # AND operation to find all instances where the guess matches the answer.
    # Results in array of ints where 1s are where snow points are matched by the CNN and the label
    correct_guesses = torch.logical_and(output_guesses, target_answers).int()

    # Sum up correct guesses into one integer value
    total_correct = torch.sum(correct_guesses)

    # Sum up target answer to get total number of true positives
    true_sum = torch.sum(target_answers)

    return total_correct/true_sum

def calc_false_positive_rate(outputs, targets):
    # Calculate False Positive Rate
    # Inputs:
    #   - outputs: tensor structure output from CNN
    #   - targets: tensor structure with skitti answers
    # Outputs:
    #   - false positive rate

    # Get the indices that shows the best guess of no snow/snow (arrays of indices 0 and 1)
    output_guesses = torch.argmax(outputs, 1)
    target_answers = torch.argmax(targets, 1)

    # Identify only cases where output is positive and true answer is false (output_guesses AND NOT target_answers)
    not_target_answers = torch.logical_not(target_answers).int()
    false_positives = torch.logical_and(output_guesses, not_target_answers).int()

    # Sum up false positives
    total_false_positives = torch.sum(false_positives)

    # Get total negatives
    false_sum = torch.sum(not_target_answers)

    return total_false_positives/(false_sum)

def calc_precision(outputs, targets):
    # Calculate Precision
    # Inputs:
    #   - outputs: tensor structure output from CNN
    #   - targets: tensor structure with skitti answers
    # Outputs:
    #   - precision (ratio of true positives and anything guessed positive)

    # Get the indices that show the best guess of no snow/snow (indices 0 and 1)
    output_guesses = torch.argmax(outputs, 1)
    target_answers = torch.argmax(targets, 1)

    # Find True Positives (output_guesses AND target_answers)
    true_positives = torch.logical_and(output_guesses, target_answers).int()

    # Find False Positives (output_guesses AND NOT target_answers)
    not_target_answers = torch.logical_not(target_answers).int()
    false_positives = torch.logical_and(output_guesses, not_target_answers).int()

    # Sum up correct guesses into one integer value
    total_true_positives = torch.sum(true_positives)
    total_false_positives = torch.sum(false_positives)

    return total_true_positives/(total_true_positives + total_false_positives)