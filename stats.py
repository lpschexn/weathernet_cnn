import torch

# Reference formulas https://towardsdatascience.com/precision-and-recall-88a3776c8007

class WNStatTracker:
    def __init__(self):
        self.cumulative_false_positives = 0
        self.cumulative_true_positives  = 0
        self.cumulative_true_sum        = 0
        self.cumulative_false_sum       = 0

    # Input: tensor structure of CNN's output or SemanticKitti answers
    # Output: Indices showing no snow/snow estimate or answer (indices are 0 or 1)
    def get_label_values(self, input):
        return torch.argmax(input, 1)

    def calc_total_false_positives(self, outputs, targets):
        output_guesses = self.get_label_values(outputs)
        target_answers = self.get_label_values(targets)

        # Identify only cases where output is positive and true answer is false (output_guesses AND NOT target_answers)
        not_target_answers = torch.logical_not(target_answers).int()
        false_positives = torch.logical_and(output_guesses, not_target_answers).int()

        # Sum up false positives
        return torch.sum(false_positives).int()

    def calc_total_true_positives(self, outputs, targets):
        output_guesses = self.get_label_values(outputs)
        target_answers = self.get_label_values(targets)

        # Find True Positives (output_guesses AND target_answers)
        true_positives = torch.logical_and(output_guesses, target_answers).int()

        return torch.sum(true_positives).int()

    def calc_true_sum(self, targets):
        target_answers = self.get_label_values(targets)
        return torch.sum(target_answers).int()

    def calc_false_sum(self, targets):
        target_answers = self.get_label_values(targets)
        not_target_answers = torch.logical_not(target_answers).int()
        return torch.sum(not_target_answers)

    def calc_true_positive_rate(self, outputs, targets):
        '''
        Calculate True Positive Rate (a.k.a. Recall)
         Inputs:
           - outputs: tensor structure output from CNN
           - targets: tensor structure with skitti answers
         Outputs:
           - true positive rate (ratio of correctly guessed true positives and anything that should have been positive)
        '''
        # Sum up target answer to get total number of true positives
        ground_truth_true_sum = self.calc_true_sum(targets)

        # If the ground truth does not have any true values, do not update cumulative values and report back a zero.
        if(ground_truth_true_sum == 0):
            return 0
        
        # Get total number of correct values in this batch
        total_true_positives = self.calc_total_true_positives(outputs, targets)
   
        # Update count for all inputs
        self.cumulative_true_positives += total_true_positives
        self.cumulative_true_sum += ground_truth_true_sum

        return total_true_positives/ground_truth_true_sum

    def calc_false_positive_rate(self, outputs, targets):
        '''
        Calculate False Positive Rate
         Inputs:
          - outputs: tensor structure output from CNN
          - targets: tensor structure with skitti answers
         Outputs:
          - false positive rate
        '''

        ground_truth_false_sum = self.calc_false_sum(targets)

        # If the ground truth does not have any false values, do not update cumulative values and report back a zero.
        if(ground_truth_false_sum == 0):
            return 0

        total_false_positives = self.calc_total_false_positives(outputs, targets)

        # Update count for all inputs
        self.cumulative_false_positives += total_false_positives
        self.cumulative_false_sum += ground_truth_false_sum

        return total_false_positives/ground_truth_false_sum

    def calc_precision(self, outputs, targets):
        '''
        Calculate Precision
        Inputs:
          - outputs: tensor structure output from CNN
          - targets: tensor structure with skitti answers
        Outputs:
          - precision (ratio of true positives and anything guessed positive)
        '''
        total_true_positives  = self.calc_total_true_positives(outputs, targets)
        total_false_positives = self.calc_total_false_positives(outputs, targets)

        # Update count for true positives. Update false positives with false positive rate

        return total_true_positives/(total_true_positives + total_false_positives)
    
    def calc_cumulative_true_positive_rate(self):
        return self.cumulative_true_positives/self.cumulative_true_sum

    def calc_cumulative_false_positive_rate(self):
        return self.cumulative_false_positives/(self.cumulative_false_sum)

    def calc_cumulative_precision(self):
        return self.cumulative_true_positives/(self.cumulative_true_positives + self.cumulative_false_positives)
