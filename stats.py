import torch

# Reference formulas https://towardsdatascience.com/precision-and-recall-88a3776c8007

class WNStatTracker:
    def __init__(self):
        self.cumulative_false_positives = 0
        self.cumulative_true_positives  = 0
        self.cumulative_true_sum        = 0
        self.cumulative_false_sum       = 0
        self.num_samples                = 0

    def __get_label_values(self, input):
        """
        # Input: tensor structure of CNN's output or SemanticKitti answers
        # Output: Indices showing no snow/snow estimate or answer (indices are 0 or 1)
        """
        return torch.argmax(input, 1)

    def __calc_total_false_positives(self, outputs, targets):
        output_guesses = self.__get_label_values(outputs)
        target_answers = self.__get_label_values(targets)

        # Identify only cases where output is positive and true answer is false (output_guesses AND NOT target_answers)
        not_target_answers = torch.logical_not(target_answers).int()
        false_positives = torch.logical_and(output_guesses, not_target_answers).int()

        # Sum up false positives
        return torch.sum(false_positives).int()

    def __calc_total_true_positives(self, outputs, targets):
        output_guesses = self.__get_label_values(outputs)
        target_answers = self.__get_label_values(targets)

        # Find True Positives (output_guesses AND target_answers)
        true_positives = torch.logical_and(output_guesses, target_answers).int()

        return torch.sum(true_positives).int()

    def __calc_true_sum(self, targets):
        target_answers = self.__get_label_values(targets)
        return torch.sum(target_answers).int()

    def __calc_false_sum(self, targets):
        target_answers = self.__get_label_values(targets)
        not_target_answers = torch.logical_not(target_answers).int()
        return torch.sum(not_target_answers)

    def __calc_true_positive_rate(self, outputs, targets):
        """
        Calculate True Positive Rate (a.k.a. Recall)
         Inputs:
           - outputs: tensor structure output from CNN
           - targets: tensor structure with skitti answers
         Outputs:
           - true positive rate (ratio of correctly guessed true positives and anything that should have been positive)
        """
        # Sum up target answer to get total number of true positives
        ground_truth_true_sum = self.__calc_true_sum(targets)

        # If the ground truth does not have any true values, do not update cumulative values and report back a zero.
        if(ground_truth_true_sum == 0):
            return -1
        
        # Get total number of correct values in this batch
        total_true_positives = self.__calc_total_true_positives(outputs, targets)

        return total_true_positives/ground_truth_true_sum

    def __calc_false_positive_rate(self, outputs, targets):
        """
        Calculate False Positive Rate
         Inputs:
          - outputs: tensor structure output from CNN
          - targets: tensor structure with skitti answers
         Outputs:
          - false positive rate
        """

        ground_truth_false_sum = self.__calc_false_sum(targets)

        # If the ground truth does not have any false values, do not update cumulative values and report back a zero.
        if(ground_truth_false_sum == 0):
            return -1

        total_false_positives = self.__calc_total_false_positives(outputs, targets)

        return total_false_positives/ground_truth_false_sum

    def __calc_precision(self, outputs, targets):
        """
        Calculate Precision
        Inputs:
          - outputs: tensor structure output from CNN
          - targets: tensor structure with skitti answers
        Outputs:
          - precision (ratio of true positives and anything guessed positive)
        """
        total_true_positives  = self.__calc_total_true_positives(outputs, targets)
        total_false_positives = self.__calc_total_false_positives(outputs, targets)

        # Update count for true positives. Update false positives with false positive rate

        return total_true_positives/(total_true_positives + total_false_positives)
    
    def __calc_cumulative_true_positive_rate(self):
        return self.cumulative_true_positives/self.cumulative_true_sum

    def __calc_cumulative_false_positive_rate(self):
        return self.cumulative_false_positives/(self.cumulative_false_sum)

    def __calc_cumulative_precision(self):
        return self.cumulative_true_positives/(self.cumulative_true_positives + self.cumulative_false_positives)

    def calc_sample_stats(self, outputs, targets):
        """
        Generates precision, recall, and false positive rate for a specific sample.
        Updates the cumulative stats values
        Inputs:
          - outputs: tensor structure output from CNN
          - targets: tensor structure with skitti answers
        Outputs:
          - sample precision
          - sample recall
          - sample false positive rate
        """

        # Calculates the important stats
        precision           = self.__calc_precision(outputs, targets).item()
        recall              = self.__calc_true_positive_rate(outputs, targets).item()
        false_positive_rate = self.__calc_false_positive_rate(outputs, targets).item()

        # Increment the cumulative stats only if the ground truth true/false results are non-zero
        true_sum  = self.__calc_true_sum(targets)
        false_sum = self.__calc_false_sum(targets)

        if(true_sum > 0):
            self.cumulative_true_sum        += true_sum
            self.cumulative_true_positives  += self.__calc_total_true_positives(outputs, targets)

        if(false_sum > 0):
            self.cumulative_false_sum       += false_sum
            self.cumulative_false_positives += self.__calc_total_false_positives(outputs, targets)

        self.num_samples += 1

        return precision, recall, false_positive_rate
    
    def calc_cumulative_stats(self):
        """
        Generates precision, recall, and false positive rate of all inputs.
        Updates the cumulative stats values
        Inputs:
          - outputs: tensor structure output from CNN
          - targets: tensor structure with skitti answers
         Outputs:
          - cumulative precision
          - cumulative recall
          - cumulative false positive rate
          - number of samples
        """

        cumulative_precision = self.__calc_cumulative_precision()
        cumulative_recall    = self.__calc_cumulative_true_positive_rate()
        cumulative_false_positive = self.__calc_cumulative_false_positive_rate()

        return cumulative_precision, cumulative_recall, cumulative_false_positive, self.num_samples
    
    def reset_cumulative_stats(self):
        self.cumulative_false_positives = 0
        self.cumulative_true_positives  = 0
        self.cumulative_true_sum        = 0
        self.cumulative_false_sum       = 0
        self.num_samples                = 0