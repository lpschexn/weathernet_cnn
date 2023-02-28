from stats import WNStatTracker
import torch

NOT_FALLING_SNOW = 0
FALLING_SNOW = 1

class TestWNStatTracker:

    def __create_target(self, labels):
        """
        Creates a target (with one batch) using the NOT_FALLING_SNOW and FALLING_SNOW labels
        """
        height, width = labels.shape
        target = torch.zeros((1, 2, height, width))

        # Convert to 2-channel labels
        for j in range(height):
            for k in range(width):
                # If NOT_FALLING_SNOW, place 1.0 in 0th channel
                # If FALLING_SNOW, place 1.0 in 1st channel
                target[0, labels[j, k], j, k] = 1.0 

        return target
    
    def get_set_1(self):
        """
        True Positive Rate (Recall) = True Positives / (True Positivies and False Negatives)
        False Positive Rate = False Positives / (True Negatives and False Positives)
        Precision = True Positives / (True Positives and False Positives)

        TP rate = 1.0
        FP rate = 0.5
        Precision = 2/3

        Returns a Tuple of predicted, truth, true positive rate, false positive rate, precision
        """

        # Create truth data
        truth_data = [[NOT_FALLING_SNOW, FALLING_SNOW],
                      [FALLING_SNOW, NOT_FALLING_SNOW]]
        truth = self.__create_target(torch.tensor(truth_data))

        # Create predicted data
        predicted_data = [[NOT_FALLING_SNOW, FALLING_SNOW],
                          [FALLING_SNOW, FALLING_SNOW]]
        predicted = self.__create_target(torch.tensor(predicted_data))

        return predicted, truth, 1.0, 0.5, 2/3
    
    def get_set_2(self):
        """
        True Positive Rate (Recall) = True Positives / (True Positivies and False Negatives)
        False Positive Rate = False Positives / (True Negatives and False Positives)
        Precision = True Positives / (True Positives and False Positives)

        TP rate = 0.0
        FP rate = 1.0
        Precision = 0.0

        Returns a Tuple of predicted, truth, true positive rate, false positive rate, precision
        """

        # Create truth data
        truth_data = [[FALLING_SNOW, NOT_FALLING_SNOW],
                      [NOT_FALLING_SNOW, NOT_FALLING_SNOW]]
        truth = self.__create_target(torch.tensor(truth_data))

        # Create predicted data
        predicted_data = [[NOT_FALLING_SNOW, FALLING_SNOW],
                          [FALLING_SNOW, FALLING_SNOW]]
        predicted = self.__create_target(torch.tensor(predicted_data))

        return predicted, truth, 0.0, 1.0, 0.0
    
    def get_set_3(self):
        """
        True Positive Rate (Recall) = True Positives / (True Positivies and False Negatives)
        False Positive Rate = False Positives / (True Negatives and False Positives)
        Precision = True Positives / (True Positives and False Positives)

        TP rate = N/A, -1
        FP rate = 0.75
        Precision = 0.0

        Returns a Tuple of predicted, truth, true positive rate, false positive rate, precision
        """

        # Create truth data
        truth_data = [[NOT_FALLING_SNOW, NOT_FALLING_SNOW],
                      [NOT_FALLING_SNOW, NOT_FALLING_SNOW]]
        truth = self.__create_target(torch.tensor(truth_data))

        # Create predicted data
        predicted_data = [[NOT_FALLING_SNOW, FALLING_SNOW],
                          [FALLING_SNOW, FALLING_SNOW]]
        predicted = self.__create_target(torch.tensor(predicted_data))

        return predicted, truth, -1, 0.75, 0.0 # N/A -> -1
    
    def eval_set(self, test_set_cb):
        stats = WNStatTracker()
        predicted, truth, tp_rate, fp_rate, precision = test_set_cb()
        eval_precision, eval_tp_rate, eval_fp_rate = stats.calc_sample_stats(predicted, truth)
        assert precision == eval_precision
        assert tp_rate == eval_tp_rate
        assert fp_rate == eval_fp_rate

    def test_set_1(self):
        self.eval_set(self.get_set_1)

    def test_set_2(self):
        self.eval_set(self.get_set_2)

    def test_set_3(self):
        self.eval_set(self.get_set_3)

    def test_cumulative(self):
        stats = WNStatTracker()

        predicted, truth, tp_rate, fp_rate, precision = self.get_set_1()
        eval_precision, eval_tp_rate, eval_fp_rate = stats.calc_sample_stats(predicted, truth)
        eval_precision, eval_tp_rate, eval_fp_rate = stats.calc_sample_stats(predicted, truth)

        # Make sure cumulative stats are accurate
        eval_precision, eval_tp_rate, eval_fp_rate,_ = stats.calc_cumulative_stats()
        assert precision == eval_precision
        assert tp_rate == eval_tp_rate
        assert fp_rate == eval_fp_rate

        