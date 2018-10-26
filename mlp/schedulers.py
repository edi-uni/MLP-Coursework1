# -*- coding: utf-8 -*-
"""Training schedulers.

This module contains classes implementing schedulers which control the
evolution of learning rule hyperparameters (such as learning rate) over a
training run.
"""

import numpy as np


class ConstantLearningRateScheduler(object):
    """Example of scheduler interface which sets a constant learning rate."""

    def __init__(self, learning_rate):
        """Construct a new constant learning rate scheduler object.

        Args:
            learning_rate: Learning rate to use in learning rule.
        """
        self.learning_rate = learning_rate

    def update_learning_rule(self, learning_rule, epoch_number):
        """Update the hyperparameters of the learning rule.

        Run at the beginning of each epoch.

        Args:
            learning_rule: Learning rule object being used in training run,
                any scheduled hyperparameters to be altered should be
                attributes of this object.
            epoch_number: Integer index of training epoch about to be run.
        """
        learning_rule.learning_rate = self.learning_rate

class CosineAnnealingWithWarmRestarts(object):
    """Cosine annealing scheduler, implemented as in https://arxiv.org/pdf/1608.03983.pdf"""

    def __init__(self, min_learning_rate, max_learning_rate, total_iters_per_period, max_learning_rate_discount_factor,
                 period_iteration_expansion_factor):
        """
        Instantiates a new cosine annealing with warm restarts learning rate scheduler
        :param min_learning_rate: The minimum learning rate the scheduler can assign
        :param max_learning_rate: The maximum learning rate the scheduler can assign
        :param total_epochs_per_period: The number of epochs in a period
        :param max_learning_rate_discount_factor: The rate of discount for the maximum learning rate after each restart i.e. how many times smaller the max learning rate will be after a restart compared to the previous one
        :param period_iteration_expansion_factor: The rate of expansion of the period epochs. e.g. if it's set to 1 then all periods have the same number of epochs, if it's larger than 1 then each subsequent period will have more epochs and vice versa.
        """
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.total_epochs_per_period = total_iters_per_period

        self.max_learning_rate_discount_factor = max_learning_rate_discount_factor
        self.period_iteration_expansion_factor = period_iteration_expansion_factor
        self.T_cur = 0.
        self.last_restart = 0
        self.previous_epoch_number = -1


    def update_learning_rule(self, learning_rule, epoch_number):
        """Update the hyperparameters of the learning rule.

        Run at the beginning of each epoch.

        Args:
            learning_rule: Learning rule object being used in training run,
                any scheduled hyperparameters to be altered should be
                attributes of this object.
            epoch_number: Integer index of training epoch about to be run.
        """
        
        multiplier = 0
        
        if (epoch_number - self.previous_epoch_number) > 1:
            if (self.last_restart + self.total_epochs_per_period) <= epoch_number:
                while (self.last_restart + self.total_epochs_per_period) <= epoch_number:
                    self.max_learning_rate *= self.max_learning_rate_discount_factor
                    self.total_epochs_per_period *= self.period_iteration_expansion_factor
                    self.last_restart += self.total_epochs_per_period

                self.T_cur = epoch_number - self.last_restart
                multiplier = self.min_learning_rate + 0.5 * (self.max_learning_rate - self.min_learning_rate) * (1 + np.cos(np.pi * (self.T_cur/self.total_epochs_per_period)))
                
        else:
            if self.T_cur == (self.total_epochs_per_period - 1):
                self.T_cur *= 0.
                self.max_learning_rate *= self.max_learning_rate_discount_factor
                self.total_epochs_per_period *= self.period_iteration_expansion_factor
                self.last_restart = epoch_number
            else:
                self.T_cur = epoch_number - self.last_restart

            multiplier = self.min_learning_rate + 0.5 * (self.max_learning_rate - self.min_learning_rate) * (1 + np.cos(np.pi * (self.T_cur/self.total_epochs_per_period)))
            
        self.previous_epoch_number = epoch_number;
        
        learning_rule.learning_rate = multiplier
        return learning_rule.learning_rate



