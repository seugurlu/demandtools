import linearmodels.system as lms
import numpy as np
import pandas as pd
import collections
import torch
import torch.nn as nn
import torch.utils.data as utils_data


class Aids:

    def __init__(self, price, expenditure, budget_share, demographic=None,
                 price_log=True, expenditure_log=True, alpha0=0):
        if price_log is False:
            price = np.log(price)  # Turn price data into logs.
        if expenditure_log is False:
            expenditure = np.log(expenditure)  # Turn expenditure data into logs.

        number_observation, number_good = price.shape
        if demographic is None:
            number_demographic = 0
            demographic_header = []
        else:
            number_demographic = demographic.shape[1]  # Number of columns of demographics input data.
            demographic_header = demographic.columns
            demographic = np.array(demographic, ndmin=2)

        try:
            price_header = price.columns  # If input is pandas data frame.
        except AttributeError:
            price_header = []  # If input is numpy array.
        price = np.array(price, ndmin=2)
        expenditure = np.array(expenditure, ndmin=2).reshape(number_observation, 1)
        try:
            budget_share_header = budget_share.columns
        except AttributeError:
            budget_share_header = []
        budget_share = np.array(budget_share, ndmin=2)

        # Assign class variables
        self.price = price
        self.expenditure = expenditure
        self.budget_share = budget_share
        self.demographic = demographic
        self.price_header = price_header
        self.budget_share_header = budget_share_header
        self.demographic_header = demographic_header
        self.number_observation = number_observation
        self.number_demographic = number_demographic
        self.number_good = number_good
        self.number_coefficient = number_good * (number_good + number_demographic + 2)
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.alpha_demographic = None
        self.estimation_output = None
        self.alpha0 = alpha0
        self.budget_elasticity = np.zeros([self.number_observation, self.number_good])
        self.uncompensated_price_elasticity = np.zeros([self.number_observation, self.number_good, self.number_good])
        self.compensated_price_elasticity = np.zeros([self.number_observation, self.number_good, self.number_good])
        self.slutsky_matrix = np.zeros([self.number_observation, self.number_good, self.number_good])

    def stone_price_index(self):
        a = (self.budget_share * self.price).sum(axis=1, keepdims=True)  # Stone Price Index
        return a

    def coefficient_check(self, alpha, beta, gamma, alpha_demographic=None):
        # Perform dimension check for input variables
        if alpha.shape != (self.number_good, 1):
            alpha = alpha.reshape(self.number_good, 1)
        if beta.shape != (self.number_good, 1):
            beta = beta.reshape(self.number_good, 1)
        if gamma.shape != (self.number_good, self.number_good):
            raise IndexError(
                'Shape of gamma should be ({}, {}) but it is {}.'.format(self.number_good, self.number_good,
                                                                         gamma.shape))
        if alpha_demographic is not None:
            if alpha_demographic.shape != (self.number_good, self.number_demographic):
                raise IndexError('Shape of alpha_demographic should be ({}, {}) but shape of the input is {}.'.format(
                    self.number_good, self.number_demographic,
                    alpha_demographic.shape))
        return alpha, beta, gamma, alpha_demographic

    def price_index(self):
        #  Calculate lna
        def cross_price_sum(vec_price):
            vec_price = vec_price.reshape(self.number_good, 1)
            return 0.5 * (self.gamma * vec_price.dot(vec_price.T)).sum()

        lna = self.alpha0 + self.price.dot(self.alpha) + \
              np.apply_along_axis(cross_price_sum, 1, self.price).reshape(self.number_observation, 1)

        if self.number_demographic != 0:
            def demographic_price_sum(vec_data):
                vec_price, vec_demographic = np.hsplit(vec_data, [self.number_good, self.number_good +
                                                                  self.number_demographic])[:2]
                vec_price = vec_price.reshape(self.number_good, 1)
                vec_demographic = vec_demographic.reshape(self.number_demographic, 1)
                return (self.alpha_demographic.dot(vec_demographic) * vec_price).sum()

            lna += np.apply_along_axis(demographic_price_sum, 1, np.append(self.price, self.demographic, axis=1)) \
                .reshape(self.number_observation, 1)

        # Calculate b
        b = np.exp(self.price.dot(self.beta))

        return lna, b

    def predict(self):
        lna = self.price_index()[0]  # Get only lna, not b
        predicted_budget_share = np.zeros([self.number_observation, self.number_good])
        for i in range(self.number_good - 1):
            predicted_budget_share[:, i] = (self.alpha[i] +
                                            (self.gamma[i, :] * self.price).sum(axis=1, keepdims=True) +
                                            self.beta[i] * (self.expenditure - lna)
                                            ).flatten()
        if self.number_demographic != 0:
            for i in range(self.number_good - 1):
                predicted_budget_share[:, i] += ((self.alpha_demographic[i, :] *
                                                  self.demographic).sum(axis=1, keepdims=True)).flatten()
        # Calculate last commodity budget share
        predicted_budget_share[:, self.number_good - 1] = (1 - predicted_budget_share[:, :self.number_good - 1]
                                                           .sum(axis=1, keepdims=True)).flatten()

        return predicted_budget_share

    def create_equations(self):
        equations = {}
        functional_input = '1'
        for good in self.price_header:
            functional_input += ' + ' + str(good)
        functional_input = functional_input + ' + total_expenditure'
        if self.number_demographic is not 0:
            for demographic in self.demographic_header:
                functional_input += ' + ' + str(demographic)
        for equation_id in range(self.number_good - 1):
            equations['eq' + str(equation_id)] = self.budget_share_header[equation_id] + ' ~ ' + functional_input
        return equations

    def sur_to_np(self, parameters):
        # Initialize arrays
        alpha = np.zeros([self.number_good, 1])
        beta = np.zeros([self.number_good, 1])
        gamma = np.zeros([self.number_good, self.number_good])
        if self.number_demographic is not 0:
            alpha_demographic = np.zeros([self.number_good, self.number_demographic])

        # Fill in Arrays
        for i in np.arange(self.number_good - 1):
            key_alpha = 'eq' + str(i) + '_Intercept'
            key_beta = 'eq' + str(i) + '_total_expenditure'
            for j in np.arange(self.number_good):
                key_gamma = 'eq' + str(i) + '_' + self.price_header[j]
                gamma[i, j] = parameters[key_gamma]
            if self.number_demographic is not 0:
                for k in np.arange(self.number_demographic):
                    key_alpha_demographic = 'eq' + str(i) + '_' + self.demographic_header[k]
                    alpha_demographic[i, k] = parameters[key_alpha_demographic]
            alpha[i] = parameters[key_alpha]
            beta[i] = parameters[key_beta]
        alpha[self.number_good - 1] = 1 - alpha.sum()
        beta[self.number_good - 1] = -beta.sum()
        gamma[-1, :] = -gamma[:-1, :].sum(axis=0)
        if self.number_demographic is not 0:
            alpha_demographic[-1, :] = -alpha_demographic[:-1, :].sum(axis=0)

        if self.number_demographic is 0:
            return alpha, beta, gamma
        else:
            return alpha, beta, gamma, alpha_demographic

    def create_restriction_matrix(self, parameters):
        number_exogeneous_var = self.number_good + self.number_demographic + 2  # number of exogeneous variables per equation. +3 for the intercept, expenditure.
        number_parameter = (self.number_good - 1) * number_exogeneous_var
        number_homogeneity_restriction = self.number_good - 1
        r = np.zeros([number_homogeneity_restriction, number_parameter])
        index_gamma = np.zeros([self.number_good - 1, self.number_good], dtype=object)
        for function_id in np.arange(self.number_good - 1):
            index_gamma[function_id, :] = ('eq' + str(function_id) + '_' + self.price_header).values
            for good_id in np.arange(self.number_good):
                r[function_id, parameters.index.get_loc(index_gamma[function_id, good_id])] = 1
        for function_id in np.arange(self.number_good - 1):
            for good_id in np.arange(function_id, self.number_good - 2):
                r_add = np.zeros([1, number_parameter])
                r_add[0, parameters.index.get_loc(index_gamma[function_id, good_id + 1])] = 1
                r_add[0, parameters.index.get_loc(index_gamma.T[function_id, good_id + 1])] = -1
                r = np.vstack([r, r_add])
        return pd.DataFrame(r)

    def optimize(self, iter_limit=500, tol=1e-6, print_iter=False, return_output=False):
        # Create SUR Model
        equations = self.create_equations()
        # Initialize
        lna = self.stone_price_index()

        # Create Adjusted Inputs
        adj_expenditure = self.expenditure - lna
        if self.number_demographic is 0:
            x = np.append(self.price, adj_expenditure, axis=1)
            data = pd.DataFrame(x, columns=np.append(self.price_header, 'total_expenditure'))
        else:
            x = np.append(np.append(self.price, adj_expenditure, axis=1), self.demographic, axis=1)
            data = pd.DataFrame(x, columns=np.append(
                np.append(self.price_header, 'total_expenditure')
                , self.demographic_header))
        data = pd.concat([data, pd.DataFrame(self.budget_share, columns=self.budget_share_header)], axis=1)

        # Initial SUR
        model = lms.SUR.from_formula(equations, data)
        output = model.fit(method='gls', debiased=True, iter_limit=iter_limit, tol=tol)
        parameters = output.params
        if self.number_demographic is 0:
            self.alpha, self.beta, self.gamma = self.sur_to_np(parameters)
            lna_new = self.price_index()[0]
        else:
            self.alpha, self.beta, self.gamma, self.alpha_demographic = self.sur_to_np(parameters)
            lna_new = self.price_index()[0]

        # Restrictions
        r = self.create_restriction_matrix(parameters)

        # AIDS Loop
        iteration = 0
        tolerance = 10e5  # A large number
        while iteration <= iter_limit and tolerance >= tol:
            lna_old = lna_new
            alpha_old = self.alpha
            beta_old = self.beta
            gamma_old = self.gamma
            if self.number_demographic != 0:
                alpha_demographic_old = self.alpha_demographic
            data['total_expenditure'] = self.expenditure - lna_old
            model = lms.SUR.from_formula(equations, data)
            model.add_constraints(r)
            output = model.fit(method='gls', debiased=True, iter_limit=iter_limit)
            parameters = output.params
            if self.number_demographic is 0:
                self.alpha, self.beta, self.gamma = self.sur_to_np(parameters)
                lna_new = self.price_index()[0]
                tolerance = np.array([np.abs(lna_new - lna_old).max(),
                                      np.abs(self.alpha - alpha_old).max(),
                                      np.abs(self.beta - beta_old).max(),
                                      np.abs(self.gamma - gamma_old).max()]).max()
            else:
                self.alpha, self.beta, self.gamma, self.alpha_demographic = self.sur_to_np(parameters)
                lna_new = self.price_index()[0]
                tolerance = np.array([np.abs(lna_new - lna_old).max(),
                                      np.abs(self.alpha - alpha_old).max(),
                                      np.abs(self.beta - beta_old).max(),
                                      np.abs(self.gamma - gamma_old).max(),
                                      np.abs(self.alpha_demographic - alpha_demographic_old).max()]).max()
            if print_iter is True:
                print("Iteration: {}: Tol:{}".format(iteration + 1, tolerance))
            iteration = iteration + 1
        self.estimation_output = output
        if return_output is True:
            return self.estimation_output
        if tolerance >= tol:
            print('Convergence is not achieved. Re-optimization with a larger iter_limit\n'
                  'or a lower tol is recommended.')

    def external_coefficient(self, alpha=None, beta=None, gamma=None, alpha_demographic=None, alpha0=None):
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma
        if alpha_demographic is not None:
            self.alpha_demographic = alpha_demographic
        if alpha0 is not None:
            self.alpha0 = alpha0

    def elasticity(self):
        price = np.array(self.price).reshape([self.number_observation, self.number_good])
        expenditure = np.array(self.expenditure).reshape([self.number_observation, 1])
        x = np.concatenate((price, expenditure), axis=1)
        if self.number_demographic != 0:
            demographic = self.demographic.reshape([self.number_observation, self.number_demographic])
            x = np.concatenate((x, demographic), axis=1)
        y = self.predict()

        # For all Observations mu_i:
        mu_i = self.beta

        # Observation specific mu_ij:
        def gen_mu_ij(input_row):
            obs_price = input_row.flatten()[:self.number_good]
            if self.number_demographic == 0:
                mu_ij = self.gamma - mu_i * (self.alpha + (self.gamma * obs_price).sum(axis=1, keepdims=True)).T
            else:
                obs_demographic = input_row.flatten()[self.number_good+1:]
                mu_ij = self.gamma - mu_i * ((self.alpha_demographic * obs_demographic).sum(axis=1, keepdims=True) +
                                             (self.alpha + (self.gamma * obs_price).sum(axis=1, keepdims=True))).T
            return mu_ij
        mu_ij_3d = np.apply_along_axis(gen_mu_ij, axis=1, arr=x)

        # Calculate Elasticities
        self.budget_elasticity = mu_i.T/y + 1
        for i in range(self.number_observation):
            self.uncompensated_price_elasticity[i, :, :] = mu_ij_3d[i, :, :]/y[i, :].reshape(self.number_good, 1) - \
                                                      np.diag(np.ones(self.number_good))
            self.compensated_price_elasticity[i, :, :] = self.uncompensated_price_elasticity[i, :, :] + \
                self.budget_elasticity[i, :].reshape(self.number_good, 1)*y[i, :]
            self.slutsky_matrix[i, :, :] = y[i, :].reshape(self.number_good, 1) * \
                self.compensated_price_elasticity[i, :, :]


class Quaids:

    def __init__(self, price, expenditure, budget_share, demographic=None,
                 price_log=True, expenditure_log=True, alpha0=0):
        if price_log is False:
            price = np.log(price)  # Turn price data into logs.
        if expenditure_log is False:
            expenditure = np.log(expenditure)  # Turn expenditure data into logs.

        number_observation, number_good = price.shape
        if demographic is None:
            number_demographic = 0
            demographic_header = []
        else:
            number_demographic = demographic.shape[1]  # Number of columns of demographics input data.
            demographic_header = demographic.columns
            demographic = np.array(demographic, ndmin=2)

        try:
            price_header = price.columns  # If input is pandas data frame.
        except AttributeError:
            price_header = []  # If input is numpy array.
        price = np.array(price, ndmin=2)
        expenditure = np.array(expenditure, ndmin=2).reshape(number_observation, 1)
        try:
            budget_share_header = budget_share.columns
        except AttributeError:
            budget_share_header = []
        budget_share = np.array(budget_share, ndmin=2)

        # Assign class variables
        self.price = price
        self.expenditure = expenditure
        self.budget_share = budget_share
        self.demographic = demographic
        self.price_header = price_header
        self.budget_share_header = budget_share_header
        self.demographic_header = demographic_header
        self.number_observation = number_observation
        self.number_demographic = number_demographic
        self.number_good = number_good
        self.number_coefficient = number_good * (number_good + number_demographic + 2)
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.lambdas = None
        self.alpha_demographic = None
        self.estimation_output = None
        self.alpha0 = alpha0
        self.budget_elasticity = np.zeros([self.number_observation, self.number_good])
        self.uncompensated_price_elasticity = np.zeros([self.number_observation, self.number_good, self.number_good])
        self.compensated_price_elasticity = np.zeros([self.number_observation, self.number_good, self.number_good])
        self.slutsky_matrix = np.zeros([self.number_observation, self.number_good, self.number_good])

    def stone_price_index(self):
        a = (self.budget_share * self.price).sum(axis=1, keepdims=True)  # Stone Price Index
        return a

    def coefficient_check(self, alpha, beta, gamma, lambdas, alpha_demographic=None):
        # Perform dimension check for input variables
        if alpha.shape != (self.number_good, 1):
            alpha = alpha.reshape(self.number_good, 1)
        if beta.shape != (self.number_good, 1):
            beta = beta.reshape(self.number_good, 1)
        if gamma.shape != (self.number_good, self.number_good):
            raise IndexError(
                'Shape of gamma should be ({}, {}) but it is {}.'.format(self.number_good, self.number_good,
                                                                         gamma.shape))
        if lambdas.shape != (self.number_good, 1):
            lambdas = lambdas.reshape(self.number_good, 1)

        if alpha_demographic is not None:
            if alpha_demographic.shape != (self.number_good, self.number_demographic):
                raise IndexError('Shape of alpha_demographic should be ({}, {}) but shape of the input is {}.'.format(
                    self.number_good, self.number_demographic,
                    alpha_demographic.shape))
        return alpha, beta, gamma, lambdas, alpha_demographic

    def price_index(self):
        #  Calculate lna
        def cross_price_sum(vec_price):
            vec_price = vec_price.reshape(self.number_good, 1)
            return 0.5 * (self.gamma * vec_price.dot(vec_price.T)).sum()

        lna = self.alpha0 + self.price.dot(self.alpha) + \
              np.apply_along_axis(cross_price_sum, 1, self.price).reshape(self.number_observation, 1)

        if self.number_demographic != 0:
            def demographic_price_sum(vec_data):
                vec_price, vec_demographic = np.hsplit(vec_data, [self.number_good, self.number_good +
                                                                  self.number_demographic])[:2]
                vec_price = vec_price.reshape(self.number_good, 1)
                vec_demographic = vec_demographic.reshape(self.number_demographic, 1)
                return (self.alpha_demographic.dot(vec_demographic) * vec_price).sum()

            lna += np.apply_along_axis(demographic_price_sum, 1, np.append(self.price, self.demographic, axis=1)) \
                .reshape(self.number_observation, 1)

        # Calculate b
        b = np.exp(self.price.dot(self.beta))

        return lna, b

    def predict(self):
        lna, b = self.price_index()
        predicted_budget_share = np.zeros([self.number_observation, self.number_good])
        for i in range(self.number_good - 1):
            predicted_budget_share[:, i] = (self.alpha[i] +
                                            (self.gamma[i, :] * self.price).sum(axis=1, keepdims=True) +
                                            self.beta[i] * (self.expenditure - lna) +
                                            self.lambdas[i] * ((self.expenditure - lna)**2)/b
                                            ).flatten()
        if self.number_demographic != 0:
            for i in range(self.number_good - 1):
                predicted_budget_share[:, i] += ((self.alpha_demographic[i, :] *
                                                  self.demographic).sum(axis=1, keepdims=True)).flatten()
        # Calculate last commodity budget share
        predicted_budget_share[:, self.number_good - 1] = (1 - predicted_budget_share[:, :self.number_good - 1]
                                                           .sum(axis=1, keepdims=True)).flatten()

        return predicted_budget_share

    def create_equations(self):
        equations = {}
        functional_input = '1'
        for good in self.price_header:
            functional_input += ' + ' + str(good)
        functional_input = functional_input + ' + total_expenditure + total_expenditure_sq'
        if self.number_demographic is not 0:
            for demographic in self.demographic_header:
                functional_input += ' + ' + str(demographic)
        for equation_id in range(self.number_good - 1):
            equations['eq' + str(equation_id)] = self.budget_share_header[equation_id] + ' ~ ' + functional_input
        return equations

    def sur_to_np(self, parameters):
        # Initialize arrays
        alpha = np.zeros([self.number_good, 1])
        beta = np.zeros([self.number_good, 1])
        lambdas = np.zeros([self.number_good, 1])
        gamma = np.zeros([self.number_good, self.number_good])
        if self.number_demographic is not 0:
            alpha_demographic = np.zeros([self.number_good, self.number_demographic])

        # Fill in Arrays
        for i in np.arange(self.number_good - 1):
            key_alpha = 'eq' + str(i) + '_Intercept'
            key_beta = 'eq' + str(i) + '_total_expenditure'
            key_lambdas = 'eq' + str(i) + '_total_expenditure_sq'
            for j in np.arange(self.number_good):
                key_gamma = 'eq' + str(i) + '_' + self.price_header[j]
                gamma[i, j] = parameters[key_gamma]
            if self.number_demographic is not 0:
                for k in np.arange(self.number_demographic):
                    key_alpha_demographic = 'eq' + str(i) + '_' + self.demographic_header[k]
                    alpha_demographic[i, k] = parameters[key_alpha_demographic]
            alpha[i] = parameters[key_alpha]
            beta[i] = parameters[key_beta]
            lambdas[i] = parameters[key_lambdas]
        alpha[self.number_good - 1] = 1 - alpha.sum()
        beta[self.number_good - 1] = -beta.sum()
        lambdas[self.number_good - 1] = -lambdas.sum()
        gamma[-1, :] = -gamma[:-1, :].sum(axis=0)
        if self.number_demographic is not 0:
            alpha_demographic[-1, :] = -alpha_demographic[:-1, :].sum(axis=0)

        if self.number_demographic is 0:
            return alpha, beta, gamma, lambdas
        else:
            return alpha, beta, gamma, lambdas, alpha_demographic

    def create_restriction_matrix(self, parameters):
        number_exogeneous_var = self.number_good + self.number_demographic + 3  # number of exogeneous variables per equation. +3 for the intercept, expenditure. expenditure squared.
        number_parameter = (self.number_good - 1) * number_exogeneous_var
        number_homogeneity_restriction = self.number_good - 1
        r = np.zeros([number_homogeneity_restriction, number_parameter])
        index_gamma = np.zeros([self.number_good - 1, self.number_good], dtype=object)
        for function_id in np.arange(self.number_good - 1):
            index_gamma[function_id, :] = ('eq' + str(function_id) + '_' + self.price_header).values
            for good_id in np.arange(self.number_good):
                r[function_id, parameters.index.get_loc(index_gamma[function_id, good_id])] = 1
        for function_id in np.arange(self.number_good - 1):
            for good_id in np.arange(function_id, self.number_good - 2):
                r_add = np.zeros([1, number_parameter])
                r_add[0, parameters.index.get_loc(index_gamma[function_id, good_id + 1])] = 1
                r_add[0, parameters.index.get_loc(index_gamma.T[function_id, good_id + 1])] = -1
                r = np.vstack([r, r_add])
        return pd.DataFrame(r)

    def optimize(self, iter_limit=1000, tol=1e-8, print_iter=False, return_output=False):
        # Create SUR Model
        equations = self.create_equations()
        # Initialize
        lna = self.stone_price_index()
        b = np.ones(self.number_observation).reshape(self.number_observation, 1)
        # Create Adjusted Inputs
        adj_expenditure = self.expenditure - lna
        adj_expenditure_sq = adj_expenditure**2/b
        if self.number_demographic is 0:
            x = np.append(np.append(self.price, adj_expenditure, axis=1), adj_expenditure_sq, axis=1)
            data = pd.DataFrame(x, columns=np.append(np.append(self.price_header, 'total_expenditure'),
                                                     'total_expenditure_sq'))
        else:
            x = np.append(np.append(np.append(self.price, adj_expenditure, axis=1), adj_expenditure_sq, axis=1),
                          self.demographic, axis=1)
            data = pd.DataFrame(x, columns=np.append(np.append(np.append(self.price_header, 'total_expenditure'),
                                                     'total_expenditure_sq'), self.demographic_header))
        data = pd.concat([data, pd.DataFrame(self.budget_share, columns=self.budget_share_header)], axis=1)


        # Initial SUR
        model = lms.SUR.from_formula(equations, data)
        output = model.fit(method='gls', debiased=True, iter_limit=iter_limit, tol=tol)
        parameters = output.params
        if self.number_demographic is 0:
            self.alpha, self.beta, self.gamma, self.lambdas = self.sur_to_np(parameters)
            lna_new, b_new = self.price_index()
        else:
            self.alpha, self.beta, self.gamma, self.lambdas, self.alpha_demographic = self.sur_to_np(parameters)
            lna_new, b_new = self.price_index()


        # Restrictions
        r = self.create_restriction_matrix(parameters)


        # QUAIDS Loop
        iteration = 0
        tolerance = 10e5  # A large number
        while iteration <= iter_limit and tolerance >= tol:
            lna_old = lna_new
            b_old = b_new
            alpha_old = self.alpha
            beta_old = self.beta
            gamma_old = self.gamma
            lambdas_old = self.lambdas
            if self.number_demographic != 0:
                alpha_demographic_old = self.alpha_demographic
            data['total_expenditure'] = self.expenditure - lna_old
            data['total_expenditure_sq'] = data['total_expenditure']**2 / b_old.flatten()
            model = lms.SUR.from_formula(equations, data)
            model.add_constraints(r)
            output = model.fit(method='gls', debiased=True, iter_limit=iter_limit)
            parameters = output.params
            if self.number_demographic == 0:
                self.alpha, self.beta, self.gamma, self.lambdas = self.sur_to_np(parameters)
                lna_new, b_new = self.price_index()
                tolerance = np.array([np.abs(lna_new - lna_old).max(),
                                      np.abs(b_new - b_old).max(),
                                      np.abs(self.alpha - alpha_old).max(),
                                      np.abs(self.beta - beta_old).max(),
                                      np.abs(self.lambdas - lambdas_old).max(),
                                      np.abs(self.gamma - gamma_old).max()]).max()
            else:
                self.alpha, self.beta, self.gamma, self.lambdas, self.alpha_demographic = self.sur_to_np(parameters)
                lna_new, b_new = self.price_index()
                tolerance = np.array([np.abs(lna_new - lna_old).max(),
                                      np.abs(b_new - b_old).max(),
                                      np.abs(self.alpha - alpha_old).max(),
                                      np.abs(self.beta - beta_old).max(),
                                      np.abs(self.gamma - gamma_old).max(),
                                      np.abs(self.lambdas - lambdas_old).max(),
                                      np.abs(self.alpha_demographic - alpha_demographic_old).max()]).max()
            if print_iter is True:
                print("Iteration: {}: Tol:{}".format(iteration + 1, tolerance))
            iteration = iteration + 1
        self.estimation_output = output
        if return_output is True:
            return self.estimation_output
        if tolerance >= tol:
            print('Convergence is not achieved. Re-optimization with a larger iter_limit\n'
                  'or a lower tol is recommended.')

    def external_coefficient(self, alpha=None, beta=None, gamma=None, lambdas=None,
                             alpha_demographic=None, alpha0=None):
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma
        if lambdas is not None:
            self.lambdas = lambdas
        if alpha_demographic is not None:
            self.alpha_demographic = alpha_demographic
        if alpha0 is not None:
            self.alpha0 = alpha0

    def elasticity(self):
        price = np.array(self.price).reshape([self.number_observation, self.number_good])
        expenditure = np.array(self.expenditure).reshape([self.number_observation, 1])
        x = np.concatenate((price, expenditure), axis=1)
        if self.number_demographic != 0:
            demographic = self.demographic.reshape([self.number_observation, self.number_demographic])
            x = np.concatenate((x, demographic), axis=1)
        y = self.predict()

        # Observation specific mu_i:
        lna, b = self.price_index()
        x = np.concatenate((x, lna, b), axis=1)

        def gen_mu(input_row):
            input_row = input_row.flatten()
            input_price = input_row[:self.number_good]
            input_expenditure = input_row[self.number_good]
            input_lna = input_row[-2]
            input_b = input_row[-1]
            mu_i = self.beta + 2*self.lambdas*(input_expenditure - input_lna)/input_b
            if self.number_demographic == 0:
                mu_ij = self.gamma - mu_i * (self.alpha + (self.gamma * input_price).sum(axis=1, keepdims=True)).T - \
                        self.lambdas*self.beta.T*((input_expenditure-input_lna)**2)/input_b
            else:
                obs_demographic = input_row[self.number_good+1:-2]
                mu_ij = self.gamma - mu_i * ((self.alpha_demographic * obs_demographic).sum(axis=1, keepdims=True) +
                                             (self.alpha + (self.gamma * input_price).sum(axis=1, keepdims=True))).T - \
                    self.lambdas * self.beta.T * ((input_expenditure - input_lna) ** 2) / input_b
            return mu_i, mu_ij
        mu_i_3d = np.zeros([self.number_observation, self.number_good, 1])
        mu_ij_3d = np.zeros([self.number_observation, self.number_good, self.number_good])

        # Calculate Elasticities
        for i in range(self.number_observation):
            mu_i_3d[i], mu_ij_3d[i] = gen_mu(x[i:i+1])
            self.budget_elasticity[i, :] = mu_i_3d[i, :, :].T/y[i, :] + 1
            self.uncompensated_price_elasticity[i, :, :] = mu_ij_3d[i, :, :]/y[i, :].reshape(self.number_good, 1) - \
                                                            np.diag(np.ones(self.number_good))
            self.compensated_price_elasticity[i, :, :] = self.uncompensated_price_elasticity[i, :, :] + \
                self.budget_elasticity[i, :].reshape(self.number_good, 1)*y[i, :]
            self.slutsky_matrix[i, :, :] = y[i, :].reshape(self.number_good, 1) * \
                self.compensated_price_elasticity[i, :, :]


def network_model(input_node, hl_node, output_node, hl_transformation, bias=True):
    if isinstance(hl_node, int):
        hl = 1
    elif isinstance(hl_node, tuple):
        hl = hl_node.__len__()
    else:
        raise TypeError("number_hl_node must be an int or a tuple of integers but it is {}".
                        format(type(hl_node)))

    if isinstance(bias, bool):
        bias_list = np.repeat(bias, hl+1)
    elif isinstance(bias, tuple):
        if all(isinstance(n, bool) for n in bias):
            bias_list = np.array(bias)
        else:
            raise TypeError("bias must be a boolean or a tuple of booleans. None boolean entry detected!!!")

    model = collections.OrderedDict()
    layer = 0
    model['hl'+str(layer+1)] = (nn.Linear(input_node, hl_node, bias=bias_list[0]) if hl == 1 else
                                nn.Linear(input_node, hl_node[layer], bias=bias_list[layer]))
    model['hl'+str(layer+1)+' transformation'] = hl_transformation
    if hl == 1:
        model['ol'] = nn.Linear(hl_node, output_node, bias_list[hl])
    else:
        while layer < hl-1:
            layer = layer + 1
            model['hl'+str(layer+1)] = nn.Linear(hl_node[layer-1], hl_node[layer], bias=bias_list[layer])
            model['hl'+str(layer+1)+' transformation'] = hl_transformation
        model['ol'] = nn.Linear(hl_node[-1], output_node, bias=bias_list[-1])
    model['ol transformation'] = nn.Softmax(dim=1)
    return nn.Sequential(model)


class DemandDataset(utils_data.Dataset):
    """Designed for a sample with log-prices, log-expenditure, and demographics. All transformation has to be applied
    beforehand. This class reads from an external csv file."""

    def __init__(self, price, expenditure, budget_share, demographics=None):
        self.x = pd.concat([price, expenditure], axis=1)
        if demographics is not None:
            self.x = pd.concat([self.x, demographics], axis=1)
        self.x = torch.tensor(np.array(self.x), dtype=torch.float32)
        self.y = torch.tensor(np.array(budget_share), dtype=torch.float32)

        if len(self.x) != len(self.y):
            raise ValueError("x and y have different numbers of observations.")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x = self.x[item, :]
        y = self.y[item, :]
        data = {'x': x, 'y': y}
        return data
