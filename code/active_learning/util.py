import random

import numpy as np
import sklearn
import sklearn.linear_model
import sklearn.model_selection
import sklearn.metrics

'''
Base class for query selectors. The select_query_idx function should be overrided in subclasses.
'''
class QuerySelector(object):
    def select_query_idx(self, X, y, observed_idxs, unobserved_idxs_set, current_model):
        pass

    # Called at the start of each individual run to reset QuerySelector object (in case the object contains any attributes that should be reset between runs)
    def reset(self):
        pass

'''
RandomQuerySelector just samples randomly from the unobserved instances, as in HW0.
'''
class RandomQuerySelector(QuerySelector):
    def select_query_idx(self, X, y, observed_idxs_list, unobserved_idxs_set, current_model):
        return random.choice(list(unobserved_idxs_set))


'''
UncertaintySamplingQuerySelector does uncertainty sampling, where uncertainty is measured by the estimated variance of the prediction, as discussed in the course Piazza.
'''
class UncertaintySamplingQuerySelector(QuerySelector):
    def select_query_idx(self, X, y, observed_idxs_list, unobserved_idxs_set, current_model):
        N = X.shape[0]
        N_observed = len(observed_idxs_list)
        N_unobserved = len(unobserved_idxs_set)

        X_observed = np.take(X, observed_idxs_list, axis=0)
        y_observed = np.take(y, observed_idxs_list, axis=0)
        X_unobserved = np.delete(X, observed_idxs_list, axis=0)
        y_unobserved = np.delete(y, observed_idxs_list, axis=0)
        idxs_unobserved = np.delete(np.arange(N), observed_idxs_list, axis=0) # denotes the original row numbers of the instances in X_unobserved and y_unobserved, in X and y.

        y_observed_pred = current_model.predict(X_observed)
        SSE = np.sum((y_observed - y_observed_pred)**2) # SSE is sum squared error on observed (training) instances
        s_squared = SSE / (N_observed - 2) # s_squared = estimated variance (of random variable eps in y = a1x1 + a2x2 + ... + b + eps). Note that N_unobserved must be >= 3, which we assume here.

        X_observed_mean = np.mean(X_observed, axis=0)
        S_xx = np.sum((X_observed - X_observed_mean)**2) # think of this as taking sum of squares of feature differences from mean for each instance (euclidean distance squared), then summing over all instances.

        # using formula from https://www.colorado.edu/amath/sites/default/files/attached-files/ch12_0.pdf page 64, as suggested by an instructor in the course Piazza (https://piazza.com/class/kyk8pt9b2yn6o4?cid=9_f3)
        tmp = np.sum((X_unobserved - X_observed_mean)**2, axis=1)
        prediction_variances = s_squared * ((1.0/N_observed) + (tmp/S_xx))

        least_certain_prediction_idx_unobserved = np.argmax(prediction_variances) # Find instance with lowest prediction probability/confidence
        least_certain_prediction_idx = idxs_unobserved[least_certain_prediction_idx_unobserved] # convert back to original instance idx

        return least_certain_prediction_idx


'''
UncertaintySamplingWithDensityQuerySelector is like UncertaintySamplingQuerySelector, but with the additional consideration of density.
The similarity function used here (to determine how similar any two instances are) is based on euclidean distance.
'''
class UncertaintySamplingWithDensityQuerySelector(QuerySelector):
    def __init__(self, pairwise_distances, beta=1):
        self.pairwise_distances = pairwise_distances # Pairwise distances for the X in question. Should be N x N. This stays constant (precomputed).
        self.beta = beta

        self.reset()

    def reset(self):
        self.is_first_selection = True
        self.distance_sums_all = np.sum(self.pairwise_distances, axis=1)

    def select_query_idx(self, X, y, observed_idxs_list, unobserved_idxs_set, current_model):
        beta = self.beta
        N = X.shape[0]
        N_observed = len(observed_idxs_list)
        N_unobserved = len(unobserved_idxs_set)

        assert(N == self.pairwise_distances.shape[0])
        assert(N == self.pairwise_distances.shape[1])

        X_observed = np.take(X, observed_idxs_list, axis=0)
        y_observed = np.take(y, observed_idxs_list, axis=0)
        X_unobserved = np.delete(X, observed_idxs_list, axis=0)
        y_unobserved = np.delete(y, observed_idxs_list, axis=0)
        idxs_unobserved = np.delete(np.arange(N), observed_idxs_list, axis=0) # denotes the original row numbers of the instances in X_unobserved and y_unobserved, in X and y.

        y_observed_pred = current_model.predict(X_observed)
        SSE = np.sum((y_observed - y_observed_pred) ** 2) # SSE is sum squared error on observed (training) instances
        s_squared = SSE / (N_observed - 2) # s_squared = estimated variance (of random variable eps in y = a1x1 + a2x2 + ... + b + eps). Note that N_unobserved must be >= 3, which we assume here.

        X_observed_mean = np.mean(X_observed, axis=0)
        S_xx = np.sum((X_observed - X_observed_mean)**2) # think of this as taking sum of squares of feature differences from mean for each instance (euclidean distance squared), then summing over all instances.

        # using formula from https://www.colorado.edu/amath/sites/default/files/attached-files/ch12_0.pdf page 64, as suggested by an instructor in the course Piazza (https://piazza.com/class/kyk8pt9b2yn6o4?cid=9_f3)
        tmp = np.sum((X_unobserved - X_observed_mean)**2, axis=1)
        prediction_variances = s_squared * ((1.0/N_observed) + (tmp/S_xx))

        prediction_variances_normalized = prediction_variances / np.amax(prediction_variances)

        unobserved_instance_vector_magnitudes = np.linalg.norm(X_unobserved, axis=1) # 2-norm by default # np.sum(np.power(X_unobserved, 2), axis=1)
        furthest_possible_distance = np.max(unobserved_instance_vector_magnitudes) * 2
        largest_possible_distance_sum = furthest_possible_distance * N_unobserved

        # Old version of density calculation (no precomputation, and python for loop, so it is quite slow)
        '''
        # Would be nice if this could be vectorized further to avoid the for loop, but I think this is fine for now
        densities = np.zeros((N_unobserved,))
        for i in range(N_unobserved):
            diffs = X_unobserved - X_unobserved[i, :] # uses numpy broadcasting to get result of subtracting row i from every row in X_unobserved
            distances = np.linalg.norm(diffs, axis=1) # # 2-norm by default # np.sum(np.power(diffs, 2), axis=1) # euclidean distances
            similarities = (furthest_possible_distance - distances) / furthest_possible_distance # results in similarity scores between 0 and 1, where 1 is most similar and 0 is least similar.
            densities[i] =  np.sum(similarities) / N_unobserved
        '''
        # print("densities (old): {}".format(densities[:10]))

        # New (faster) version of density calculation
        if (self.is_first_selection):
            # Need to take care of starting observations on first selection, for self.distance_sums_all
            for observed_idx in observed_idxs_list:
                self.distance_sums_all -= self.pairwise_distances[observed_idx, :]
                self.distance_sums_all[observed_idx] = np.nan
            self.is_first_selection = False
        distance_sums = np.delete(self.distance_sums_all, observed_idxs_list, axis=0) # select only those that are unobserved
        assert(not np.isnan(distance_sums).any()) # verify that no nans (previously returned/observed indices) are still labeled as unobserved
        similarity_sums = (largest_possible_distance_sum - distance_sums) / furthest_possible_distance
        densities = similarity_sums / N_unobserved
        # print("densities (new): {}".format(densities[:10]))
        # print("")

        scores = prediction_variances_normalized * np.power(densities, beta)

        best_idx_unobserved = np.argmax(scores)
        best_idx = idxs_unobserved[best_idx_unobserved] # convert back to original instance idx

        self.distance_sums_all -= self.pairwise_distances[best_idx, :]
        self.distance_sums_all[best_idx] = np.nan

        return best_idx


'''
QueryByCommitteeQuerySelector uses a straightforward approach of bagging (bootstrap aggregating) and training many models to form a committee, then finding the most disagreed-upon unobserved instance (where disagreement is measured by variance among committee-model predictions).
'''
class QueryByCommitteeQuerySelector(QuerySelector):
    def __init__(self, base_learner_class, bootstrap_sample_size_multiplier=1, bootstrap_model_count=100):
        self.bootstrap_sample_size_multiplier = bootstrap_sample_size_multiplier
        self.bootstrap_model_count = bootstrap_model_count
        self.base_learner_class = base_learner_class

    def select_query_idx(self, X, y, observed_idxs_list, unobserved_idxs_set, current_model):
        N = X.shape[0]
        bootstrap_sample_size_multiplier = self.bootstrap_sample_size_multiplier
        bootstrap_model_count = self.bootstrap_model_count
        base_learner_class = self.base_learner_class

        num_classes = len(np.unique(y))

        X_observed = np.take(X, observed_idxs_list, axis=0)
        y_observed = np.take(y, observed_idxs_list, axis=0)
        X_unobserved = np.delete(X, observed_idxs_list, axis=0)
        y_unobserved = np.delete(y, observed_idxs_list, axis=0)
        idxs_unobserved = np.delete(np.arange(N), observed_idxs_list, axis=0) # denotes the original row numbers of the instances in X_unobserved and y_unobserved, in X and y.

        N_unobserved = X_unobserved.shape[0]
        N_observed = X_observed.shape[0]

        models = [None] * bootstrap_model_count # committee of models
        bootstrap_sample_size = int(round(N_observed * bootstrap_sample_size_multiplier))
        for i in range(bootstrap_model_count):
            bootstrap_sample_idxs = random.choices(np.arange(N_observed), k=bootstrap_sample_size)
            bootstrap_sample_X = np.take(X_observed, bootstrap_sample_idxs, axis=0)
            bootstrap_sample_y = np.take(y_observed, bootstrap_sample_idxs, axis=0)
            reg_model = base_learner_class()
            reg_model.fit(bootstrap_sample_X, bootstrap_sample_y)
            models[i] = reg_model

        preds = np.zeros((bootstrap_model_count, N_unobserved))
        for i in range(bootstrap_model_count):
            preds[i, :] = models[i].predict(X_unobserved)

        # Will consider "disagreement" on a particular instance to be simply the variance in the committee-model predictions for it
        pred_vars = np.var(preds, axis=0)
        best_idx_unobserved = np.argmax(pred_vars)
        best_idx = idxs_unobserved[best_idx_unobserved] # convert back to original instance idx

        return best_idx


'''
Main class for Active Learning simulations (REGRESSION ONLY FOR NOW, NOT CLASSIFICATION)
'''
class ActiveLearningSimulation:
    '''
    Initialize an active learning simulation. Once initialized, the simulation can be run multiple times.

    Inputs:
    X: input samples (same format as would be provided to sklearn classifier/regressor)
    y: input labels (same format as would be provided to sklearn classifier/regressor)
    n_start_observations: number of observations (observed instances) to start with
    observed_set_size_limit: limit on the number of observed instances, after which the main simulation loop will end
    base_learner_class: Class of base_learner (expected to be an sklearn regression class, e.g. sklearn.linear_model.LinearRegression), that can be called with a blank constructor to create a new base learner.
    query_selector: object that is an instance of a subclass of QuerySelector, to select queries at each round of the simulation. This object (if applicable) should be initialized with the same X and y as the ActiveLearningSimulation at hand is.
    '''
    def __init__(self, X, y, n_start_observations=5, observed_set_size_limit=50, base_learner_class=None, query_selector=None, start_observations_idxs=None):
        # Assertions to ensure that the data (X and y) shapes are acceptable:
        assert(len(X.shape) == 2)
        assert(len(y.shape) == 1)
        assert(X.shape[0] == y.shape[0])

        # Set base_learner_class to the default for the simulation mode (LogisticRegression for classification, LinearRegression for regression), if it was left unspecified.
        if (base_learner_class is None):
            base_learner_class = sklearn.linear_model.LinearRegression

        # Use the default query selector (random) by default if another one is not provided.
        if (query_selector is None):
            query_selector = RandomQuerySelector()

        self.X = X
        self.y = y
        self.n_start_observations = n_start_observations
        self.observed_set_size_limit = observed_set_size_limit
        self.base_learner_class = base_learner_class
        self.query_selector = query_selector
        self.start_observations_idxs = start_observations_idxs

    # Helper function for 5-fold cross-validation, using the method suggested in the HW0 assignment description
    def cross_validate_custom(self, selected_idxs_list, n_splits=5):
        X_selected = np.take(self.X, selected_idxs_list, axis=0)
        y_selected = np.take(self.y, selected_idxs_list, axis=0)
        N_selected = y_selected.shape[0]

        kf = sklearn.model_selection.KFold(n_splits=n_splits)
        MSEs = []
        for train_idxs, test_idxs in kf.split(X_selected):
            X_selected_train, X_selected_test = X_selected[train_idxs], X_selected[test_idxs]
            y_selected_train, y_selected_test = y_selected[train_idxs], y_selected[test_idxs]
            reg_model = self.base_learner_class()
            reg_model.fit(X_selected_train, y_selected_train)
            y_selected_test_pred = reg_model.predict(X_selected_test)
            MSE = sklearn.metrics.mean_squared_error(y_selected_test, y_selected_test_pred)
            MSEs.append(MSE)

        cv_mean_MSE = np.mean(MSEs)
        return cv_mean_MSE

    '''
    Main simulation function.

    Inputs:
    random_seed (optional): random seed to provide to random.seed at the beginning of this run.

    Output is a dict, which has the following fields:
    n_rounds: number of rounds of the main simulation loop that were run, and thus, the length of each of the lists below
    observed_instances: list, where the ith element is the number of observed instances for round i
    unobserved_instances: list, where the ith element is the number of unobserved instances for round i
    unobserved_MSEs (ONLY present if classification_mode=="regression"): list, where the ith element is the model MSE for the unobserved instances for round i
    '''
    def run(self, random_seed=None):
        X = self.X
        y = self.y
        n_start_observations = self.n_start_observations
        observed_set_size_limit = self.observed_set_size_limit
        base_learner_class = self.base_learner_class
        query_selector = self.query_selector

        query_selector.reset()

        # N is the total number of instances
        N = X.shape[0]

        # Set random seed
        random.seed(a=random_seed)

        # Initialize output dict, and any subfields (e.g. lists/sets/etc) which require initialization
        output_dict = {}
        output_dict["observed_instances"] = [] # number of observed instances, for each round
        output_dict["unobserved_instances"] = [] # number of unobserved instances, for each round
        output_dict["cv_mean_MSEs"] = [] # cross-validation mean MSE, for each round
        output_dict["observed_MSEs"] = [] # MSE of model on observed (training) instances, for each round
        output_dict["unobserved_MSEs"] = [] # MSE of model (trained on observed instances) on unobserved instances, for each round

        # Initialize observed and unobserved sets
        unobserved_idxs_set = set(list(range(N)))
        observed_idxs_list = []

        # Randomly sample and mark the proper number (n_start_observations) of instances as observed, to start (or load preset start observations)
        start_observations_idxs = None
        if (self.start_observations_idxs is None):
            start_observations_idxs = random.sample(list(unobserved_idxs_set), n_start_observations)
        else:
            start_observations_idxs = self.start_observations_idxs
        for instance_idx in start_observations_idxs:
            unobserved_idxs_set.remove(instance_idx)
            observed_idxs_list.append(instance_idx)
        n_start_observations = len(observed_idxs_list)

        # Main simulation loop (each iteration of this loop is a round of the simulation)
        n_rounds = 0
        while (len(unobserved_idxs_set) > 0 and len(observed_idxs_list) <= observed_set_size_limit):

            N_observed = len(observed_idxs_list)
            N_unobserved = len(unobserved_idxs_set)

            output_dict["observed_instances"].append(N_observed)
            output_dict["unobserved_instances"].append(N_unobserved)

            X_observed = np.take(X, observed_idxs_list, axis=0)
            y_observed = np.take(y, observed_idxs_list, axis=0)
            X_unobserved = np.delete(X, observed_idxs_list, axis=0)
            y_unobserved = np.delete(y, observed_idxs_list, axis=0)

            # Perform cross-validation (could skip if it takes a significant amout of time and is not really necessary)
            '''
            cv_mean_MSE = self.cross_validate_custom(observed_idxs_list)
            output_dict["cv_mean_MSEs"].append(cv_mean_MSE)
            '''

            # Train model on observed instances
            reg_model = base_learner_class()
            reg_model.fit(X_observed, y_observed)

            # Evaluate model on observed (train) instances
            y_observed_pred = reg_model.predict(X_observed)
            observed_MSE = sklearn.metrics.mean_squared_error(y_observed, y_observed_pred)
            output_dict["observed_MSEs"].append(observed_MSE)

            # Evaluate model on unobserved instances
            y_unobserved_pred = reg_model.predict(X_unobserved)
            unobserved_MSE = sklearn.metrics.mean_squared_error(y_unobserved, y_unobserved_pred)
            output_dict["unobserved_MSEs"].append(unobserved_MSE)

            # Observe a new instance, chosen by the query_selector
            new_instance_idx = query_selector.select_query_idx(X, y, observed_idxs_list, unobserved_idxs_set, reg_model)
            unobserved_idxs_set.remove(new_instance_idx)
            observed_idxs_list.append(new_instance_idx)

            n_rounds += 1

        output_dict["n_rounds"] = n_rounds
        return output_dict

    '''
    Helper function to run simulation multiple times, each with a different random seed, as specified by random_seeds.
    Output will be a list output_dicts, where output_dict[i] is the output produced by running self.run(random_seed=random_seeds[i])
    '''
    def run_with_random_seeds(self, random_seeds):
        output_dicts = []
        for seed in random_seeds:
            output_dict = self.run(random_seed=seed)
            output_dicts.append(output_dict)
        return output_dicts

'''
Subclass for MLP Learning simulations (to pass parameters)
'''

class NeuralLearningSimulation(ActiveLearningSimulation):
    def __init__(self, X, y, n_start_observations=5, observed_set_size_limit=50, base_learner_class=sklearn.neural_network.MLPRegressor, query_selector=None, start_observations_idxs=None):
        super().__init__(X, y, n_start_observations=5, observed_set_size_limit=50, base_learner_class=sklearn.neural_network.MLPRegressor, query_selector=None, start_observations_idxs=None)

    def run(self, random_seed=None):
        X = self.X
        y = self.y
        n_start_observations = self.n_start_observations
        observed_set_size_limit = self.observed_set_size_limit
        base_learner_class = self.base_learner_class
        query_selector = self.query_selector

        query_selector.reset()

        # N is the total number of instances
        N = X.shape[0]

        # Set random seed
        random.seed(a=random_seed)

        # Initialize output dict, and any subfields (e.g. lists/sets/etc) which require initialization
        output_dict = {}
        output_dict["observed_instances"] = [] # number of observed instances, for each round
        output_dict["unobserved_instances"] = [] # number of unobserved instances, for each round
        output_dict["cv_mean_MSEs"] = [] # cross-validation mean MSE, for each round
        output_dict["observed_MSEs"] = [] # MSE of model on observed (training) instances, for each round
        output_dict["unobserved_MSEs"] = [] # MSE of model (trained on observed instances) on unobserved instances, for each round

        # Initialize observed and unobserved sets
        unobserved_idxs_set = set(list(range(N)))
        observed_idxs_list = []

        # Randomly sample and mark the proper number (n_start_observations) of instances as observed, to start (or load preset start observations)
        start_observations_idxs = None
        if (self.start_observations_idxs is None):
            start_observations_idxs = random.sample(list(unobserved_idxs_set), n_start_observations)
        else:
            start_observations_idxs = self.start_observations_idxs
        for instance_idx in start_observations_idxs:
            unobserved_idxs_set.remove(instance_idx)
            observed_idxs_list.append(instance_idx)
        n_start_observations = len(observed_idxs_list)

        # Main simulation loop (each iteration of this loop is a round of the simulation)
        n_rounds = 0
        while (len(unobserved_idxs_set) > 0 and len(observed_idxs_list) <= observed_set_size_limit):

            N_observed = len(observed_idxs_list)
            N_unobserved = len(unobserved_idxs_set)

            output_dict["observed_instances"].append(N_observed)
            output_dict["unobserved_instances"].append(N_unobserved)

            X_observed = np.take(X, observed_idxs_list, axis=0)
            y_observed = np.take(y, observed_idxs_list, axis=0)
            X_unobserved = np.delete(X, observed_idxs_list, axis=0)
            y_unobserved = np.delete(y, observed_idxs_list, axis=0)

            # Perform cross-validation (could skip if it takes a significant amout of time and is not really necessary)
            '''
            cv_mean_MSE = self.cross_validate_custom(observed_idxs_list)
            output_dict["cv_mean_MSEs"].append(cv_mean_MSE)
            '''

            # Train model on observed instances
            reg_model = base_learner_class(max_iter = 10000)
            reg_model.fit(X_observed, y_observed)

            # Evaluate model on observed (train) instances
            y_observed_pred = reg_model.predict(X_observed)
            observed_MSE = sklearn.metrics.mean_squared_error(y_observed, y_observed_pred)
            output_dict["observed_MSEs"].append(observed_MSE)

            # Evaluate model on unobserved instances
            y_unobserved_pred = reg_model.predict(X_unobserved)
            unobserved_MSE = sklearn.metrics.mean_squared_error(y_unobserved, y_unobserved_pred)
            output_dict["unobserved_MSEs"].append(unobserved_MSE)

            # Observe a new instance, chosen by the query_selector
            new_instance_idx = query_selector.select_query_idx(X, y, observed_idxs_list, unobserved_idxs_set, reg_model)
            unobserved_idxs_set.remove(new_instance_idx)
            observed_idxs_list.append(new_instance_idx)

            n_rounds += 1

        output_dict["n_rounds"] = n_rounds
        return output_dict

# END ACTIVE LEARNING STUFF, BEGIN DOE STUFF

'''
Runs Fedorov Algorithm to get an approximation of the D-optimal selection.
Note that this assumes that linear regression is the base learner.
'''
class DOE:
    def get_DOE_idxs(X, k, random_seed=None):
        output_dict = {}

        # N is the total number of instances
        N = X.shape[0]
        N_design = k
        N_free = N - N_design

        # Set random seed
        if (random_seed is not None):
            random.seed(a=random_seed)

        # Best (maximum) score per this function is D-optimal
        def get_score(X):
            return np.linalg.det(X.T @ X)

        # Initialize design and free idxs
        free_idxs_set = set(list(range(N)))
        design_idxs_list = []

        # Select starting design indices randomly
        start_design_idxs = random.sample(list(free_idxs_set), N_design)
        for instance_idx in start_design_idxs:
            free_idxs_set.remove(instance_idx)
            design_idxs_list.append(instance_idx)

        # Main Fedorov Algorithm loop
        t = 0
        while (True):

            X_design = np.take(X, design_idxs_list, axis=0)

            current_score = get_score(X_design)

            # print("Score at t={}: {}".format(t, current_score))

            best_j = None
            best_free_idx = None
            best_score = None
            for j in range(N_design):
                for free_idx in free_idxs_set:
                    X_design_tmp = X_design.copy()
                    X_design_tmp[j, :] = X[free_idx, :]
                    score = get_score(X_design_tmp)
                    if (best_score is None or score > best_score):
                        best_j = j
                        best_free_idx = free_idx
                        best_score = score

            if (best_score > current_score):
                free_idxs_set.add(design_idxs_list[best_j])
                free_idxs_set.remove(best_free_idx)
                design_idxs_list[best_j] = best_free_idx
            else:
                break

            t += 1

        X_design_final = np.take(X, design_idxs_list, axis=0)
        score_final = get_score(X_design_final)

        # print("Final score: {}".format(current_score))

        output_dict["idxs"] = design_idxs_list
        output_dict["score"] = score_final
        output_dict["t"] = t

        return output_dict

    def get_DOE_idxs_multiple_trials(X, k, n_trials, random_seed=None):
        # Set random seed
        if (random_seed is not None):
            random.seed(a=random_seed)

        best_output_dict = None
        for i in range(n_trials):
            # print("Starting trial {}".format(i))
            trial_output_dict = DOE.get_DOE_idxs(X, k, random_seed=None)
            if (best_output_dict is None or trial_output_dict["score"] > best_output_dict["score"]):
                best_output_dict = trial_output_dict

        return best_output_dict

    def get_DOE_idxs_with_random_seeds(X, k, n_trials, random_seeds):
        output_dicts = []
        for seed in random_seeds:
            output_dict = DOE.get_DOE_idxs_multiple_trials(X, k, n_trials, random_seed=seed)
            output_dicts.append(output_dict)
        return output_dicts

    def get_DOE_unobserved_accs_from_idxs_dicts(X, y, DOE_idxs_dicts):
        DOE_regression_unobserved_MSEs = []
        for DOE_idxs_dict in DOE_idxs_dicts:
            DOE_idxs = DOE_idxs_dict["idxs"]

            X_design = np.take(X, DOE_idxs, axis=0)
            y_design = np.take(y, DOE_idxs, axis=0)
            X_unobserved = np.delete(X, DOE_idxs, axis=0)
            y_unobserved = np.delete(y, DOE_idxs, axis=0)

            reg_model = sklearn.linear_model.LinearRegression()
            reg_model.fit(X_design, y_design)

            # Test model on unobserved instances
            y_unobserved_pred = reg_model.predict(X_unobserved)
            unobserved_MSE = sklearn.metrics.mean_squared_error(y_unobserved, y_unobserved_pred)
            DOE_regression_unobserved_MSEs.append(unobserved_MSE)

        return DOE_regression_unobserved_MSEs
