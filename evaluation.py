import numpy as np

def sampling_zeros(train, test, synthetic):
    """Count the combinations of variables from the synthetic data which 
    are in the test set but not in the training set.

    Reference: https://arxiv.org/pdf/1909.07689.pdf

    Parameters
    ----------
    train : pd.DataFrame
        The training set used to create the synthetic data 

    test : pd.DataFrame
        The testing set not seen by the generative model

    synthetic : pd.DataFrame 
        The synthetic data sampled by a generative model trained on the training set

    Returns 
    -------
    count : int
        The number of sampling zeros
    """
    train_set = set(tuple(i) for i in train.to_numpy())
    test_set = set(tuple(i) for i in test.to_numpy())
    synthetic_set = set(tuple(i) for i in synthetic.to_numpy())

    sampling_zeros = synthetic_set.intersection(test_set) - train_set
    count = len(sampling_zeros)
    return count


def srmse(data1, data2):
    """ Compute Standardized Root Mean Squared Error between two datasets.

    Reference: https://www.researchgate.net/publication/282815687_A_Bayesian_network_approach_for_population_synthesis


    Parameters
    ----------
    data1 : pd.DataFrame
        Dataset for which to compute SRMSE

    data2 : pd.DataFrame
        Dataset for which to compute SRMSE 

    Returns
    -------
    SRMSE : float
        Standardized Root Mean Squared Error between data1 and data2
    """
    columns = list(data1.columns.values)
    # Relative frequency
    data1_f = data1.value_counts(normalize=True)
    data2_f = data2.value_counts(normalize=True)
    # Total numbers of categories
    Mi = [data1_f.index.get_level_values(l).union(data2_f.index.get_level_values(l)).unique().size for l in range(len(columns))]
    M = np.prod(Mi)
    # SRMSE
    SRMSE = ((data1_f.subtract(data2_f, fill_value=0)**2) * M).sum()**(.5)
    return SRMSE