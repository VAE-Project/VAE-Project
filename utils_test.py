import torch
import numpy as np
import pandas as pd
import warnings
from scipy.stats import chisquare
from scipy.special import kl_div
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
# from models import Gibbs


# Preprocess

def train_val_test_split(df,
                         test_size=0.3,
                         ignore_pumas=True,
                         data_folder='/content/gdrive/MyDrive/Maryland/data/TrainValTest',
                         save=True,
                         random_state=42,
                         shuffle=True):
    if ignore_pumas:
        df = df.drop(['PUMA', 'ST'], axis=1)

    train, test = train_test_split(df,
                                   test_size=test_size,
                                   random_state=random_state,
                                   shuffle=shuffle)

    train, val = train_test_split(train,
                                  test_size=test_size,
                                  shuffle=False)

    if save:
        train.to_csv(path_or_buf=data_folder+"/train.csv")
        val.to_csv(path_or_buf=data_folder+"/val.csv")
        test.to_csv(path_or_buf=data_folder+"/test.csv")

    return train, val, test


def train_test(df, test_size=0.3, ignore_pumas=True, data_folder='/content/gdrive/MyDrive/Maryland/data/TrainValTest', save=True, random_state=42, shuffle=True):
    if ignore_pumas:
        df = df.drop(['PUMA', 'ST'], axis=1)

    train, test = train_test_split(df,
                                   test_size=test_size,
                                   random_state=random_state,
                                   shuffle=shuffle)
    if save:
        train.to_csv(path_or_buf=data_folder+"/train_only.csv")
        test.to_csv(path_or_buf=data_folder+"/test_only.csv")

    return train, test



# Data Generation
def generate_samples_vae(encoder,decoder, train_df,args):
    Tensor = torch.cuda.FloatTensor if args.device == "cuda" else torch.FloatTensor
    encoder.to(args.device)
    decoder.to(args.device)
    # model needs to be on device before
    z=Tensor(encoder(Tensor(train_df.values)))
    batch_synthetic = decoder(z)
    return np.round(batch_synthetic.cpu().detach().numpy())

def generate_samples(generator, batch_size, args):
    Tensor = torch.cuda.FloatTensor if args.device == "cuda" else torch.FloatTensor
    generator.to(args.device)
    # model needs to be on device before
    z = Tensor(np.random.normal(
        0, 1, size=(batch_size, args.random_dim)))
    batch_synthetic = generator(z)
    return np.round(batch_synthetic.cpu().detach().numpy())


def generate_samples_gibbs(gibbs_sampler, train: pd.DataFrame, N: int):
    # Initial value
    df = train.sample(n=1, random_state=42, axis=0)
    df.reset_index(drop=True, inplace=True)
    columns = train.columns

    for i in range(1, N): 

        X_i = df.loc[i-1]

        for dim in columns:
            cond_ecdf = gibbs_sampler.cond_ECDF(train, X_i, dim)
            x = gibbs_sampler.sample_dimension(cond_ecdf, dim)
            X_i[dim] = x
        
        df.loc[i] = X_i
    
    return df    

def project_samples(batch, dim_to_project, domain):
    """
    batch: a dataframe of synthetic samples
    dim_to_project: list of string ["HINCP", ...]
    domain: a dictionary {"HINCP": array of possible values, "NP": ...}
    """
    # find projection value
    def _find_nearest(array, value): # Using pytorch may be faster when using a GPU
        idx = (np.abs(array - value)).argmin()
        return array[idx]
    # apply projection
    for index, row in batch.iterrows():
        for dim in dim_to_project:
            row[dim] = _find_nearest(domain[dim], row[dim])
    return batch



# Model Evaluation

# def DWP(train_real: pd.DataFrame, 
#         train_synthetic: pd.DataFrame, 
#         val: pd.DataFrame, 
#         label: str):
#     """
#     This task indirectly measures how well the model captures the interdimensional
#     relationships of the real samples. We choose one dimension y to be the label. 
#     We then train two multi-class classifiers, one on the real data and one 
#     on the synthetic data. Then, we use the two models to predict the label of a 
#     pre-drawn test set from the real data. We can assume that the closer the 
#     performance of the model trained on the synthetic data is to that of the 
#     model trained on the real data, the better the quality of the synthetic dataset.

#     Source: https://arxiv.org/pdf/1703.06490.pdf
#     """
#     # Data
#     label_real = train_real[label].values
#     train_real = train_real.drop([label], axis=1).values
#     label_synthetic = train_synthetic[label].values
#     train_synthetic = train_synthetic.drop([label], axis=1).values
#     val_label = val[label].values
#     val = val.drop([label], axis=1)

#     # Logistic regression
#     model_real = LogisticRegression().fit(train_real, label_real)
#     model_synthetic = LogisticRegression().fit(train_synthetic, label_synthetic)

#     # Evaluate performance
#     pred_real = model_real.predict(val)
#     pred_synthetic = model_synthetic.predict(val)
#     f1_real = f1_score(val_label, pred_real, average="weighted")
#     f1_synthetic = f1_score(val_label, pred_synthetic, average="weighted")

#     return f1_real, f1_synthetic


def DWP(train: pd.DataFrame, test: pd.DataFrame, label: str):
    """
    This task indirectly measures how well the model captures the interdimensional
    relationships of the real samples. We choose one dimension y to be the label. 
    We then train two multi-class classifiers, one on the real data and one 
    on the synthetic data. Then, we use the two models to predict the label of a 
    pre-drawn test set from the real data. We can assume that the closer the 
    performance of the model trained on the synthetic data is to that of the 
    model trained on the real data, the better the quality of the synthetic dataset.

    Source: https://arxiv.org/pdf/1703.06490.pdf
    """
    # Data
    X_train = train.drop([label], axis=1)
    y_train = train[label]
    X_test = test.drop([label], axis=1)
    y_test = test[label]
    # Model
    classifier = LogisticRegression(solver="lbfgs", max_iter=1000)
    classifier.fit(X_train, y_train)
    # Evaluation 
    prediction = classifier.predict(X_test)
    score = f1_score(y_test, prediction, average="weighted")
    return score


def chi2(col1: pd.Series, col2: pd.Series):
    """
    This metric uses the Chi-Squared test to compare the distributions
    of the two categorical columns. It returns the resulting p-value so that
    a small value indicates that we can reject the null hypothesis (i.e. and
    suggests that the distributions are different). It tests the null hypothesis 
    that the categorical data has the given frequencies.

    Notes:
    - The sum of the observed and expected frequencies must be the same for the test to be valid.
    - The test is invalid when the observed or expected frequencies in each category are too small. 
    A typical rule is that all of the observed and expected frequencies should be at least 5.
    - A low value means there is a high correlation between the two columns (min is 0).
    - The test is not symmetric.
    """
    is_test_valid = True
    # Compute frequencies
    f_exp = col1.value_counts()
    f_obs = col2.value_counts()
    # Append zeroes if there is no example in some category
    for index in f_exp.index():
        if f_exp[index] < 5:
            is_test_valid = False
        if index not in f_obs.index:  # Will be an invalid test but will not crash
            f_obs = f_obs.append(pd.Series(data=[0], index=[index]))  
            is_test_valid = False
    for index in f_obs.index():
        if f_obs[index] < 5:
            is_test_valid = False
        if index not in f_exp.index:
            f_exp = f_exp.append(pd.Series(data=[0], index=[index]))  
            is_test_valid = False
    # Is test valid 
    if not is_test_valid:
        warnings.warn("All of the observed and expected frequencies should be at least 5 for the test to be valid")
    return chisquare(f_exp, f_obs)[1]  # Returns the p-value


def discrete_kl_div(col1: pd.Series, col2: pd.Series):
    """
    This metric compares two discrete columns using Kullback-Leibler Divergence. 
    We compute the relative frequencies of the observed and expected columns and return 
    1 / (1 + sum(kl_div(rel_exp_freq, rel_obs_freq))). 
    Since kl_div has an output in [0, inf) where 0 means the columns have the same distribution, 
    this metric returns in (0, 1] and 1 means the distributions are the same. 
    """
    # Compute frequencies
    f_exp = col1.value_counts()
    f_obs = col2.value_counts()
    # Append zeroes if there is no example in some category
    for index in f_exp.index():
        if index not in f_obs.index:
            f_obs = f_obs.append(pd.Series(data=[0], index=[index]))  
    for index in f_obs.index():
        if index not in f_exp.index:
            f_exp = f_exp.append(pd.Series(data=[0], index=[index]))  
    # Numerical stability
    f_exp += 1e-5
    f_obs += 1e-5
    # relative frequencies
    f_obs, f_exp = f_obs / np.sum(f_obs), f_exp / np.sum(f_exp)

    return 1 / (1 + np.sum(kl_div(f_exp, f_obs)))


def sampling_zeros(
    train: pd.DataFrame,
    test: pd.DataFrame,
    synthetic: pd.DataFrame):
    """
    They define sampling zeros as combinations of variables which are in the 
    test set but not in the training set. We can examine how many sampling zeros
    are recovered using the models as a proportion of the total sampling zeros.

    This function returns the number of sampling zeros

    Reference: https://arxiv.org/pdf/1909.07689.pdf
    """
    train_set = set(tuple(i) for i in train.to_numpy())
    test_set = set(tuple(i) for i in test.to_numpy())
    synthetic_set = set(tuple(i) for i in synthetic.to_numpy())

    sampling_zeros = synthetic_set.intersection(test_set) - train_set

    return len(sampling_zeros)



# Miscellaneous

def to_device(tensors, device):
    if isinstance(tensors, torch.Tensor):
        return tensors.to(device=device)
    elif isinstance(tensors, dict):
        return dict(
            (key, to_device(tensor, device)) for (key, tensor) in tensors.items()
        )
    else:
        raise NotImplementedError("Unknown type {0}".format(type(tensors)))
