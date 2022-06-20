import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF


class CopulaScaler():
    """This class can encode features with probability integral transform and 
    decode features with pseudo-inverse of the provided data. It can also 
    be used to resample discrete encoded features uniformly between the steps 
    of their ECDF.

    ECDF
    $$
    \hat{F}(x) = \frac{1}{N}\sum_{i=1}^N \mathds{1}_{x_i \leq x}
    $$

    Pseudo-inverse
    $$
    \hat{F}^{-1}(u) = \min(x: F(x) \geq u)
    $$
    """
        
    def fit(self, data):
        """Compute de ECDF and sorted pairs (x, ECDF(x)) to be used for scaling.

        Parameters
        ----------
        data : pd.DataFrame
            The data for which to compute the empirical cumulative distributon function
            (ECDF) and the pseudo-inverse. Its ECDFs will be used in the encode and 
            decode methods. 

        Returns
        -------
        self : object
            Fitted scaler
        """
        self.data = data
        self.columns = data.columns
        self.ecdfs = {}  # Empirical CDF as a step function.
        self.uniques = {}  # {column : [(x, ecdf(x)), ...], ...} sorted

        for column in self.columns:
            feature = data[column]
            # ECDF
            ecdf = ECDF(feature)
            self.ecdfs[column] = ecdf
            # Setting the table for pseudo inverse ecdf
            x_u = []  # [(x_1, ECDF(x_1)), ..., (x_d, ECDF(x_d))] with x_i < x_j for i < j
            uniques = np.sort(feature.unique())  # [x_1, ..., x_d]
            for x in uniques:
                u = ecdf(x)
                x_u.append((x, u))
            self.uniques[column] = x_u

        return self
    
            
    def transform(self, data):
        """Compute the probability integral transform of the data.

        Parameters
        ----------
        data : pd.DataFrame
            The data used to scale,

        Returns
        -------
        data_tr : pd.DataFrame
            The transformed data 
        """
        data_tr = data.copy()
        for column in self.columns:
            data_tr[column] = data_tr[column].apply(self.ecdfs[column])
        return data_tr

    
    def inverse_transform(self, data):
        """Scale back the data to the primal space with the pseudo-inverse.

        Parameters
        ----------
        data : pd.DataFrame 
            The rescaled data to be transformed back

        Returns
        -------
        data_tr : pd.DataFrame
            The trasnformed data
        """
        data_tr = data.copy()
        for column in self.columns:
            x_u = np.array(self.uniques[column])
            data_tr[column] = data_tr[column].apply(CopulaScaler.pseudo_inverse, args=(x_u,))
        return data_tr


    def resampling_trick(self, data):
        """Resample discrete encoded features uniformly between the steps of the ECDF.

        Parameters
        ----------
        data : pd.DataFrame
            The rescaled data for which to resample the discrete features.

        categorical : array-like
            The categorical columns for which to resample

        Returns
        -------
        data_tr : pd.DataFrame
            The data but with resampled discrete features 
        """

        def sample_uniform(u, x_u):
            us = x_u[:, 1]
            idx = list(us).index(u)
            if idx == 0:
                return np.random.uniform(low=0, high=u)
            else:
                return np.random.uniform(low=us[idx-1], high=u)

        data_tr = data.copy()
        for column in self.columns:
            x_u = np.array(self.uniques[column])
            data_tr[column] = data_tr[column].apply(sample_uniform, args=(x_u,))
        return data_tr


    @staticmethod
    def pseudo_inverse(u, x_u):
        """Compute pseudo inverse.

        Parameters
        ----------
        u : float
            The value for which to compute the pseudo-inverse

        x_u : array-like of Tuples
            Array of pairs (x, ECDF(x)) 

        Returns
        -------
        x : float
            The pseudo-inverse value of u given the ECDF x_u
        """
        us = x_u[:, 1]
        valid_i = np.ma.MaskedArray(us, us<u)
        x = x_u[np.ma.argmin(valid_i), 0]
        return x

