import unittest
import pandas as pd
from NDR import NDR
import warnings
import pickle as pk


class Testing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = pd.read_csv('yahoo_389c_5047r.csv')
        prices = data.dropna(axis=1).set_index('Date')
        train_start_point = 3000
        cls.X = prices[train_start_point:]
        cls.assertIsNotNone(cls, cls.X)

    def test_NMF_fit(self):
        X = self.X
        n_comp_list = [2, 3, 4]
        window_size_list = [5, 10]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                for n_comp in n_comp_list:
                    for window_size in window_size_list:
                        NMF = NDR(is_NMF=True, n_comp=n_comp, window_size=window_size)
                        NMF.fit(X)
                        with open(f"models/NMF_{str(n_comp)}_{str(window_size)}.pkl", "wb") as file:
                            pk.dump(NMF, file)
            except Exception as e:
                self.assertLogs("fit error, error message: " + e)

    def test_NPCA_fit(self):
        X = self.X
        n_comp_list = [2, 3, 4]
        window_size_list = [5, 10]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                for n_comp in n_comp_list:
                    for window_size in window_size_list:
                        NPCA = NDR(is_NPCA=True, n_comp=n_comp, window_size=window_size)
                        NPCA.fit(X)
                        with open(f"models/NPCA_{str(n_comp)}_{str(window_size)}.pkl", "wb") as file:
                            pk.dump(NPCA, file)
            except Exception as e:
                self.assertLogs("fit error, error message: " + e)

    def test_NPCA_shape(self):
        X = self.X
        n_comp_list = [2, 3, 4]
        window_size_list = [5, 10]

        for n_comp in n_comp_list:
            for window_size in window_size_list:
                with open(f"models/NPCA_{str(n_comp)}_{str(window_size)}.pkl", "rb") as file:
                    NPCA = pk.load(file)
                X_reduced = NPCA.transform(X)
                self.assertEqual(X_reduced.shape[1], X.shape[1])
                self.assertEqual(window_size * X_reduced.shape[0] + (X.shape[0] % window_size), X.shape[0])

    def _test_not_neg(self, X):
        not_neg = X < 0
        for array in not_neg:
            for elem in array:
                self.assertFalse(elem)

    def test_NMF_not_neg(self):
        n_comp_list = [2, 3, 4]
        window_size_list = [5, 10]

        for n_comp in n_comp_list:
            for window_size in window_size_list:
                with open(f"models/NMF_{str(n_comp)}_{str(window_size)}.pkl", "rb") as file:
                    NMF = pk.load(file)
                X_reduced = NMF.nmf_prices
                self._test_not_neg(X_reduced)

    def test_NPCA_not_neg(self):
        n_comp_list = [2, 3, 4]
        window_size_list = [5, 10]

        for n_comp in n_comp_list:
            for window_size in window_size_list:
                with open(f"models/NPCA_{str(n_comp)}_{str(window_size)}.pkl", "rb") as file:
                    NPCA = pk.load(file)
                X_reduced = NPCA.npca_prices
                self._test_not_neg(X_reduced)


if __name__ == '__main__':
    unittest.main()
