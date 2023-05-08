import unittest
import os
import shutil
import data_download as dd

class TestDownload(unittest.TestCase):
    
    def test_download(self):
        tickers = ['AAPL', 'GOOG', 'MSFT']
        dd.download(tickers)
        for ticker in tickers:
            filename = 'app_test/{}.csv'.format(ticker)
            self.assertTrue(os.path.exists(filename))
            os.remove(filename)
            
    def test_train_nn(self):
        dd.train_nn()
        self.assertTrue(os.path.exists('model.h5'))
    
if __name__ == '__main__':
    unittest.main()