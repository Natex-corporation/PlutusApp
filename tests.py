import unittest
import os
import data_download as dd
import unittest
import datetime
import pandas as pd
from unittest.mock import patch, MagicMock
from io import StringIO
import uuid

class TestDownload(unittest.TestCase):
    
    def test_download(self):
        tickers = ['AAPL', 'GOOG', 'MSFT']
        dd.download(tickers)
        for ticker in tickers:
            filename = 'app_test/{}.csv'.format(ticker)
            self.assertTrue(os.path.exists(filename))
            os.remove(filename)
            
    '''def test_train_nn(self):
        dd.train_nn()
        self.assertTrue(os.path.exists('model.h5'))'''
    
    


class TestLiveData(unittest.TestCase):

    def test_live_data_on_weekends_and_holidays(self):
        # Define a weekend and a holiday
        weekend = datetime.date(2023, 5, 6)  # Saturday
        holiday = datetime.date(2023, 1, 1)  # New Year's Day

        # Test on a weekend
        with patch('data_download.datetime') as mock_datetime:
            mock_datetime.date.today.return_value = weekend
            with patch('sys.stdout', new=StringIO()) as fake_output:
                dd.live_data(['AAPL'])
                self.assertEqual(fake_output.getvalue().strip(), "Stock market data cannot be downloaded on weekends or holidays")
        
        # Test on a holiday
        with patch('data_download.datetime') as mock_datetime:
            mock_datetime.date.today.return_value = holiday
            with patch('sys.stdout', new=StringIO()) as fake_output:
                dd.live_data(['AAPL'])
                self.assertEqual(fake_output.getvalue().strip(), "Stock market data cannot be downloaded on weekends or holidays")
                
def assign_unique_ID():
    now = datetime.datetime.now()
    unique_id = str(now.date()) + "_" + str(now.time().strftime('%H%M%S')) + "_" + str(uuid.uuid4())

    print(unique_id)

class TestAssignUniqueID(unittest.TestCase):

    def test_assign_unique_ID(self):
        # Call the function to assign a unique ID
        assign_unique_ID()

        # Verify that the output is a string
        self.assertIsInstance(assign_unique_ID(), str)

    
if __name__ == '__main__':
    unittest.main()