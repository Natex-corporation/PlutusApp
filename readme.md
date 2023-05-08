# Plutus Trading App
This is a Python application that allows users to select multiple stock tickers and download their trading data for training a neural network. The application also provides an option to select a trading API and keys for paper trading. The trading data is updated every 30 seconds during market hours and a prediction is generated for each stock ticker. The results are displayed in a new window along with a chart.

## Requirements
- Python 3.7+
- tkinter
- pandas
- data_download
- Pillow
- Tensorflow
## Installation
1. Clone the repository
2. Install the required packages using pip

```
pip install -r requirements.txt
```
## Usage
1. Run the application by executing the following command:

``` 
python appV2.py 
```
2. Select one or more stock tickers from the list.
3. Choose a trading API and enter your API keys (optional).
4. Click on the "Start" button.
A new window will open displaying the trading results and a chart.
## License
This project is licensed under the MIT License. See the [LICENSE](/LICENSE) file for details.

Don't forget to regularly update your API keys and check for any changes in the trading data. Happy trading! 📈💰