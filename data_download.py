import yfinance as yf
from datetime import date
from datetime import timedelta
import pandas as pd
import os
from difflib import SequenceMatcher
import datetime
import pandas as pd
import yfinance as yf
import requests
import uuid
from github import Github
from github import InputGitTreeElement
import shutil
import tensorflow as tf
import time
import signal

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

checker = 0
def download(tickers):
    """Downloads the newest data that are avaiilable for training 

    Args:
        tickers (string_list): the list of strings should contain stock tickers in all capital letters
    """
    
    for name in tickers:
        reps = os.walk('')
        print(reps) 
        today = date.today()
        yesterday =  str(today - timedelta(days=7))
        today = str(today)
        print(today, yesterday)
        data = yf.download(name, start=yesterday, end=today, interval='1m') #str(yesterday); str(today)
        #print (data, 'data')
        df = pd.DataFrame(data)
        if(len(df)>20):
            df.to_csv('app_test/'+name+'.csv', mode='a', header=True)
        
def train_nn():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    import seaborn as sns
    import os
    from datetime import datetime
    from difflib import SequenceMatcher
    import warnings
    from sklearn.preprocessing import MinMaxScaler
    
    warnings.filterwarnings("ignore")
    
    linker = 'app_test'#'finised_files'
    names = os.listdir(linker)
    itemss = os.scandir()
    if os.path.exists('model.h5'):
        os.remove('model.h5')
    
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(units=512, return_sequences=True, input_shape=(512, 6)))
    model.add(keras.layers.LSTM(units=512))
    model.add(keras.layers.Dense(256))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(128))
    model.add(keras.layers.Dropout(0.5))   
    model.add(keras.layers.Dense(64))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(32))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1))
    model.summary
    model.compile(optimizer='adam', loss='mean_squared_error')

    for imar in names:
        print(imar)
        data = pd.read_csv(linker + '/' + imar) 
        data.info()
    
        close_data = data.filter(['Close', 'Open', "High", "Low", "Volume", "Adj Close"])
        dataset = close_data.values
        dataset_close = close_data['Close'].values
        training = int(np.ceil(len(dataset)*.7))   

        scaler = MinMaxScaler(feature_range=(0, 1))
        scalar_close = MinMaxScaler(feature_range=(0, 1))
        dataset_close = dataset_close.reshape(-1,1)
        slop = scalar_close.fit_transform(dataset_close)
        scaled_data = scaler.fit_transform(dataset)
        train_data = scaled_data[0:int(training), :]

        x_train = []
        y_train = []
    
        for i in range(512, len(train_data)):
            #print (i, '6', train_data[i-512:i, 0:6])
            x_train.append(train_data[i-512:i, 0:6])
            y_train.append(train_data[i, 0])
    
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 6))
  
       
        hystory = model.fit(x_train, y_train, epochs=5)

        test_data = scaled_data[training - 512:, :]
        x_test = []
        y_test = dataset[training:, :]
        for i in range(512, len(test_data)):
            x_test.append(test_data[i-512:i, 0:6])
  
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 6))

        predictions = model.predict(x_test)
        predictions = scalar_close.inverse_transform(predictions)
  
    # evaluation metrics
        mse = np.mean(((predictions - y_test) ** 2))
    #print("MSE", mse)
        print("RMSE", np.sqrt(mse))

        train = data[:training]
        test = data[training:]
        test['Predictions'] = predictions
  
        
    model.save('model.h5')
    
def live_data(tickers):
    
    # Get today's date and check if it's a weekend or holiday
    today = datetime.date.today()
    weekday = today.weekday()
    holidays = pd.read_csv('sources_linker/trading_days.csv')[' ']
    is_holiday = today in holidays # List of holidays
    if weekday >= 5 or is_holiday:
        print("Stock market data cannot be downloaded on weekends or holidays")
        return

    # Download stock market data for each ticker
    for name in tickers:
        yesterday =  today - datetime.timedelta(days=1)
        data = yf.download(name, start=str(yesterday), end=str(today), interval='1m')
        df = pd.DataFrame(data)
        df.to_csv('live_data/'+name+'.csv', mode='a', header=False)
        
def setup_env_file():
    env_path = ".env"
    #expected_vars = ["API_provider", "Secret_key", "Private_key", "Paper_trading", "limit_usage", "limit_computing", "selected_stocks", "unique_ID", "github_token"]

    # Create .env file if it doesn't already exist
    if not os.path.exists(env_path):
        with open(env_path, "w") as f:
            f.write("API_provider=\nSecret_key=\nPrivate_key=\nPaper_trading=\nlimit_usage=\nlimit_computing=\nselected_stocks=\nunique_ID=\ngithub_token=\n")

    try:
        os.getenv('API_provider')
        os.getenv('Secret_key')
        os.getenv('Private_key')
        os.getenv('Paper_trading')
        os.getenv('limit_usage')
        os.getenv('limit_computing')
        os.getenv('selected_stocks')
        os.getenv('unique_ID')
        os.getenv('github_token')
    except:
        print('something went wrong')

def download_model(github_url='https://github.com/Natex-corporation/models', model_filename='model.h5'):
    """
    Download a machine learning model from a GitHub repository.

    Args:
        github_url (str): The URL of the GitHub repository where the model is located.
        model_filename (str): The filename of the model to download.

    Returns:
        None
    """
    response = requests.get(f"{github_url}/raw/models/{model_filename}")
    with open(model_filename, "wb") as f:
        f.write(response.content)

def assign_unique_ID():
    """
    Generates a unique ID by concatenating the current date and time, and a UUID. Writes the unique ID and other environment
    variables to a file named ".env".
    
    Returns:
    None
    """
    API_provider = os.getenv('API_provider')
    Secret_key = os.getenv('Secret_key')
    Public_key = os.getenv('Public_key')
    Paper_trading = os.getenv('Paper_trading')
    limit_usage = os.getenv('limit_usage')
    limit_computing = os.getenv('limit_computing')
    selected_stocks = os.getenv('selected_stocks')
    unique_ID = os.getenv('unique_ID')
    github_token = os.getenv('github_token')
    git_executable = os.getenv('git_executable')
    now = datetime.datetime.now()
    unique_ID = str(now.date()) + "_" + str(now.time().strftime('%H%M%S')) + "_" + str(uuid.uuid4())

    with open(".env", "w") as f:
        f.write(f"API_provider={API_provider}\n")
        f.write(f"Secret_key={Secret_key}\n")
        f.write(f"Public_key={Public_key}\n")
        f.write(f"Paper_trading={Paper_trading}\n")
        f.write(f"limit_usage={limit_usage}\n")
        f.write(f"limit_computing={limit_computing}\n")
        f.write(f"selected_stocks={selected_stocks}\n")
        f.write(f"unique_ID={unique_ID}\n")
        f.write(f"github_token={github_token}\n")
        f.write(f"git_executable={git_executable}\n")        

def add_model_files_to_branch(repository_name: str, unique_ID: str, base_branch: str, model_files= ['Trained.h5', 'Info.csv']) -> None:
    git_executable_path = os.getenv('git_executable')#'''C:/Program Files/Git/cmd/git.exe'''
    os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = git_executable_path
    github_token = os.getenv('github_token')
    import git
    from git import Repo
    from github import Github, Repository
    import shutil

    # Authenticate with Github using the personal access token
    g = Github(github_token)

    # Get the repository object
    repo = g.get_repo(repository_name)

    # Check if a branch with the unique ID already exists
    try:
        existing_branch = repo.get_branch(unique_ID)
    except:
        existing_branch = None

    # If the branch already exists, clone its contents
    if existing_branch:
        print(f"Branch '{unique_ID}' already exists. Cloning its contents...")
        local_dir = 'E:\PlutusApp\send'
        os.mkdir(local_dir)
        Repo.clone_from(f"https://github.com/{repository_name}.git", local_dir, branch=unique_ID)

    # If the branch doesn't exist, create a new branch
    else:
        print(f"Creating a new branch '{unique_ID}'...")
        # Get the base branch object
        base_branch_obj = repo.get_branch(base_branch)

        # Get the base commit sha
        base_commit_sha = base_branch_obj.commit.sha

        # Create a new branch
        new_branch_ref = f"refs/heads/{unique_ID}"
        new_branch = repo.create_git_ref(ref=new_branch_ref, sha=base_commit_sha)

        # Clone the repository to a local directory
        local_dir = 'E:\PlutusApp\send'
        os.mkdir(local_dir)
        Repo.clone_from(f"https://github.com/{repository_name}.git", local_dir, branch=unique_ID)

    # Add the new files to the local directory
    for file in model_files:
        shutil.copy(file, local_dir)

    commit_message = f"Add model files at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    # Add and commit the new files
    repo = Repo(local_dir)
    repo.index.add(model_files)
    repo.index.commit(commit_message)

    # Push the changes to the new branch
    remote_branch_ref = f"refs/heads/{unique_ID}"
    try:
        repo.remote(name='origin').push(refspec=f"{remote_branch_ref}:{remote_branch_ref}")
    except git.GitCommandError as e:
        print(f"Error occurred while pushing changes: {e}")
        print("Changes were not pushed to the remote repository.")
    else:
        print("Changes were successfully pushed to the remote repository.")

    print(f"Branch '{unique_ID}' created successfully!")
    
def delete_send_folder():
    folder_path = 'E:\PlutusApp\send' # specify the path to the 'send' folder
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"'{folder_path}' folder has been deleted successfully.")
    else:
        print(f"'{folder_path}' folder does not exist.")
        
def benchmark_training_speed(num_epochs=100):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
    y_train = tf.keras.utils.to_categorical(y_train, 10)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

    times = {}
    with tf.device('/cpu:0'):
        start_time = time.time()
        for epoch in range(num_epochs):
            for x, y in train_ds:
                with tf.GradientTape() as tape:
                    predictions = model(x)
                    loss = loss_fn(y, predictions)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        cpu_time = time.time() - start_time
        print("Training time on CPU: {:.4f}s".format(cpu_time))
        times['CPU'] = cpu_time

    if tf.test.is_gpu_available():
        with tf.device('/gpu:0'):
            start_time = time.time()
            for epoch in range(num_epochs):
                for x, y in train_ds:
                    with tf.GradientTape() as tape:
                        predictions = model(x)
                        loss = loss_fn(y, predictions)
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            gpu_time = time.time() - start_time
            print("Training time on GPU: {:.4f}s".format(gpu_time))
            times['GPU'] = gpu_time
    else:
        print("GPU acceleration is not available. Training on CPU.")
    
    return times

def training_details(github_url='https://github.com/Natex-corporation/models', unique_ID='model.h5'):
    unique_ID=os.getenv('unique_ID')
    unique_ID=unique_ID+'.csv'
    response = requests.get(f"{github_url}/raw/main/details/{unique_ID}")
    with open(unique_ID, "wb") as f:
        f.write(response.content)
    