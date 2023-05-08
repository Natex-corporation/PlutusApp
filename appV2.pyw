import tkinter as tk
import datetime
import pandas as pd
import data_download as dd
from PIL import ImageTk, Image
import datetime
import os
import magic

print(datetime.datetime.now())

#os.remove('app_test/ABBV.csv')
names = os.listdir('app_test')
for i in names:
    os.remove('app_test/'+i)

all_tickers= pd.read_csv('sources_linker/tickers.csv')
all_tickers = all_tickers['Symbol']
print(all_tickers)

dt = pd.read_csv('sources_linker/trading_days.csv')
trading_day = False
today = datetime.date.today()
today = str(today)

t=[]

for c in today:
    t.append(ord(c))
    
for i in range(len(dt[' '])):
    k=[]
    for c in dt[' '][i]:
        k.append(ord(c))
    if k==t:
        trading_day = True
dd.setup_env_file()
class App:
    def __init__(self, master):   
        self.master = master
        master.title("Plutus")
        self.master.geometry("150x500")

        # Creating a styled label with a larger font size and bold text
        self.ticker_label = tk.Label(master, text="Select Ticker(s)", font=("Helvetica", 14, "bold"), fg="#555")
        self.ticker_label.pack(pady=10)

        # Adding a border and a background color to the Listbox
        self.lb = tk.Listbox(master, selectmode=tk.MULTIPLE, bd=0, bg="#eee")

        # Inserting each ticker into the Listbox
        for i in all_tickers:
            self.lb.insert(tk.END, f"{i}")

        self.lb.pack(fill='both', expand=True)
        
        '''# create dropdown
        self.dropdown_var = tk.StringVar()
        self.dropdown_var.trace("w", lambda name, index, mode: self.update_input_fields())
        self.dropdown = tk.OptionMenu(master, self.dropdown_var, 'Alpaca api', 'coming soon')
        self.dropdown.pack()'''

        # Creating a styled dropdown menu with a border and a background color
        self.dropdown_var = tk.StringVar()
        self.dropdown_var.set('Choose api')
        self.dropdown_var.trace("w", lambda name, index, mode: self.update_input_fields())
        self.dropdown_options = ['Alpaca API', 'Coming Soon']
        self.dropdown = tk.OptionMenu(master, self.dropdown_var, *self.dropdown_options)
        self.dropdown.config(bd=0, bg="#eee", font=("Helvetica", 12))
        self.dropdown.pack(pady=5)      

        # Creating styled labels and input fields
        self.input_label = tk.Label(master, text="Secret Key", font=("Helvetica", 14, "bold"), fg="#555")
        self.input_entry = tk.Entry(master, font=("Helvetica", 12))
        self.input_label_pub = tk.Label(master, text="Public Key", font=("Helvetica", 14, "bold"), fg="#555")
        self.input_entry_pub = tk.Entry(master, font=("Helvetica", 12))
        self.input_checkbox_var = tk.BooleanVar()
        self.input_checkbox = tk.Checkbutton(master, text="Paper Trade", variable=self.input_checkbox_var, font=("Helvetica", 12))

        # Styling the Start button with a larger font size, bold text, and a background color
        self.start_button = tk.Button(master, text="Start", font=("Helvetica", 14, "bold"), fg="white", bg="#555", command=self.start)

        

    def update_input_fields(self):
        if self.dropdown_var.get() == 'Alpaca API':
            # Adding padding to the labels, input fields, and checkbox for better spacing
            self.input_label.pack(pady=5)
            self.input_entry.pack(pady=2)
            self.input_label_pub.pack(pady=5)
            self.input_entry_pub.pack(pady=2)
            self.input_checkbox.pack(pady=2)
            self.start_button.pack(pady=10)
            
        else:
            print('nothing')
            
    def start_trading(self):
        selected_item = self.dropdown_var.get()
        selected_indices = self.lb.curselection()
        selected_items = [self.lb.get(index) for index in selected_indices]
        secret_key =  self.input_entry.get()
        publick_key = self.input_entry_pub.get()
        checkbox_state = self.input_checkbox_var.get()
        self.Donation.destroy()
        print("Selected items:")
        tickers = []
        for item in selected_items:
            print(item)
            tickers.append(str(item))
            
        
        env_path = ".env"
        with open(env_path, "w") as f:
            f.write(f"API_provider={selected_item}\n")
            f.write(f"Secret_key={secret_key}\n")
            f.write(f"Public_key={publick_key}\n")
            f.write(f"Paper_trading={checkbox_state}\n")
            f.write(f"limit_usage={self.computing_checkbox_var.get()}\n")
        
        #self.start_trading_button = tk.Button(master, text="Start", font=("Helvetica", 14, "bold"), fg="white", bg="#555", command=self.start)
        #training download
        if os.getenv('limit_usage') == False:
            dd.download(tickers)
            dd.train_nn()
        else: 
            dd.download_model('https://github.com/Natex-corporation/models')
            
        # close input window and open new window
        self.master.withdraw()
        self.new_window = tk.Toplevel()
        self.new_window.title("Plutus Trading")
        self.new_window.geometry("1000x1000")
        self.new_window.protocol("WM_DELETE_WINDOW", self.on_new_window_close)

        # create label to display file contents
        self.trading_days = tk.Label(self.new_window, text="")
        self.trading_days.pack()
        self.file_label = tk.Label(self.new_window, text="")
        self.file_label.pack()
        self.img = ImageTk.PhotoImage(Image.open("Figure_1.png"))

        # Create a Label Widget to display the text or Image
        self.label = tk.Label(self.new_window, image = self.img)
        self.label.pack()

        # start timer to update label every 30 seconds
        self.update_file_label()
        self.new_window.after(3000, self.update_file_label)
                
    def start(self):
        if os.getenv('limit_usage') != False and os.getenv('limit_usage') != True:
            self.master.withdraw()
            self.Donation = tk.Toplevel()
            self.Donation.title("Plutus Trading")
            self.Donation.geometry("100x100")
            self.Donation.protocol("WM_DELETE_WINDOW", self.on_new_window_close)
        
            self.computing_checkbox_var = tk.BooleanVar(self.Donation)
            self.computing_checkbox = tk.Checkbutton(self.Donation, text="Contribute to project", variable=self.computing_checkbox_var, font=("Helvetica", 12))
            self.computing_checkbox.pack(pady=2)
            '''if self.computing_checkbox_var.get() == True:
                os.environ["limit_usage"] = 'False'
            else:
                os.environ["limit_usage"] = 'True'''
            self.start_trading_button = tk.Button(self.Donation, text="Start", font=("Helvetica", 14, "bold"), fg="white", bg="#555", command=self.start_trading)
            self.start_trading_button.pack()
        else:
            self.start_trading()
        
        
    def update_file_label(self):
        # read file and set label text
        
        if(trading_day==True):
            self.trading_days.config(text='Dnes se obchoduje') 
            magic.magic()
            with open("items.txt", "r") as f:
                contents = f.read()
            self.file_label.config(text=contents)
            
        if(trading_day==False):
            self.trading_days.config(text='Dnes se neobchoduje') #= tk.Label(self.new_window, )
            #self.trading_days.pack()
        # start timer to update label again in 30 seconds
        dd.live_data((pd.read_csv('output.csv'))['tickers'])
        self.new_window.after(3000, self.update_file_label)

    def on_new_window_close(self):
        # restore main window and destroy new window
        self.master.deiconify()
        self.new_window.destroy()

root = tk.Tk()
app = App(root)
root.mainloop()
