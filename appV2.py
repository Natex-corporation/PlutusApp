import tkinter as tk
import time
import datetime
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import data_download as dd


df= pd.read_csv('sources_linker/tickers.csv')
df = df['Symbol']
print(df)

trading_day = False
today = datetime.date.today()
dt = pd.read_csv('sources_linker/trading_days.csv')
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

class App:
    def __init__(self, master):
        self.master = master
        master.title("Multi-Select Demo")

        self.lb = tk.Listbox(master, selectmode=tk.MULTIPLE)
        for i in df:
            self.lb.insert(tk.END, f"{i}")
        self.lb.pack()
        
        # create dropdown
        self.dropdown_var = tk.StringVar()
        self.dropdown_var.trace("w", lambda name, index, mode: self.update_input_fields())
        self.dropdown = tk.OptionMenu(master, self.dropdown_var, 'Alpaca api', 'coming soon')
        self.dropdown.pack()

        # create input fields
        self.input_label = tk.Label(master, text="secret key")
        #self.input_label.pack()
        self.input_entry = tk.Entry(master)
        #self.input_entry.pack()
        self.input_label_pub = tk.Label(master, text="public key")
        #self.input_label_pub.pack()
        self.input_entry_pub = tk.Entry(master)
        #self.input_entry_pub.pack()
        self.input_checkbox_var = tk.BooleanVar()
        self.input_checkbox = tk.Checkbutton(master, text="paper trade?", variable=self.input_checkbox_var)
        #self.input_checkbox.pack()

        # create start button
        self.start_button = tk.Button(master, text="Start", command=self.start)
        #self.start_button.pack()

        # initialize new_window as None
        self.new_window = None

    def update_input_fields(self):
        if self.dropdown_var.get() == 'Alpaca api':
            self.input_label.pack()
            self.input_entry.pack()
            self.input_label_pub.pack()
            self.input_entry_pub.pack()
            #(state='normal')
            self.input_checkbox.pack()#(state='normal')
            self.start_button.pack()#(state='normal')
        else:
            print('nothing')
            '''self.input_entry.delete(0, tk.END)
            self.input_entry.config(state='disabled')
            self.input_checkbox.deselect()
            self.input_checkbox.config(state='disabled')
            self.start_button.config(state='disabled')'''

    def start(self):
        selected_item = self.dropdown_var.get()
        selected_indices = self.lb.curselection()
        selected_items = [self.lb.get(index) for index in selected_indices]
        print("Selected items:")
        tickers = []
        for item in selected_items:
            print(item)
            tickers.append(str(item))
        print("Selected item:", selected_item)
        print("Input secret:", self.input_entry.get())
        print("Input secret:", self.input_entry_pub.get())
        checkbox_state = self.input_checkbox_var.get()
        print("Checkbox:", checkbox_state)
        
        #training download
        dd.download(tickers)
        
        # close input window and open new window
        self.master.withdraw()
        self.new_window = tk.Toplevel()
        self.new_window.title("New Window")
        self.new_window.geometry("500x500")
        self.new_window.protocol("WM_DELETE_WINDOW", self.on_new_window_close)

        # create label to display file contents
        self.file_label = tk.Label(self.new_window, text="Trading")
        self.file_label.pack()

        # start timer to update label every 30 seconds
        self.update_file_label()
        self.new_window.after(30000, self.update_file_label)

    def update_file_label(self):
        # read file and set label text
        
  
        # the figure that will contain the plot
        fig = Figure(figsize = (5, 5), dpi = 100)
  
        # list of squares
        y = pd.read_csv('results/AAL.csv')
        y = y['0']
        # adding the subplot
        plot1 = fig.add_subplot(111)
        # plotting the graph
        plot1.plot(y)
        canvas = FigureCanvasTkAgg(fig, master=self.new_window)  
        canvas.draw()
        canvas.get_tk_widget().pack()
        # creating the Matplotlib toolbar
        toolbar = NavigationToolbar2Tk(canvas, master=self.new_window)
        toolbar.update()
        canvas.get_tk_widget().pack()
        
        with open("items.txt", "r") as f:
            contents = f.read()
        self.file_label.config(text=contents)

        # start timer to update label again in 30 seconds
        self.new_window.after(30000, self.update_file_label)

    def on_new_window_close(self):
        # restore main window and destroy new window
        self.master.deiconify()
        self.new_window.destroy()

root = tk.Tk()
app = App(root)
root.mainloop()
