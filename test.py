import tkinter as tk
from tkinter import ttk

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        # Personnaliser le style du menu
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TMenubar', background='pink', foreground='white', font=('Arial', 10))
        style.configure('TMenu', background='#444444', foreground='white', font=('Arial', 10), borderwidth=0, activebackground='#555555', activeforeground='white')

        # Créer le menu
        menu_bar = tk.Menu(self)
        self.config(menu=menu_bar)

        # Créer les options du menu
        file_menu = tk.Menu(menu_bar, tearoff=False)
        menu_bar.add_cascade(label='File', menu=file_menu)
        file_menu.add_command(label='Open', command=self.open_file)
        file_menu.add_command(label='Save', command=self.save_file)
        file_menu.add_separator()
        file_menu.add_command(label='Exit', command=self.quit)

    def open_file(self):
        print('Open file')

    def save_file(self):
        print('Save file')

if __name__ == '__main__':
    app = App()
    app.mainloop()
