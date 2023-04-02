
import tkinter as tk
from tkinter import filedialog, Canvas
from PIL import Image, ImageTk
import PIL
import cv2
import numpy as np
#from tkinter import*

class VO_MainApplication:
    def __init__(self, master):
        #self : est une instance de la classe elle meme, pour pouvoir acceder a toutes les var et mth de la classe
        #master : est un objet de la classe Tk qui represente le fentre principale
        # 
        self.master = master
        self.master.title("Projet VO")
        self.frame = tk.Frame(self.master)

        self.image = None
        self.original_image = None
        self.modified_image = None

        #Creer un objet de menu et l'ajouter a fenetre
        menuBar = tk.Menu(self.master)
        self.master.config(menu= menuBar)
        #SOUS MENU IMAGE
        img_menu = tk.Menu(menuBar, tearoff=0)
        menuBar.add_cascade(label='Image', menu=img_menu)
        img_menu.add_command(label='Ouvrire', accelerator='Ctrl+O' ,command=self.open_image)
        img_menu.add_command(label='Enregistrer', accelerator='Ctrl+S', command=self.save_image)
        img_menu.add_separator()
        img_menu.add_command(label='Quitter', accelerator='Alt+F4', command=self.master.destroy)

        #SOUS MENU TRANSFORMATION
        transformation_menu = tk.Menu(menuBar, tearoff=0)
        menuBar.add_cascade(label='Transformation', menu=transformation_menu)
        transformation_menu.add_command(label='Negative', command=self.function)
        transformation_menu.add_command(label='Rotation', command=self.function)
        transformation_menu.add_command(label='Redimension', command=self.function)
        transformation_menu.add_command(label='Rectangle', command=self.function)
        transformation_menu.add_command(label='Histogramme', command=self.function)
        transformation_menu.add_command(label='Etirement', command=self.function)
        transformation_menu.add_command(label='Egalisation', command=self.function)

        #SOUS MENU BINARISATION
        binarisation_menu = tk.Menu(menuBar, tearoff=0)
        menuBar.add_cascade(label='Binarisation', menu=binarisation_menu)
        binarisation_menu.add_command(label='Seuillage manuel', command=self.function)
        binarisation_menu.add_command(label='OTSU', command=self.function)
        binarisation_menu.add_command(label='Autres', command=self.function)
        
        #SOUS MENU FILTRAGE
        filtrage_menu = tk.Menu(menuBar, tearoff=0)
        menuBar.add_cascade(label='Filtrage', menu=filtrage_menu)
        filtrage_menu.add_command(label='Gaussien', command=self.function)
        filtrage_menu.add_command(label='Moyenneur', command=self.function)
        filtrage_menu.add_command(label='Median', command=self.function)

        #SOUS MENU EXTRACTION CONTOURS
        extraction_menu = tk.Menu(menuBar, tearoff=0)
        menuBar.add_cascade(label='Extraction Contours', menu=extraction_menu)
        extraction_menu.add_command(label='Gradient', command=self.function)
        extraction_menu.add_command(label='Sobel', command=self.function)
        extraction_menu.add_command(label='Robert', command=self.function)
        extraction_menu.add_command(label='Laplacien', command=self.function)

        #SOUS MENU MORPHOLOGIE
        morphologie_menu = tk.Menu(menuBar, tearoff=0)
        menuBar.add_cascade(label='Morphologie', menu=morphologie_menu)
        morphologie_menu.add_command(label='Erosion', command=self.erosion)
        morphologie_menu.add_command(label='Dilatation', command=self.dilatation)
        morphologie_menu.add_command(label='Ouverture', command=self.function)
        morphologie_menu.add_command(label='Fermeture', command=self.function)
        morphologie_menu.add_command(label='Filtrage morphologique', command=self.function)

        #SOUS MENU SEGMENTATION
        segmentation_menu = tk.Menu(menuBar, tearoff=0)
        menuBar.add_cascade(label='Segmentation', menu=segmentation_menu)
        segmentation_menu.add_command(label='Croissance de regions D', command=self.function)
        segmentation_menu.add_command(label='Partition de regions D', command=self.function)
        segmentation_menu.add_command(label='k_means', command=self.function)
  
        #SOUS MENU POINTS D INTERET
        ptsInteret_menu = tk.Menu(menuBar, tearoff=0)
        menuBar.add_cascade(label="Point d'interet", menu=ptsInteret_menu)
        ptsInteret_menu.add_command(label='xx', command=self.function)
        ptsInteret_menu.add_command(label='xx', command=self.function)
        ptsInteret_menu.add_command(label='xx', command=self.function)
  
        #SOUS MENU POINTS D INTERET
        compression_menu = tk.Menu(menuBar, tearoff=0)
        menuBar.add_cascade(label="Compression", menu=compression_menu)
        compression_menu.add_command(label='xx', command=self.function)
        compression_menu.add_command(label='xx', command=self.function)
        compression_menu.add_command(label='xx', command=self.function)

    

        # Création du bouton pour ouvrir une image
        self.button_open_image = tk.Button(master, text="Ouvrir une image", command=self.open_image)
        self.button_open_image.pack()

        # Création des deux zones d'affichage pour les images
        self.canvas_original_image = tk.Canvas(master, width=400, height=400)
        self.canvas_original_image.pack(side=tk.LEFT, padx=10, pady=10)

        self.canvas_modified_image = tk.Canvas(master, width=400, height=400)
        self.canvas_modified_image.pack(side=tk.LEFT, padx=10, pady=10)
         




        # Créer un canevas pour afficher l'image
        #self.canvas = Canvas(self.master)



        

    # Methode pour ouvrir une image
    def open_image(self):
        imgPath = filedialog.askopenfilename()
        if imgPath :
            self.original_image = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
            self.display_original_image()
        

    """"
        # Afficher l'image dans le canevas
        self.canvas.config(width=image.width, height=image.height)
        self.canvas.pack()
        img_tk = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, image=img_tk, anchor='nw')
        self.canvas.image = img_tk  # pour empêcher la suppression de l'image par le ramasse-miettes
    """

    def display_original_image(self):
        if self.original_image is not None :
            img = Image.fromarray(self.original_image)
            img = img.resize((400, 400))
            img_tk = ImageTk.PhotoImage(img)
            self.canvas_original_image.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.canvas_original_image.image = img_tk

    def display_modified_image(self):
        if self.modified_image is not None :
            img = Image.fromarray(self.modified_image)
            img = img.resize((400, 400))
            img_tk = ImageTk.PhotoImage(img)
            self.canvas_modified_image.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.canvas_modified_image.image = img_tk

    # Methode pour enregistrer une image
    def save_image(self):
        imgPath = filedialog.asksaveasfilename(initialdir="/", title="Select file",
                                          filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
        self.image.save(imgPath)
        print("L'image a été enregistrée dans :", imgPath)


    def function(self) :
        print("this is function")

    def erosion(self) :
        if self.original_image is not None :
            ES= np.ones((5,5),np.uint8)
            img = np.array(self.original_image)
            self.modified_image = cv2.erode(src=img,kernel=ES,iterations=1)
            self.display_modified_image()

    def dilatation(self) :
        if self.original_image is not None :
            ES= np.ones((5,5),np.uint8)
            img = np.array(self.original_image)
            self.modified_image = cv2.dilate(src=img,kernel=ES,iterations=1)
            self.display_modified_image()

root = tk.Tk()
app = VO_MainApplication(root)

root.mainloop()

