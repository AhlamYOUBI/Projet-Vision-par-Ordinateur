
import tkinter as tk
from tkinter import filedialog, Canvas, simpledialog
from PIL import Image, ImageTk
import PIL
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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
        transformation_menu.add_command(label='Negative', command=self.negative)
        transformation_menu.add_command(label='Rotation', command=self.rotation)
        transformation_menu.add_command(label='Redimension', command=self.redimensionner)
        transformation_menu.add_command(label='Rectangle', command=self.function)
        transformation_menu.add_command(label='Histogramme', command=self.histogramme)
        transformation_menu.add_command(label='Etirement', command=self.etirement)
        transformation_menu.add_command(label='Egalisation', command=self.egalisation)

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
        extraction_menu.add_command(label='Gradient', command=self.gradient)
        extraction_menu.add_command(label='Sobel', command=self.sobel)
        extraction_menu.add_command(label='Robert', command=self.robert)
        extraction_menu.add_command(label='Laplacien', command=self.laplacien)

        #SOUS MENU MORPHOLOGIE
        morphologie_menu = tk.Menu(menuBar, tearoff=0)
        menuBar.add_cascade(label='Morphologie', menu=morphologie_menu)
        morphologie_menu.add_command(label='Erosion', command=self.erosion)
        morphologie_menu.add_command(label='Dilatation', command=self.dilatation)
        morphologie_menu.add_command(label='Ouverture', command=self.ouverture)
        morphologie_menu.add_command(label='Fermeture', command=self.fermeture)
        morphologie_menu.add_command(label='Filtrage morphologique', command=self.function)

        #SOUS MENU SEGMENTATION
        segmentation_menu = tk.Menu(menuBar, tearoff=0)
        menuBar.add_cascade(label='Segmentation', menu=segmentation_menu)
        segmentation_menu.add_command(label='Croissance de regions D', command=self.croissance)
        segmentation_menu.add_command(label='Partition de regions D', command=self.function)
        segmentation_menu.add_command(label='k_means', command=self.kmeans)
  
        #SOUS MENU POINTS D INTERET
        ptsInteret_menu = tk.Menu(menuBar, tearoff=0)
        menuBar.add_cascade(label="Point d'interet", menu=ptsInteret_menu)
        ptsInteret_menu.add_command(label='xx', command=self.function)
        ptsInteret_menu.add_command(label='xx', command=self.function)
        ptsInteret_menu.add_command(label='xx', command=self.function)
  
        #SOUS MENU POINTS D INTERET
        compression_menu = tk.Menu(menuBar, tearoff=0)
        menuBar.add_cascade(label="Compression", menu=compression_menu)
        compression_menu.add_command(label='compression', command=self.compression)
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


    def ouverture(self) :
        if self.original_image is not None :
            ES = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
            img = np.array(self.original_image)
            self.modified_image = cv2.morphologyEx(img, cv2.MORPH_OPEN, ES)
            self.display_modified_image()


    def fermeture(self) :
        if self.original_image is not None :
            ES = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
            img = np.array(self.original_image)
            self.modified_image = cv2.morphologyEx(img, cv2.MORPH_CLOSE, ES)
            self.display_modified_image()


    def negative(self):
        if self.original_image is not None:
            img = np.array(self.original_image)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Appliquer le négatif
            img_negative = 255 - img

            self.modified_image = img_negative
            self.display_modified_image()

    def etirement(self, lower_pct=5, upper_pct=95):
        # l'étirement améliore le contraste local de l'image *en étirant ou réduisant* la plage des niveaux de gris utilisée.
        if self.original_image is not None:
            img = np.array(self.original_image)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Calculer les niveaux de gris minimum et maximum
            p_lower, p_upper = np.percentile(img, (lower_pct, upper_pct))
            # Appliquer la transformation linéaire
            img_etiree = (img - p_lower) * (255 / (p_upper - p_lower))
            # Clipper les valeurs en dehors de [0, 255]
            img_etiree = np.clip(img_etiree, 0, 255).astype(np.uint8)
            
            self.modified_image = img_etiree
            self.display_modified_image()


    def egalisation(self):
        #l'égalisation d'histogramme améliore le contraste global de l'image en répartissant les niveaux de gris *de manière uniforme*.
        if self.original_image is not None:
            img = np.array(self.original_image)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Appliquer l'égalisation d'histogramme
            img_egualise = cv2.equalizeHist(img)

            self.modified_image = img_egualise
            self.display_modified_image()


    def histogramme(self):
        if self.original_image is not None:
            if self.original_image is not None:
                img = np.array(self.original_image)
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            plt.plot(hist, color='black')
            plt.xlabel('Niveau de gris')
            plt.ylabel('Nombre de pixels')
            plt.show()

    def rotation(self):
        
        if self.original_image is not None:
            angle = simpledialog.askfloat("Rotation", "Entrez l'angle de rotation en degrés : ", parent=self.master)
            if angle is None:
                return
            rows, cols = self.original_image.shape[:2]
            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            self.modified_image = cv2.warpAffine(self.original_image, M, (cols, rows))
            self.display_modified_image()
############################## RAHMA ##########################################
    def sobel(self):
        if self.original_image is not None:
            sobel_x = cv2.Sobel(self.original_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(self.original_image, cv2.CV_64F, 0, 1, ksize=3)
            sx =cv2.convertScaleAbs(sobel_x)
            sy =cv2.convertScaleAbs(sobel_y)
            self.modified_image = cv2.addWeighted(sx, 0.5, sy, 0.5, 0)
            self.display_modified_image()

    def gradient(self):
        if self.original_image is not None:
            dx = cv2.Sobel(self.original_image, cv2.CV_64F, 1, 0 ,ksize=3) # Calculer le gradient en x
            dy = cv2.Sobel(self.original_image, cv2.CV_64F, 0, 1 ,ksize=3) # Calculer le gradient en y
            magnitude = np.sqrt(dx**2 + dy**2)
            magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            magnitude = np.uint8(magnitude)
            _, thresholded = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
            self.modified_image =thresholded
            self.display_modified_image()

    def robert(self):
        if self.original_image is not None:
            roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
            roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
            gradient_x = cv2.filter2D(self.original_image, cv2.CV_32F, roberts_x)
            gradient_y = cv2.filter2D(self.original_image, cv2.CV_32F, roberts_y)

            # Calculez le module du gradient
            magnitude = cv2.magnitude(gradient_x, gradient_y)
            # Vous pouvez ajuster la valeur de seuil en fonction de vos besoins
            seuil = 50
            _, seuil_img = cv2.threshold(magnitude, seuil, 255, cv2.THRESH_BINARY)
            self.modified_image =seuil_img
            self.display_modified_image()

    def laplacien(self):
        if self.original_image is not None:
            # Appliquez l'opérateur de Laplacien pour détecter les contours
            laplacian = cv2.Laplacian(self.original_image, cv2.CV_64F)

            # Convertissez l'image du Laplacien en image en entiers signés en utilisant la méthode absolute()
            laplacian = cv2.convertScaleAbs(laplacian)

            # Seuil de l'image du Laplacien
            seuil = 30
            _, seuil_img = cv2.threshold(laplacian, seuil, 255, cv2.THRESH_BINARY)
            self.modified_image =seuil_img
            self.display_modified_image()

    def croissance(self):
        if self.original_image is not None:
           return self.original_image
    
    def kmeans(self):
        if self.original_image is not None:
            # Redimensionner l'image (facultatif)
            image = cv2.resize(self.original_image, (800, 600))

            # Convertir l'image en une matrice de pixels
            pixels = image.reshape(-1, 3)

            # Effectuer la segmentation K-means
            kmeans = KMeans(n_clusters=8)  # Choisir le nombre de clusters souhaité
            kmeans.fit(pixels)
            labels = kmeans.labels_
            self.modified_image = kmeans.cluster_centers_[labels].reshape(image.shape)
            self.display_modified_image()
    
    def compression(self):
        if self.original_image is not None:
            self.modified_image='image_compressed.png'
            cv2.imwrite(self.modified_image, self.original_image, [cv2.IMWRITE_JPEG_QUALITY, 50])
            self.display_modified_image()


################################################################################
    def redimensionner(self):
        if self.original_image is not None:
            width = simpledialog.askinteger("Redimensionnement", "Entrez la nouvelle largeur : ", parent=self.master)
            height = simpledialog.askinteger("Redimensionnement", "Entrez la nouvelle hauteur : ", parent=self.master)

            self.resized_img = self.original_image.resize((width, height))
            self.display_resized_image()

    def display_resized_image(self):
        # Créer un widget Canvas pour afficher l'image redimensionnée
        self.canvas_modified_image = tk.Canvas(self.modified_image, width=self.resized_img.width, height=self.resized_img.height)
        self.canvas_modified_image.pack()

        # Convertir l'image en format Tkinter et l'afficher dans le widget Canvas
        tk_image = ImageTk.PhotoImage(self.resized_img)
        self.canvas_modified_image.create_image(0, 0, anchor=tk.NW, image=tk_image)
        self.canvas_modified_image.image = tk_image
################################################################################3

def similarite(pixel1, pixel2):
    # Calculer la différence entre les valeurs des canaux BGR pour les deux pixels
    diff_b = abs(int(pixel1[0]) - int(pixel2[0]))
    diff_g = abs(int(pixel1[1]) - int(pixel2[1]))
    diff_r = abs(int(pixel1[2]) - int(pixel2[2]))
    # Calculer la distance euclidienne entre les valeurs des canaux BGR pour les deux pixels
    distance = np.sqrt(diff_b**2 + diff_g**2 + diff_r**2)
    return distance




root = tk.Tk()
app = VO_MainApplication(root)

root.mainloop()


