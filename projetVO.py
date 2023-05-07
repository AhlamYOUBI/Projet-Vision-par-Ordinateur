
import tkinter as tk
from tkinter import filedialog, simpledialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tkinter import messagebox




class VO_MainApplication:
    def __init__(self, master):

        self.master = master
        self.master.title("Projet VO")
        self.frame = tk.Frame(self.master)

        self.image = None
        self.original_image = None
        self.modified_image = None
        self.resized_img = None
        

        #Creer un objet de menu et l'ajouter a fenetre
        menuBar = tk.Menu(self.master)
        self.master.config(menu= menuBar)
        #SOUS MENU IMAGE
        img_menu = tk.Menu(menuBar, tearoff=0)
        menuBar.add_cascade(label='Image', menu=img_menu)
        img_menu.add_command(label='Ouvrir', command=self.open_image)
        img_menu.add_command(label='Enregistrer', command=self.save_image)
        img_menu.add_separator()
        img_menu.add_command(label='Quitter', command=self.master.destroy)

        #SOUS MENU TRANSFORMATION
        transformation_menu = tk.Menu(menuBar, tearoff=0)
        menuBar.add_cascade(label='Transformation', menu=transformation_menu)
        transformation_menu.add_command(label='Negative', command=self.negative)
        transformation_menu.add_command(label='Rotation', command=self.rotation)
        transformation_menu.add_command(label='Redimension', command=self.redimensionner)
        transformation_menu.add_command(label='Rectangle', command=self.selection_image)
        transformation_menu.add_command(label='Histogramme NG', command=self.histogramme)
        transformation_menu.add_command(label='Histogramme RGB', command=self.histogrammeRGB)
        transformation_menu.add_command(label='Etirement', command=self.etirement)
        transformation_menu.add_command(label='Egalisation', command=self.egalisation)

        #SOUS MENU BINARISATION
        binarisation_menu = tk.Menu(menuBar, tearoff=0)
        menuBar.add_cascade(label='Binarisation', menu=binarisation_menu)
        binarisation_menu.add_command(label='Seuillage manuel', command=self.binarize_global)
        binarisation_menu.add_command(label='OTSU', command=self.binarize_otsu)
        binarisation_menu.add_command(label='Moyenne pondérée', command=self.binarize_weighted_mean)
        
        #SOUS MENU FILTRAGE
        filtrage_menu = tk.Menu(menuBar, tearoff=0)
        menuBar.add_cascade(label='Filtrage', menu=filtrage_menu)
        filtrage_menu.add_command(label='Gaussien', command=self.filter_gaussian)
        filtrage_menu.add_command(label='Moyenneur', command=self.filter_moyenneur)
        filtrage_menu.add_command(label='Median', command=self.filter_median)

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
        morphologie_menu.add_command(label='Filtrage Morphologique', command=self.filtrage_morphologique)

        #SOUS MENU SEGMENTATION
        segmentation_menu = tk.Menu(menuBar, tearoff=0)
        menuBar.add_cascade(label='Segmentation', menu=segmentation_menu)
        segmentation_menu.add_command(label='Croissance de regions D', command=self.croissanceD)
        segmentation_menu.add_command(label='Partition de regions D', command=self.partitionD)
        segmentation_menu.add_command(label='k_means', command=self.KMeansSegmentation)
  
        #SOUS MENU POINTS D INTERET
        ptsInteret_menu = tk.Menu(menuBar, tearoff=0)
        menuBar.add_cascade(label="Point d'interet", menu=ptsInteret_menu)
        ptsInteret_menu.add_command(label='hough_lines', command=self.hough_lines) 
        ptsInteret_menu.add_command(label='hough_circles', command=self.hough_circles)
        ptsInteret_menu.add_command(label='shi_tomasi', command=self.shi_tomasi)
  
        #SOUS MENU POINTS D INTERET
        compression_menu = tk.Menu(menuBar, tearoff=0)
        menuBar.add_cascade(label="Compression", menu=compression_menu)
        compression_menu.add_command(label='Huffman', command=self.function)
        compression_menu.add_command(label='LZW', command=self.function)
        compression_menu.add_command(label='Ondelette', command=self.function)

        # Création du bouton pour ouvrir une image
        self.button_open_image = tk.Button(master, text="Ouvrir une image", command=self.open_image)
        self.button_open_image.pack()

        # Création du bouton pour reinistialiser une image
        self.button_reset_image = tk.Button(master, text="Réinitialiser une image", command=self.reset_image)
        self.button_reset_image.pack()

        # Création des deux zones d'affichage pour les images
        self.canvas_original_image = tk.Canvas(master, width=400, height=400)
        self.canvas_original_image.pack(side=tk.LEFT, padx=10, pady=10)

        self.canvas_modified_image = tk.Canvas(master, width=400, height=400)
        self.canvas_modified_image.pack(side=tk.LEFT, padx=10, pady=10)



    def open_image(self):
        imgPath = filedialog.askopenfilename()
        if imgPath :
            self.original_image = cv2.imread(imgPath)
            self.display_original_image()
        

    def display_original_image(self):
        if self.original_image is not None :
            image = self.original_image
            image= cv2.resize(np.copy(image), (400, 400))
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            photo = ImageTk.PhotoImage(image_pil)
            self.canvas_original_image.delete("all")
            self.canvas_original_image.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas_original_image.image = photo


    def display_modified_image(self):
        if self.modified_image is not None :
            image = self.modified_image
            image= cv2.resize(np.copy(image), (400, 400))
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            photo = ImageTk.PhotoImage(image_pil)
            self.canvas_modified_image.delete("all")
            self.canvas_modified_image.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas_modified_image.image = photo


    def save_image(self):
        imgPath = filedialog.asksaveasfilename(initialdir="/", title="Select file",
                                          filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
        self.image.save(imgPath)
        print("L'image a été enregistrée dans :", imgPath)


    def erosion(self):
        if self.original_image is not None:
            gray = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2GRAY)
            ES = np.ones((5,5), np.uint8)
            eroded = cv2.erode(src=gray, kernel=ES, iterations=1)
            self.modified_image = eroded
            self.display_modified_image()


    def dilatation(self) :
        if self.original_image is not None :
            gray = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2GRAY)
            ES= np.ones((5,5),np.uint8)
            dilated = cv2.dilate(src=gray, kernel=ES, iterations=1)
            self.modified_image = cv2.cvtColor(dilated, cv2.COLOR_GRAY2RGB)
            self.display_modified_image()


    def ouverture(self):
        if self.original_image is not None:
            gray = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2GRAY)
            ES = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
            opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, ES)
            self.modified_image = opening
            self.display_modified_image()


    def fermeture(self):
        if self.original_image is not None:
            gray = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2GRAY)
            ES = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
            img = np.array(gray)
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
        if self.original_image is not None:
            img = np.array(self.original_image)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            p_lower, p_upper = np.percentile(img, (lower_pct, upper_pct))
            img_etiree = (img - p_lower) * (255 / (p_upper - p_lower))
            img_etiree = np.clip(img_etiree, 0, 255).astype(np.uint8)
            self.modified_image = img_etiree
            self.display_modified_image()


    def egalisation(self):
        if self.original_image is not None:
            img = np.array(self.original_image)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_egualise = cv2.equalizeHist(img)

            self.modified_image = img_egualise
            self.display_modified_image()


    def histogramme(self):
        if self.original_image is not None:
            img = np.array(self.original_image)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                hist = cv2.calcHist([img], [0], None, [256], [0, 256])
                plt.plot(hist, color='black')
                plt.xlabel('Niveau de gris')
                plt.ylabel('Nombre de pixels')
                plt.show()

    
    def histogrammeRGB(self):
        if self.original_image is not None:
            img = np.array(self.original_image)
            colors = ('b', 'g', 'r')
            plt.figure()
            for i, col in enumerate(colors):
                hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                plt.plot(hist, color=col)
            plt.xlim([0, 256])
            plt.xlabel('Couleur RGB')
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


    def sobel(self):
        if self.original_image is not None:
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sx = cv2.convertScaleAbs(sobel_x)
            sy = cv2.convertScaleAbs(sobel_y)
            self.modified_image = cv2.addWeighted(sx, 0.5, sy, 0.5, 0)
            self.display_modified_image()


    def gradient(self):
        if self.original_image is not None:
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0 ,ksize=3)
            dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1 ,ksize=3)
            magnitude = np.sqrt(dx**2 + dy**2)
            magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            magnitude = np.uint8(magnitude)
            _, thresholded = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
            self.modified_image = thresholded
            self.display_modified_image()


    def robert(self):
        if self.original_image is not None:
            gray = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2GRAY)
            roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
            roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
            gradient_x = cv2.filter2D(gray, cv2.CV_32F, roberts_x)
            gradient_y = cv2.filter2D(gray, cv2.CV_32F, roberts_y)

            # Calculez le module du gradient
            magnitude = cv2.magnitude(gradient_x, gradient_y)
            seuil = 50
            _, seuil_img = cv2.threshold(magnitude, seuil, 255, cv2.THRESH_BINARY)
            self.modified_image =seuil_img
            self.display_modified_image()


    def robert(self):
        if self.original_image is not None:
            gray = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2GRAY)
            roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
            roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
            gradient_x = cv2.filter2D(gray, cv2.CV_32F, roberts_x)
            gradient_y = cv2.filter2D(gray, cv2.CV_32F, roberts_y)

            # Calculez le module du gradient
            magnitude = cv2.magnitude(gradient_x, gradient_y)
            seuil = 50
            _, seuil_img = cv2.threshold(magnitude, seuil, 255, cv2.THRESH_BINARY)
            self.modified_image =seuil_img
            self.display_robert_image()

    def display_robert_image(self):
        if self.modified_image is not None:
            # Convertir le tableau numpy en uint8 avant de le convertir en image PIL
            image_pil = Image.fromarray(cv2.cvtColor(self.modified_image.astype('uint8'), cv2.COLOR_BGR2RGB))
            self.image_tk_modified = ImageTk.PhotoImage(image_pil)
            photo = self.image_tk_modified
            self.canvas_modified_image.delete("all")
            self.canvas_modified_image.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas_modified_image.image = photo


    def laplacien(self):
        if self.original_image is not None:
            gray = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian = cv2.convertScaleAbs(laplacian)
            seuil = 30
            _, seuil_img = cv2.threshold(laplacian, seuil, 255, cv2.THRESH_BINARY)
            self.modified_image = seuil_img
            self.display_modified_image()

    def filtrage_morphologique(self):
        if self.original_image is not None:
            gray = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2GRAY)
            kernel = np.ones((3,3),np.uint8)
            morph = cv2.morphologyEx(gray,cv2.MORPH_OPEN, kernel)
            self.modified_image = morph
            self.display_modified_image()


    def croissanceD(self):
            seed = (50,50)
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            mask = np.zeros_like(gray)
            points_to_add = [seed]
            while len(points_to_add) > 0:
                pt = points_to_add[0]
                points_to_add = points_to_add[1:]
                if pt[0] >= 0 and pt[0] < gray.shape[1] and pt[1] >= 0 and pt[1] < gray.shape[0] and mask[pt[1], pt[0]] == 0:
                    if abs(int(gray[pt[1], pt[0]]) - int(gray[seed[1], seed[0]])) <= 50:
                        mask[pt[1], pt[0]] = 255
                        points_to_add.append((pt[0] + 1, pt[1]))
                        points_to_add.append((pt[0] - 1, pt[1]))
                        points_to_add.append((pt[0], pt[1] + 1))
                        points_to_add.append((pt[0], pt[1] - 1))
            self.image_traitee = mask
            self.modified_image = self.image_traitee
            self.display_modified_image()


    def partitionD(self):
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown==255] = 0
        
        markers = cv2.watershed(self.original_image, markers)
        image_out = np.zeros_like(self.original_image)
        image_out[markers == -1] = [255,0,0]  # Marquer les bords avec une couleur rouge
        
        self.modified_image = image_out
        self.display_modified_image()


    def KMeansSegmentation(self):
        if self.original_image is not None:
            img = np.array(self.original_image)
            
            Z = img.reshape((-1,3))
            Z = np.float32(Z)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            K = 8
            ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

            center = np.uint8(center)
            res = center[label.flatten()]
            res2 = res.reshape((img.shape))
            
            self.modified_image = res2
            self.display_modified_image()


    def reset_image(self):
        if self.original_image is not None:
            self.modified_image = self.original_image.copy()
            self.display_modified_image()


    def selection_image(self):
        self.modified_image = None
        self.resized_img = None
        if self.original_image.size > 0:
            r1 = messagebox.showinfo("Sélection", 'Veuillez sélectionner une région et cliquer sur le bouton "espace" our "entrer" pour voir la région sélectionnée')
            if r1 == "ok":
                roi = cv2.selectROI(self.original_image)
                self.modified_image = self.original_image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
                self.display_modified_image()
        else:
            messagebox.showerror("Erreur", "Veuillez choisir une image!")


    def hough_lines(self):
        if self.original_image is not None:
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 200, apertureSize=3)

            lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=100)

            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(self.original_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            self.modified_image = self.original_image
            self.display_modified_image()


    def hough_circles(self):
        if self.original_image is not None:
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=30, minRadius=0, maxRadius=0)

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    cv2.circle(self.original_image, (x, y), r, (0, 255, 0), 2)

            self.modified_image = self.original_image
            self.display_modified_image()


    def shi_tomasi(self):
        if self.original_image is not None:
            gray = cv2.cvtColor(self.original_image , cv2.COLOR_BGR2GRAY)
            corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
            corners = np.int0(corners)
            for corner in corners:
                x,y = corner.ravel()
                cv2.circle(self.original_image,(x,y),3,(0,0,255),-1)
            self.image_traitee = self.original_image
            self.display_modified_image()

    def redimensionner(self):
        self.modified_image = None
        if self.original_image is not None:
            width = simpledialog.askinteger("Redimensionnement", "Entrez la nouvelle largeur : ", parent=self.master)
            height = simpledialog.askinteger("Redimensionnement", "Entrez la nouvelle hauteur : ", parent=self.master)

            img = Image.fromarray(self.original_image)
            self.resized_img = img.resize((width, height))
            print("New image dimensions:", self.resized_img.size)
            self.display_resized_image()


    def display_resized_image(self):        
        self.canvas_modified_image.delete("all")

        if self.resized_img is not None:
            width, height = self.resized_img.size

            self.canvas_modified_image = tk.Canvas(self.modified_image, width=width, height=height)
            self.canvas_modified_image.pack()

            tk_image = ImageTk.PhotoImage(self.resized_img)
            self.canvas_modified_image.create_image(0, 0, anchor=tk.NW, image=tk_image)
            self.canvas_modified_image.image = tk_image


    def filter_gaussian(self):
        if self.original_image is not None :
            sigma = simpledialog.askfloat("Filtre Gaussien", "Entrez la valeur de l'écart type (entre 0.5 et 5.0) :", parent=self.master)
            self.modified_image = cv2.GaussianBlur(self.original_image, (0, 0), sigmaX=sigma, sigmaY=sigma)
            self.display_modified_image()


    def filter_moyenneur(self):
        if self.original_image is not None:
            filter_size = simpledialog.askinteger("Filtre Moyenneur", "Entrez la taille du filtre (nombre impair >3) :", parent=self.master)
            self.modified_image = cv2.blur(self.original_image, (filter_size, filter_size))
            self.display_modified_image()


    def filter_median(self):
        if self.original_image is not None:
            filter_size = simpledialog.askinteger("Filtre Médian", "Entrez la taille du filtre (nombre impair) :", parent=self.master)
            self.modified_image = cv2.medianBlur(self.original_image, filter_size)
            self.display_modified_image()


    def binarize_global(self):
        if self.original_image is not None:
            threshold = simpledialog.askinteger("Seuillage manuel", "Entrez la valeur de seuil (entre 0 et 255) :", parent=self.master)
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
            self.modified_image = binary_image
            self.display_modified_image()


    def binarize_otsu(self):
         if self.original_image is not None:
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.modified_image = binary_image
            self.display_modified_image()


    def binarize_weighted_mean(self):
        if self.original_image is not None:
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            threshold_value = cv2.mean(gray_image)[0]
            ret, self.modified_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
            self.display_modified_image()


    def function(self) :
        print("this is function")








root = tk.Tk()

root.iconbitmap('icone.ico')

app = VO_MainApplication(root)

root.mainloop()


