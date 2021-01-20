# Run this script in order to use the GUI application
# Required libraries in your local python environment:
# tkinter, PIL, pandas, numpy, cv2, ghostscript, tensorflow

import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import pandas as pd
import PredictionModel
from tkinter import ttk

# Canvas Drawing Dimensions
drawW = 280
drawH= 280

# Canvas Colours
drawBg = "black"
drawCol = "white"

# Loading of Hiragana Class Characters
char_df = pd.read_csv('./classmap.csv',encoding='utf-8')['char']
char_df = np.asarray(char_df)

# Set column Width and Row Height of table
col_width = 35

class ExampleApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.testImage = None
        self.predictedClass = "    "
        self.predictedConfidence = "0.00"
        self.geometry('670x900')
        self.isImage = False
        
        # GUI Heading Label setup
        self.headingLabel = tk.Label(self,text="Classification of Handwritten Japanese Hiragana", font=("TKHeadingFont",16,"bold"))
        self.drawingLabel = tk.Label(self,text="Draw Character Here",font="TKHeadingFont")
        self.currentImageLabel = tk.Label(self,text="Current Image",font="TKHeadingFont")
        
        # Drawing Canvas Setup
        self.previous_x = self.previous_y = 0
        self.x = self.y = 0
        self.points_recorded = []
        self.canvas = tk.Canvas(self, width=drawW, height=drawH,bg="red", cursor="dot")
        self.canvas.create_rectangle(0,0,drawW+1,drawH+1,fill=drawBg)
        self.canvas.bind("<Motion>", self.tell_me_where_you_are)
        self.canvas.bind("<B1-Motion>", self.draw_from_where_you_are)
        
        #Button Definitions
        self.button_save = tk.Button(self, width= 10, text = "Save Drawing", command = self.save_canvas_to_image)
        self.button_clear = tk.Button(self, width=10, text = "Clear", command = self.clear_all)
        self.button_load = tk.Button(self, width= 10, text = "Load", command = self.loadImage)
        self.button_classify = tk.Button(self, width=10, text = "Classify", command = self.classify)
        
        # Image Display Setup
        self.imageLabel = tk.Label(self, width=38, height=18,bg=drawBg)
        
        # Classification Result Labels
        self.classificationLabel = tk.Label(self,text="Predicted Character",font=("TKDefaultFont",18,"bold"))
        self.predictionCharResultLabel = tk.Label(self,text=self.predictedClass, relief="ridge", font=("TKDefaultFont",28))
        self.predictionCharResultLabel.configure(text=self.predictedClass)
        
        # Classification Confidence Labels
        self.confidenceLabel = tk.Label(self,text="Prediction Confidence",font=("TKDefaultFont",18,"bold"))
        self.predictionCharConfidenceLabel = tk.Label(self,text=self.predictedConfidence, font=("TKDefaultFont",20))
        self.predictionCharConfidenceLabel.configure(text=self.predictedConfidence)
        
        # Hiragana Table Setup
        self.tableLabel = tk.Label(self,text="Hiragana Characters",font=("TKDefaultFont",18,"bold"))
        
        style = ttk.Style(self)
        style.configure('Treeview', rowheight=col_width)
               
        self.table = ttk.Treeview(self,columns=(1,2,3,4,5,6,7),show="tree",height=7)
        self.table.column("#0", width=0, minwidth=0, stretch=tk.NO)
        self.table.column(1, width=col_width, minwidth=col_width, stretch=tk.NO)
        self.table.column(2, width=col_width, minwidth=col_width, stretch=tk.NO)
        self.table.column(3, width=col_width, minwidth=col_width, stretch=tk.NO)
        self.table.column(4, width=col_width, minwidth=col_width, stretch=tk.NO)
        self.table.column(5, width=col_width, minwidth=col_width, stretch=tk.NO)
        self.table.column(6, width=col_width, minwidth=col_width, stretch=tk.NO)
        self.table.column(7, width=col_width, minwidth=col_width, stretch=tk.NO)
        for i in range(7):
            self.table.insert("",'end', values=(char_df[7*i],char_df[7*i+1],char_df[7*i+2],char_df[7*i+3],char_df[7*i+4],char_df[7*i+5],char_df[7*i+6]),tags='T')
        self.table.tag_configure('T', font='Arial 20')
        
        # Just a label for spacing purposes
        self.emptyLabel = tk.Label(self,height=1)
        
        # Placing all canvas widgets in grid to display in window
        self.headingLabel.grid(column=0,row=0, columnspan=5, rowspan=1, padx=5,pady=5)
        self.drawingLabel.grid(column=0,row=1, columnspan=2, rowspan=1, padx=5,pady=5)
        self.canvas.grid(column=0,row=2, columnspan=2, rowspan=2, padx=20,pady=0)
        self.button_save.grid(column=0,row=4, columnspan=1, rowspan=1, padx=0,pady=5)
        self.button_clear.grid(column=1,row=4, columnspan=1, rowspan=1, padx=0,pady=5)
        
        self.currentImageLabel.grid(column=3,row=1, columnspan=2, rowspan=1, padx=10,pady=0)
        self.imageLabel.grid(column=3,row=2, columnspan=2, rowspan=2, padx=0,pady=0, ipadx=0,ipady=0)
        self.button_load.grid(column=3,row=4, columnspan=1, rowspan=1, padx=0,pady=5)
        self.button_classify.grid(column=4,row=4, columnspan=1, rowspan=1, padx=0,pady=5)
        
        self.emptyLabel.grid(column=0,row=5, columnspan=5, rowspan=1, padx=0,pady=0)
        
        self.classificationLabel.grid(column=0,row=5, columnspan=7, rowspan=1, padx=0,pady=5)
        self.predictionCharResultLabel.grid(column=2,row=6, columnspan=1, rowspan=1, padx=0,pady=5)
        
        self.confidenceLabel.grid(column=0,row=7, columnspan=7, rowspan=1, padx=0,pady=5)
        self.predictionCharConfidenceLabel.grid(column=2,row=8, columnspan=1, rowspan=1, padx=0,pady=5)
        
        self.tableLabel.grid(column=0,row=9, columnspan=7, rowspan=1, padx=0,pady=5)
        self.table.grid(column=0,row=10, columnspan=7, rowspan=1, padx=0,pady=5)


    # Method determines where the mouse is on the canvas
    def tell_me_where_you_are(self, event):
        self.previous_x = event.x
        self.previous_y = event.y
    
    # Draw a line from the last mouse position to the current mouse position
    def draw_from_where_you_are(self, event):
        if self.points_recorded:
            self.points_recorded.pop()
            self.points_recorded.pop()

        self.x = event.x
        self.y = event.y
        self.canvas.create_line(self.previous_x, self.previous_y, 
                                self.x, self.y,fill=drawCol,width=3,smooth=1)
        self.points_recorded.append(self.previous_x)
        self.points_recorded.append(self.previous_y)
        self.points_recorded.append(self.x)     
        self.points_recorded.append(self.x)        
        self.previous_x = self.x
        self.previous_y = self.y

# Clear the drawing canvas and reset to initial state
    def clear_all(self):
        self.canvas.delete("all")
        self.points_recorded[:] = []
        self.canvas.create_rectangle(0,0,drawW+1,drawH+1,fill=drawBg)
    
# Save Drawing on canvas to image, and load that image
    def save_canvas_to_image(self):
        self.canvas.postscript(file="drawing.eps")
        imageToSave = Image.open("drawing.eps")
        imageToSave = imageToSave.resize((drawW+1,drawH+1))
        if(imageToSave):
            imageToSave.save("drawing.png","png")
            self.loadDrawImageToLabel()
    
    # Load a drawn image
    def loadDrawImageToLabel(self):
        self.openImg('./drawing.png')

# Load a supplied image to the label in GUI        
    def openImg(self,filename):
        # Load image and convert colors
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # COnvert to PIL image for resizing
        img = Image.fromarray(img)
        img = img.resize((drawW+1,drawH+1))
        
        # Convert image to image format for display in TKInter
        tkImg = ImageTk.PhotoImage(img)
        
        # Resize image to dimensions for input into the model for classification
        img = img.resize((28,28))
        self.isImage = True
        self.testImage = np.asarray(img)
        
        # Display loaded image to the GUI
        self.imageLabel.configure(image=tkImg,width=0,height=0)
        self.imageLabel.image = tkImg
        
        # Open dialogue box for image selection by user
    def loadImage(self):

        file= filedialog.askopenfilename(filetypes = (("Images files","*.png"),("Video Files","*.mp4"),("all files","*.*")), initialdir='./')
        
        if(len(file)>0):
            self.openImg(file)
        
# Method For button to call the prediction method of the prediction model script and display the predicted class and confidence
    def classify(self):
        if(self.isImage):
            result, confidence = PredictionModel.Predict(self.testImage)
            self.predictedClass = char_df[result]
            self.predictionCharResultLabel.configure(text=self.predictedClass)
            
            self.predictedConfidence = np.around(confidence,decimals=2)
            self.predictionCharConfidenceLabel.configure(text=self.predictedConfidence)
        else:
            messagebox.showerror("Invalid Input", "No image supplied for classification!")


if __name__ == "__main__":
    app = ExampleApp()
    app.mainloop()