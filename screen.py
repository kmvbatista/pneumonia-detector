import tkinter as tk
from tkinter import filedialog, Canvas, NW
from model import Model
from PIL import ImageTk,Image 

class Screen:
  def __init__(self, master : tk.Tk):
    self.model: Model = None
    self.photo_path = None
    self.master = master
    self.master.title("Qual foto você quer testar?")
    self.setUpInputs()

  
  def setUpInputs(self):
    self.setUpInputLabels()
    tk.Button(self.master,text='Escolher Modelo', command=self.get_model)\
      .grid(row=0, column=1, sticky=tk.W,pady=4)
    tk.Button(self.master,text='Carregar pesos', command=self.get_weights)\
      .grid(row=2, column=1, sticky=tk.W,pady=4)
    tk.Button(self.master,text='Escolher foto', command=self.get_photo)\
      .grid(row=3, column=1, sticky=tk.W,pady=4)


  def get_photo(self):
    filename = self.get_file_name(('jpeg', '*.jpeg'))
    tk.Label(self.master, text="Arquivo "+filename).grid(row=4)
    self.predict(filename)

  def predict(self, photo_path):
    print(photo_path)
    prediction = self.model.predict_image_from_path(photo_path)
    self.showResults(prediction)
    self.show_image(photo_path)

  def showResults(self, prediction):
    tk.Label(self.master, text=f"A imagem é {prediction}").grid(row=5)
    
  def show_image(self, imagePath):
    image = Image.open(imagePath).convert('RGB')
    image = image.resize((500, 400), Image.LINEAR)
    render = ImageTk.PhotoImage(image)
    img = tk.Label(self.master, image=render).grid(row=6)
    img.image = render
    img.place(x=0, y=0)

  def get_model(self):
    modelPath = filedialog.askdirectory()
    tk.Label(self.master, text="Modelo na pasta "+modelPath).grid(row=1)
    self.model = Model(modelPath)

  def get_weights(self):
    filename = self.get_file_name(('h5', '*.h5'))
    self.model.load_weights(filename)
     
  def setUpInputLabels(self):
    tk.Label(self.master, text="Qual modelo?").grid(row=0)
    tk.Label(self.master, text="Carregar pesos").grid(row=2)
    tk.Label(self.master, text="Qual foto?").grid(row=3)
  

  def get_file_name(self, fileType: tuple):
    return filedialog.askopenfilename(filetypes=(fileType, ('All files', '.*')))

root = tk.Tk()

Screen(root)

root.mainloop()
