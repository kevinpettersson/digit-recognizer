import tkinter as tk

class App:

    def __init__(self):
    
        self.root = tk.Tk()
        self.root.title("Digit Recognizer")

        self.canvas = tk.Canvas(self.root, width=640, height=480, bg='white')
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.pack()

        button = tk.Button(self.root, text="Predict", height=2, width=8)
        button.bind('<Button-1>', self.predict)
        button.pack(anchor="w")

        button = tk.Button(self.root, text="Clear", height=2, width=8)
        button.bind('<Button-1>', self.clear)
        button.pack(anchor="e")

        self.root.mainloop()

    def paint(self, event):
        x = event.x
        y = event.y
        self.canvas.create_line(x, y, x+1, y+1, width=5)     

    def clear(self, event):
        print("clear")

    def predict(self, event):
        print("pred")
        
        


if __name__ == "__main__":
    app = App()

