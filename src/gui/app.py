import tkinter as tk
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as transforms


class App:

    def __init__(self, model):

        self.model = model  # The nueral network

        self.image = Image.new("L", (280, 280), "black")
        self.draw = ImageDraw.Draw(self.image)
    
        self.root = tk.Tk()
        self.root.title("Digit Recognizer")
        self.root.resizable(False, False)

        # canvas
        self.canvas = tk.Canvas(self.root, width=280, height=280, bg='white')
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.pack(pady=10)

        # button frame (horizontal row)
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=6)

        predict_btn = tk.Button(btn_frame, text="Predict", height=2, width=10, command=self.predict)
        predict_btn.pack(side="left", padx=10)

        clear_btn = tk.Button(btn_frame, text="Clear", height=2, width=10, command=self.clear)
        clear_btn.pack(side="left", padx=10)

        self.root.mainloop()

    def paint(self, event):
        x = event.x
        y = event.y

        r = 12

        # what the user see
        self.canvas.create_oval(
            x-r, y-r,
            x+r, y+r,
            fill="black",
            outline="black"
        )  

        # whats actually being drawn to the image
        self.draw.ellipse(
            [x-r, y-r, x+r, y+r],
            fill="white"
        )

    def clear(self):
        self.canvas.delete("all")

        self.image = Image.new("L", (280, 280), "black")
        self.draw = ImageDraw.Draw(self.image)

    def predict(self):
        img = self.crop_image().resize((28, 28))

        transform = transforms.ToTensor()

        img = transform(img).unsqueeze(0)  # [1,1,28,28]

        with torch.no_grad():
            outputs = self.model(img)

            digit = torch.argmax(outputs, dim=1).item()

        self.canvas.create_text(
            140, 30,
            text=f"The model predicted: {digit}",
            fill="red",
            font=("Arial", 18)
        )

    def crop_image(self):
        bbox = self.image.getbbox()
        if bbox:
            self.image.crop(bbox)
        else:
            self.image
       
        pad = 20
        w, h = self.image.size

        new_img = Image.new("L", (w + 2*pad, h + 2*pad), "black")
        new_img.paste(self.image, (pad, pad))

        return new_img
    
if __name__ == "__main__":
    app = App()

