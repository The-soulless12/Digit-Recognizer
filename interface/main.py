import tkinter as tk
import numpy as np

GRID_SIZE = 28
CELL_SIZE = 15

class DrawingApp:
    def __init__(self, master):
        self.master = master
        master.title("🖌️ Reconnaissance de chiffres - MNIST")
        master.configure(bg="#f0f0f0")

        self.drawing = True
        self.cells = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.predict_after_id = None  

        master.resizable(False, False)

        # --- Topbar avec image effacer --- #
        self.topbar = tk.Frame(master, bg="#f0f0f0", height=1)
        self.topbar.pack(fill="x", pady=(2,0), ipady=0)
        self.topbar.grid_columnconfigure(0, weight=1)

        from PIL import Image, ImageTk
        img = Image.open("../interface/effacer.png").resize((32, 32), Image.Resampling.LANCZOS).convert("RGBA")

        # légère ombre réduite
        from PIL import ImageFilter
        shadow = Image.new("RGBA", (36, 36), (0,0,0,0))
        mask = img.split()[-1]
        shadow.paste((0,0,0,90), (2,2), mask)
        shadow.paste(img, (0,0), img)

        self.effacer_img = ImageTk.PhotoImage(shadow)

        self.clear_button = tk.Button(
            self.topbar, command=self.clear_grid, image=self.effacer_img,
            borderwidth=0, highlightthickness=0, bg="#f0f0f0", activebackground="#f0f0f0",
            relief="flat", cursor="hand2"
        )
        self.clear_button.grid(row=0, column=1, sticky="e", padx=(0, 20))

        # --- Grille de dessin --- #
        self.canvas = tk.Canvas(master, width=GRID_SIZE*CELL_SIZE, height=GRID_SIZE*CELL_SIZE,
                                bg="#ffffff", highlightthickness=0)
        self.canvas.pack(padx=20, pady=10)
        self.draw_grid()
        self.canvas.bind('<B1-Motion>', self.on_mouse_drag)

        # --- Zone prédictions --- #
        self.bottom_frame = tk.Frame(master, bg="#f0f0f0")
        self.bottom_frame.pack(pady=10)

        self.pred_canvas = tk.Canvas(self.bottom_frame, width=10*40, height=60, bg="#f0f0f0", highlightthickness=0)
        self.pred_canvas.pack(side="left")
        self.pred_boxes = []
        for i in range(10):
            x0, y0 = i*40, 0
            x1, y1 = x0+40, 40
            box = self.pred_canvas.create_rectangle(x0, y0, x1, y1, fill="white", outline="black", width=2)
            self.pred_canvas.create_text(x0+20, y1+15, text=str(i), font=("Comic Sans MS", 11))
            self.pred_boxes.append(box)

        # Charger les modèles
        try:
            from tensorflow.keras.models import load_model
            self.model_3 = load_model("../model/Model-3x3-99.57.keras")
            self.model_5 = load_model("../model/Model-5x5-99.45.keras")
            self.model_7 = load_model("../model/Model-7x7-99.59.keras")
        except Exception as e:
            self.model_3 = self.model_5 = self.model_7 = None
            print(f"Erreur chargement modèles: {e}")

    def draw_grid(self):
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                x0 = j * CELL_SIZE
                y0 = i * CELL_SIZE
                x1 = x0 + CELL_SIZE
                y1 = y0 + CELL_SIZE
                self.canvas.create_rectangle(x0, y0, x1, y1, outline="#ccc", fill='white', tags=f"cell_{i}_{j}")

    def on_mouse_drag(self, event):
        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE
        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
            self.cells[row, col] = 1
            self.canvas.itemconfig(f"cell_{row}_{col}", fill='black')
            if self.predict_after_id is not None:
                self.master.after_cancel(self.predict_after_id)
            self.predict_after_id = self.master.after(300, self.predict_digit)

    def clear_grid(self):
        self.cells.fill(0)
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                self.canvas.itemconfig(f"cell_{i}_{j}", fill='white')
        for box in self.pred_boxes:
            self.pred_canvas.itemconfig(box, fill="white")

    def predict_digit(self):
        if self.model_3 is None or self.model_5 is None or self.model_7 is None:
            return

        img = self.cells.reshape(1, 28, 28, 1)

        pred_3 = self.model_3.predict(img, verbose=0)
        pred_5 = self.model_5.predict(img, verbose=0)
        pred_7 = self.model_7.predict(img, verbose=0)

        ensemble_pred = (pred_3 + pred_5 + pred_7) / 3
        probs = ensemble_pred.flatten()

        # reset
        for box in self.pred_boxes:
            self.pred_canvas.itemconfig(box, fill="white")

        # top1 uniquement
        best_idx = int(np.argmax(probs))
        self.pred_canvas.itemconfig(self.pred_boxes[best_idx], fill="#ff8ed6")  # rose pastel


if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
