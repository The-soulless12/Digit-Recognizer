import tkinter as tk
from tkinter import messagebox
import numpy as np

GRID_SIZE = 28
CELL_SIZE = 20

class DrawingApp:
    def __init__(self, master):
        self.master = master
        master.title("🖌️ Reconnaissance de chiffres - MNIST")
        master.configure(bg="#f0f0f0")
        self.drawing = False
        self.cells = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        # Canvas central
        self.canvas = tk.Canvas(master, width=GRID_SIZE*CELL_SIZE, height=GRID_SIZE*CELL_SIZE,
                                bg="#ffffff", highlightthickness=0)
        self.canvas.pack(padx=20, pady=20)
        self.draw_grid()
        self.canvas.bind('<B1-Motion>', self.on_mouse_drag)

        # Footer frame pour les boutons
        self.footer = tk.Frame(master, bg="#f0f0f0")
        self.footer.pack(pady=10)

        self.start_button = tk.Button(self.footer, text="🖊️ Dessiner", command=self.enable_drawing,
                                      bg="#4CAF50", fg="white", padx=15, pady=5, relief="flat",
                                      activebackground="#45a049")
        self.start_button.grid(row=0, column=0, padx=10)

        self.clear_button = tk.Button(self.footer, text="🧹 Effacer", command=self.clear_grid,
                                      bg="#f44336", fg="white", padx=15, pady=5, relief="flat",
                                      activebackground="#d32f2f")
        self.clear_button.grid(row=0, column=1, padx=10)

        self.predict_button = tk.Button(self.footer, text="🤖 Prédire le chiffre", command=self.predict_digit,
                                        bg="#2196F3", fg="white", padx=15, pady=5, relief="flat",
                                        activebackground="#1976D2")
        self.predict_button.grid(row=0, column=2, padx=10)

        self.result_label = tk.Label(master, text="", bg="#f0f0f0", font=("Helvetica", 14))
        self.result_label.pack(pady=10)

        # Charger les modèles
        try:
            from tensorflow.keras.models import load_model
            self.model_3 = load_model("../model/Model-3x3-99.57.keras")
            self.model_5 = load_model("../model/Model-5x5-99.45.keras")
            self.model_7 = load_model("../model/Model-7x7-99.59.keras")
        except Exception as e:
            self.model_3 = self.model_5 = self.model_7 = None
            self.result_label.config(text=f"Erreur chargement modèles: {e}")

    def enable_drawing(self):
        self.drawing = True
        messagebox.showinfo("Info", "Mode dessin activé ! Tracez sur la grille avec le clic gauche.")

    def draw_grid(self):
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                x0 = j * CELL_SIZE
                y0 = i * CELL_SIZE
                x1 = x0 + CELL_SIZE
                y1 = y0 + CELL_SIZE
                self.canvas.create_rectangle(x0, y0, x1, y1, outline="#ccc", fill='white', tags=f"cell_{i}_{j}")

    def on_mouse_drag(self, event):
        if not self.drawing:
            return
        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE
        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
            self.cells[row, col] = 1
            self.canvas.itemconfig(f"cell_{row}_{col}", fill='black')

    def clear_grid(self):
        self.cells.fill(0)
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                self.canvas.itemconfig(f"cell_{i}_{j}", fill='white')
        self.result_label.config(text="")

    def predict_digit(self):
        if self.model_3 is None or self.model_5 is None or self.model_7 is None:
            self.result_label.config(text="Modèles non chargés.")
            return

        img = self.cells.reshape(1, 28, 28, 1)
        pred_3 = self.model_3.predict(img, verbose=0)
        pred_5 = self.model_5.predict(img, verbose=0)
        pred_7 = self.model_7.predict(img, verbose=0)
        ensemble_pred = (pred_3 + pred_5 + pred_7) / 3
        digit = np.argmax(ensemble_pred)
        self.result_label.config(text=f"Chiffre prédit (ensemble) : {digit}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()