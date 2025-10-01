# Interface de dessin de chiffres avec grille (tkinter)
import tkinter as tk

GRID_SIZE = 28  # 28x28 pour correspondre au modèle MNIST
CELL_SIZE = 20  # Taille d'une case en pixels

class DrawingApp:
	def __init__(self, master):
		self.master = master
		master.title("Reconnaissance de chiffres - MNIST")
		self.drawing = False
		self.cells = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
		self.canvas = tk.Canvas(master, width=GRID_SIZE*CELL_SIZE, height=GRID_SIZE*CELL_SIZE, bg='white')
		self.canvas.pack(padx=10, pady=10)
		self.draw_grid()

		self.start_button = tk.Button(master, text="Start", command=self.enable_drawing)
		self.start_button.pack(pady=10)

		self.predict_button = tk.Button(master, text="Prédire le chiffre", command=self.predict_digit)
		self.predict_button.pack(pady=5)
		self.result_label = tk.Label(master, text="")
		self.result_label.pack(pady=5)

		# Charger le modèle Keras
		try:
			from tensorflow.keras.models import load_model
			self.model = load_model("../model/model.keras")
		except Exception as e:
			self.model = None
			self.result_label.config(text=f"Erreur chargement modèle: {e}")

	def enable_drawing(self):
		self.drawing = True

	def draw_grid(self):
		for i in range(GRID_SIZE):
			for j in range(GRID_SIZE):
				x0 = j * CELL_SIZE
				y0 = i * CELL_SIZE
				x1 = x0 + CELL_SIZE
				y1 = y0 + CELL_SIZE
				self.canvas.create_rectangle(x0, y0, x1, y1, outline='gray', fill='white', tags=f"cell_{i}_{j}")
		self.canvas.bind('<B1-Motion>', self.on_mouse_drag)

	def on_mouse_drag(self, event):
		if not self.drawing:
			return
		col = event.x // CELL_SIZE
		row = event.y // CELL_SIZE
		if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
			self.cells[row][col] = 1
			self.canvas.itemconfig(f"cell_{row}_{col}", fill='black')

	def predict_digit(self):
		if self.model is None:
			self.result_label.config(text="Modèle non chargé.")
			return
		import numpy as np
		# Convertir la grille en image 28x28
		img = np.array(self.cells, dtype=np.float32)
		img = img.reshape(1, 28, 28, 1)
		pred = self.model.predict(img)
		digit = np.argmax(pred)
		self.result_label.config(text=f"Chiffre prédit : {digit}")

if __name__ == "__main__":
	root = tk.Tk()
	app = DrawingApp(root)
	root.mainloop()
