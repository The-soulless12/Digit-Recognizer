# Digit-Recognizer
Application graphique en Python capable de reconnaître en temps réel les chiffres manuscrits dessinés à l’écran grâce à un ensemble de CNNs entraînés sur le dataset MNIST.

# Fonctionnalités
- Classification des 10 chiffres manuscrits via un ensemble de 3 CNNs entraînés sur MNIST avec une précision globale de 99,71 %.
- Interface interactive permettant la détection en temps réel du chiffre dessiné à l’écran.

# Structure du projet
- interface/ : Contient l’interface graphique de l’application.  
- model/ : Contient le notebook Python retraçant l’entraînement et la création des 3 CNNs.

# Prérequis
- Python 3.x  
- Packages nécessaires : numpy, tensorflow, keras, matplotlib, scikit-learn, pillow & tkinter.

# Note
- Pour exécuter le projet, saisissez la commande ``python main.py`` dans votre terminal à partir du répertoire ``interface/``.
- Les modèles préentraînés sont accessibles aux liens suivants : [model-3x3-99.57.keras](https://drive.google.com/file/d/1UQpg8H6RsAXj9AUaYk7y_OdiC9DVaD0Z/view?usp=sharing), [model-5x5-99.45.keras](https://drive.google.com/file/d/1vXbWTHgVp7emPNJcgA-oFmfCZEyaYXvF/view?usp=sharing) & [model-7x7-99.59.keras](https://drive.google.com/file/d/19yPCKLUE2a_w2PEe_g-u8lY9NKNe6fV2/view?usp=sharing) et pour assurer leur bon fonctionnement, ils doivent être déposés dans le répertoire ``model/``.
- Pour atteindre une précision finale de **99,71 %**, la technique utilisée repose sur l’apprentissage par ensemble de **3 CNNs distincts** dont les caractéristiques sont les suivantes :

<div align="center">
<table>
  <thead>
    <tr>
      <th>Modèle</th>
      <th>Couches convolutionnelles</th>
      <th>Filtres par couche</th>
      <th>Taille du noyau</th>
      <th>Padding</th>
      <th>Batch Normalization</th>
      <th>Couche de sortie</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><strong>CNN 3×3</strong></td><td>10</td><td>16 → 176</td><td>3×3</td><td>valid</td><td>Après chaque conv</td><td>Softmax (10)</td></tr>
    <tr><td><strong>CNN 5×5</strong></td><td>5</td><td>32 → 160</td><td>5×5</td><td>valid</td><td>Après chaque conv</td><td>Softmax (10)</td></tr>
    <tr><td><strong>CNN 7×7</strong></td><td>4</td><td>48 → 192</td><td>7×7</td><td>valid</td><td>Après chaque conv</td><td>Softmax (10)</td></tr>
  </tbody>
</table>
