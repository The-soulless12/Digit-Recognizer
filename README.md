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
      <th align="center">Modèle du CNN</th>
      <th align="center">Couches convolutionnelles</th>
      <th align="center">Filtres par couche</th>
      <th align="center">Taille du noyau</th>
      <th align="center">Padding</th>
      <th align="center">Batch Normalization</th>
      <th align="center">Couche de sortie</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="center"><strong>CNN 3×3</strong></td><td align="center">10</td><td align="center">16 → 176</td><td align="center">3×3</td><td align="center">valid</td><td align="center">Après chaque conv</td><td align="center">Softmax (10)</td></tr>
    <tr><td align="center"><strong>CNN 5×5</strong></td><td align="center">5</td><td align="center">32 → 160</td><td align="center">5×5</td><td align="center">valid</td><td align="center">Après chaque conv</td><td align="center">Softmax (10)</td></tr>
    <tr><td align="center"><strong>CNN 7×7</strong></td><td align="center">4</td><td align="center">48 → 192</td><td align="center">7×7</td><td align="center">valid</td><td align="center">Après chaque conv</td><td align="center">Softmax (10)</td></tr>
  </tbody>
</table>
</div>

- Chaque modèle a été entraîné sur un maximum de **150 epochs** sur Google Colab avec une accélération GPU T4, en utilisant un batch size de 120 (500 itérations par epochs) ainsi que deux callbacks. Le EarlyStopping (patience = 12 et basé sur la val_accuracy) et le LearningRateScheduler qui réduit le taux d’apprentissage de 2 % à chaque epoch. Le tableau suivant résume les performances obtenues pour chaque modèle :

<div align="center">
<table>
  <thead>
    <tr>
      <th align="center">Modèle</th>
      <th align="center">Époque arrêtée (EarlyStopping)</th>
      <th align="center">Temps d’exécution (s)</th>
      <th align="center">Accuracy (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="center"><strong>CNN 3×3</strong></td><td align="center">46</td><td align="center">1116,77</td><td align="center">99,57</td></tr>
    <tr><td align="center"><strong>CNN 5×5</strong></td><td align="center">36</td><td align="center">790,39</td><td align="center">99,45</td></tr>
    <tr><td align="center"><strong>CNN 7×7</strong></td><td align="center">50</td><td align="center">1043,47</td><td align="center">99,59</td></tr>
  </tbody>
</table>
</div>

- L’accuracy globale a été ensuite calculée en faisant la moyenne des vecteurs de sortie softmax des trois modèles tout en comparant la classe prédite finale avec les labels réels.
