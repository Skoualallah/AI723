# Installation de Tesseract OCR pour Windows

Pour utiliser la fonctionnalité OCR de l'application, vous devez installer Tesseract OCR sur votre système Windows.

## Installation de Tesseract

### Méthode 1 : Installation depuis le site officiel

1. Téléchargez l'installateur Tesseract pour Windows depuis :
   https://github.com/UB-Mannheim/tesseract/wiki

2. Choisissez la version appropriée (généralement la dernière version 64-bit)

3. Exécutez l'installateur et suivez les instructions

4. **IMPORTANT** : Pendant l'installation, assurez-vous de :
   - Installer Tesseract dans le chemin par défaut : `C:\Program Files\Tesseract-OCR`
   - Cocher l'option pour installer les données de langue française (French language data)

### Méthode 2 : Installation avec Chocolatey

Si vous utilisez Chocolatey, vous pouvez installer Tesseract avec :

```bash
choco install tesseract
```

## Installation des dépendances Python

Installez les dépendances Python nécessaires :

```bash
pip install -r requirements.txt
```

## Vérification de l'installation

Pour vérifier que Tesseract est correctement installé :

1. Ouvrez une invite de commande
2. Tapez : `tesseract --version`
3. Vous devriez voir la version de Tesseract affichée

## Langues supportées

Par défaut, l'application utilise le français ('fra') pour l'OCR. Si vous souhaitez utiliser une autre langue :

- L'application est configurée pour utiliser le français par défaut
- Les données de langue française doivent être installées avec Tesseract
- Pour ajouter d'autres langues, téléchargez les fichiers de données de langue depuis :
  https://github.com/tesseract-ocr/tessdata

## Utilisation dans l'application

Une fois Tesseract installé, vous pouvez utiliser les fonctionnalités OCR :

1. **📷 Image** : Cliquez sur ce bouton pour sélectionner une image depuis votre ordinateur
2. **📋 OCR** : Cliquez sur ce bouton pour extraire le texte d'une image copiée dans le presse-papier

Le texte extrait sera automatiquement inséré dans la zone de texte du chat.

## Résolution des problèmes

### Erreur : "Tesseract not found"

Si vous obtenez cette erreur :

1. Vérifiez que Tesseract est installé dans `C:\Program Files\Tesseract-OCR`
2. Si installé ailleurs, modifiez le chemin dans `ocr_handler.py`
3. Redémarrez l'application après l'installation

### Aucun texte extrait

Si aucun texte n'est extrait de l'image :

1. Assurez-vous que l'image contient du texte lisible
2. Vérifiez que les données de langue française sont installées
3. Essayez avec une image de meilleure qualité ou plus grande résolution

## Formats d'images supportés

- PNG
- JPEG/JPG
- BMP
- GIF
- TIFF
