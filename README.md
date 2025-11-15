# AI Chat Assistant avec Base de Connaissances PDF

Application de chat avec intelligence artificielle utilisant OpenRouter, avec support de base de connaissances PDF.

## Fonctionnalités

- **Chat IA**: Interface de chat moderne pour communiquer avec différents modèles LLM via OpenRouter
- **Base de Connaissances PDF**: Ajoutez des PDFs pour enrichir le contexte de l'IA
- **Configuration Flexible**: Configurez votre clé API et choisissez le modèle à utiliser
- **Interface Moderne**: GUI sombre et intuitive avec CustomTkinter

## Installation

1. Clonez ce dépôt:
```bash
git clone <url-du-repo>
cd AI723
```

2. Installez les dépendances:
```bash
pip install -r requirements.txt
```

## Configuration

### Obtenir une clé API OpenRouter

1. Visitez [OpenRouter](https://openrouter.ai/)
2. Créez un compte ou connectez-vous
3. Générez une clé API dans les paramètres
4. Copiez votre clé API

### Configurer l'application

1. Lancez l'application
2. Allez dans l'onglet "Configuration"
3. Entrez votre clé API OpenRouter
4. Choisissez un modèle (exemples):
   - `anthropic/claude-3.5-sonnet` (recommandé)
   - `openai/gpt-4-turbo`
   - `google/gemini-pro`
   - `meta-llama/llama-3.1-70b-instruct`
5. Cliquez sur "Sauvegarder la Configuration"

## Utilisation

### Lancer l'application

```bash
python app.py
```

### Onglet Chat

1. Tapez votre message dans la zone de texte
2. Cliquez sur "Envoyer" ou appuyez sur Ctrl+Entrée
3. L'assistant répondra en utilisant le modèle configuré
4. Le contexte des PDFs ajoutés sera automatiquement utilisé

### Onglet Base de Connaissances PDF

1. Cliquez sur "Ajouter un PDF"
2. Sélectionnez un fichier PDF
3. Le texte sera extrait et ajouté à la base de connaissances
4. Ce contenu sera automatiquement fourni au LLM comme contexte
5. Vous pouvez supprimer des PDFs à tout moment

### Raccourcis Clavier

- **Ctrl+Entrée**: Envoyer le message dans le chat

## Modèles Disponibles

Voici quelques modèles populaires disponibles via OpenRouter:

- **Claude 3.5 Sonnet**: `anthropic/claude-3.5-sonnet` - Excellent équilibre performance/coût
- **GPT-4 Turbo**: `openai/gpt-4-turbo` - Très performant pour des tâches complexes
- **Gemini Pro**: `google/gemini-pro` - Bon pour le raisonnement
- **Llama 3.1 70B**: `meta-llama/llama-3.1-70b-instruct` - Open source, performant

Pour voir la liste complète des modèles: https://openrouter.ai/models

## Fichiers de Configuration

L'application crée automatiquement deux fichiers:

- `config.json`: Stocke votre clé API et le modèle sélectionné
- `pdf_knowledge_base.json`: Stocke le contenu extrait des PDFs

## Structure du Projet

```
AI723/
├── app.py                      # Application principale
├── pdf_handler.py              # Gestion des PDFs
├── openrouter_client.py        # Client API OpenRouter
├── requirements.txt            # Dépendances
├── config.json                 # Configuration (généré)
└── pdf_knowledge_base.json     # Base de connaissances (généré)
```

## Dépannage

### Erreur "Veuillez configurer votre clé API"
- Allez dans l'onglet Configuration et entrez votre clé API OpenRouter

### Erreur de connexion API
- Vérifiez votre connexion internet
- Vérifiez que votre clé API est valide
- Vérifiez que le modèle spécifié existe sur OpenRouter

### PDF vide ou texte non extrait
- Certains PDFs sont des images et nécessitent l'OCR
- Essayez avec un autre PDF contenant du texte sélectionnable

## Technologies Utilisées

- **CustomTkinter**: Interface graphique moderne
- **PyPDF2**: Extraction de texte des PDFs
- **Requests**: Communication avec l'API OpenRouter
- **OpenRouter**: Gateway pour accéder à différents LLMs

## Licence

Ce projet est fourni tel quel pour usage personnel et éducatif.

## Support

Pour toute question ou problème, ouvrez une issue sur le dépôt GitHub.
