#!/bin/bash

# Nom de l'environnement Conda
ENV_NAME="dip-tv-py-311"

# Vérifier si l'environnement Conda existe
if conda info --envs | grep -q "^$ENV_NAME"; then
    echo "L'environnement Conda '$ENV_NAME' existe déjà. Activation de l'environnement..."
else
    echo "Création de l'environnement Conda '$ENV_NAME' avec Python 3.11..."
    conda create -n $ENV_NAME python=3.11 -y || { echo "Erreur lors de la création de l'environnement."; return 1; }
fi

# Activer l'environnement
echo "Activation de l'environnement '$ENV_NAME'..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME || { echo "Impossible d'activer l'environnement '$ENV_NAME'."; return 1; }

# Vérifier l'existence du fichier requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Installation des dépendances à partir de requirements.txt..."
    pip install uv
    uv pip install -r requirements.txt || { echo "Erreur lors de l'installation des dépendances."; return 1; }
else
    echo "Le fichier requirements.txt est introuvable. Vérifiez votre dossier."
    return 1
fi

# Installation du package en mode editable
echo "Installation du package en mode editable..."
pip install -e . || { echo "Erreur lors de l'installation du package."; return 1; }

# # Annonce des tests
# echo "Tests en cours d'exécution..."

# # Lancer les tests avec Python
# if [ -f "tests/quick/test_suite.py" ]; then
#     python -m tests.quick.test_suite || { echo "Erreur lors de l'exécution des tests."; return 1; }
#     python -m tests.complete.test_suite || { echo "Erreur lors de l'exécution des tests."; return 1; }
#     echo "Tests terminés avec succès."
# else
#     echo "Le fichier Tests/test.py est introuvable. Vérifiez votre dossier."
#     return 1
# fi

# Conclusion
echo "Script terminé avec succès."