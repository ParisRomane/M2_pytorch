# Projet Initiation aux réseaux de neuronnes via Torch


## Contenu du Projet
Le projet est organisé de la manière suivante : 

**notebooks** Dans se dossier ce trouve le premier notebook fait lors du TP, avant refactoring. Je le garde car il contient des détails dit en TP.

**models** Les définitions des différents modèles de réseaux de neurones.

**src** On défini ici tout les outils et scripts qui permettent de faire apprendre ou de tester le réseau de neuronne.

    **src/data_preprocessing** Défini l'ouverture et les preprocessing des données. (Données utilisées : MINST_fashion, )

    **src/training** Défini les fonctions et le script d'entrainement des données. 
    
    **src/evaluation** Défini les fonctions et le script de test et évaluation des données. 

**logs** Dossiers des logs où se trouve la sauvegarde des modèles et leur performances. 


## Installation 
Pour cloner le projet faire :

```bash
git clone https://github.com/ParisRomane/M2_pytorch.git
```


Après vous êtes mis dans l'environnement python de votre choix, les modules à installer peut être installés grâce à : 

```bash
pip install -r requirements.txt
```

## Utilisation
Pour lancer le TP 1 : 

```bash
./TP1.sh
```

Pour lancer le TP2.b : 
```bash
./TP2.sh
```