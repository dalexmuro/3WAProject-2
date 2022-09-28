## main.py
Code original fourni.

Utilise les datasets _**tmdb_5000_credits.csv**_ et _**tmdb_5000_credits.csv**_.

<br/>

## sent_analysis.ipynb
Modèle d'analyse de sentiments.

Utilise le dataset _**imdb_dataset.csv**_ et crée le fichier _**cleaned_reviews_sentiment.plk**_.

Crée et sauvegarde le modèle _**model_sentiment.sav**_.

<br/>

## cleantext.py
Méthodes **cleanhtml** et **cleantext**, qui permettent de nettoyer le texte des reviews.

<br/>

## get_data.py
Ouvre le fichier _**full_data.plk**_ if it exists.
Otherwise, Utilise le dataset _**mpst_full_data.csv**_, 
recupère les informations manquantes à partir de l'API, traite les reviews en utilisant le modèle
_**model_recommendation.plk**_ et crée le fichier _**full_data.plk**_

<br/>

## sujet_2.ipynb
Récuperation et calcul des reviews et ratings.

Utilise le dataset _**mpst_full_data.csv**_ et crée les fichiers _**movies.plk**_,  et _**sig.plk**_.

<br/>

## final_main.py
Code modifié. Point d'entrée

Utilise le dataset _**full_data.plk**_ et le modèmles _**tfidf_overview_reccommendation.plk**_, _**tfidf_scores_reccommendation.plk**_ et _**sig_reccommendation.plk**_, ou les crée s'ils n'existent pas.

