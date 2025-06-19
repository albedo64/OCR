import keras_ocr
import matplotlib.pyplot as plt
import tensorflow as tf
from spellchecker import SpellChecker
from fuzzywuzzy import fuzz

class OCR() :
  def __init__(self, language='fr'):
    self.pipeline = keras_ocr.pipeline.Pipeline()
    self.spell = SpellChecker(language=language)
    self.fuzzy = fuzz

    # Prend le chemin d'accès d'une image en entrée
  def read_images(self, image_path):
    images = [keras_ocr.tools.read(image_path)]
    return images

    #Pour la reconnaissance de texte
  def recognize_text(self, images):
    prediction_groups = self.pipeline.recognize(images)
    return prediction_groups


    """
        Permet d'afficher l'image avec les différentes valeurs prédites.
        Elle prend en entrée l'image lue par 'read_image' et les prédictions
    """
  def plot_predictions(self, images, prediction_groups):
      fig, axs = plt.subplots(nrows=len(images), figsize=(10, 20))
      if len(images) == 1:
          axs = [axs]
      for ax, image, predictions in zip(axs, images, prediction_groups):
          keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)

    #Fonction permettant d'obtenir que les prédictions ou textes prédits par le modèle
  def get_text(self, prediction_groups_part):
    texts = []
    info_image = prediction_groups_part
    for text, box in info_image :
      texts.append(text)
    return texts


    """Fonction qui calcule le niveau de différence entre deux mots.
        Utilisée pour determiner le niveau de précision de la correction orthographique
    """
  def similarity_word(self, word, corrected_word):
    return self.fuzzy.ratio(word, corrected_word) / 100.0


    """
        Permet la correction orthographique d'un ensemble de mots de la variable 'text'.
        Prend en entrée le texte entier et la langue de correction ('fr' ou 'en')
        Donne en sortie le texte corrigé et le niveau de précision de la correction
    """
  def correct_spelling(self, text, language):
      words = text.split()
      corrected_words = []
      similarity_words = []

      for word in words:
          corrected_word = self.spell.correction(word)
          similarity = self.similarity_word(word, corrected_word)

          # Vérifie si le mot corrigé n'est pas None et est de type str
          if corrected_word is not None and isinstance(corrected_word, str):
            corrected_words.append(corrected_word)
            similarity_words.append(similarity)


      corrected_text = " ".join(corrected_words)
      if corrected_text == '' :
        similarity = 0
      else :
        similarity = max(similarity_words)
      return corrected_text, similarity


    """
        Permet d'obtenir une liste de tuple de la forme (i, j, k):
            - i : La valeure prédite par le modele
            - j : Sa valeure corrigée
            - k : Le niveau de précidion de la correction
    """
  def get_all(self, texts):
    Mots = []
    score = []
    donnees = []

    for i in range(len(texts)) :
      corrected_text_1, similarity_score_1 = self.correct_spelling(texts[i], 'en')  
      corrected_text_2, similarity_score_2 = self.correct_spelling(texts[i], 'fr')

      if similarity_score_1 > similarity_score_2 :
        corrected_text = corrected_text_1
        similarity_score = similarity_score_1
      else :
        corrected_text = corrected_text_2
        similarity_score = similarity_score_2

      Mots.append(corrected_text)
      score.append(similarity_score)

    for i,j,k in zip(texts, Mots, score) :
      lien = (i, j, k)
      donnees.append(lien)

    return donnees

    """
        Faire le tout en un. 
        Du path de l'image à l'ensemble de données sur la prédiction, la correction (en 'en' et 'fr') et la précision.
        - 'show' est une variable booleenne qui permet d'afficher ou non l'image avec les prédictions dessus
    """
  def all_in_one_OCR(self, image_path, show):
    all_infos = {}
    images = self.read_images(image_path)
    prediction_groups = self.recognize_text(images)
    if show == True :
      self.plot_predictions(images, prediction_groups)
    n = len(prediction_groups)
    for i in range(n) :
      texts = self.get_text(prediction_groups[i])
      donnees = self.get_all(texts)
      all_infos[f"Image {i}"] = donnees
    return all_infos