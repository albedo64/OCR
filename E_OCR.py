import tensorflow as tf
from ocr_class import OCR

img1 = tf.keras.utils.get_file(
'img1.jpg',
'https://prod.cdn-medias.jeuneafrique.com/cdn-cgi/image/q=auto,f=auto,metadata=none,width=1215,fit=cover/https://prod.cdn-medias.jeuneafrique.com/medias/2021/01/12/jad20210112-ass-cameroun-carte-identite.jpg')

# Création d'une instance de la classe OCR
ocr_instance = OCR()

# Appel de la méthode all_in_one_OCR avec l'image img1
resultat = ocr_instance.all_in_one_OCR(img1)

# Affichage du résultat
print(resultat)