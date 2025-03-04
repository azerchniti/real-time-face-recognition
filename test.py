from deepface import DeepFace

# Test avec deux images (remplace par tes propres chemins d'images)
result = DeepFace.verify("wanted_person\download(1).jpg", "other_people\download(1).jpg")

print("Résultat de la comparaison :", result)
