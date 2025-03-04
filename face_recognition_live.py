import cv2
import face_recognition

# Charger l'image de la personne recherchée
wanted_image = face_recognition.load_image_file("known_faces/wanted.jpg")
wanted_encoding = face_recognition.face_encodings(wanted_image)[0]  # Extraire les caractéristiques du visage

# Ouvrir la webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir en RGB (face_recognition utilise RGB et non BGR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Détecter les visages dans la frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Comparer avec la personne recherchée
        matches = face_recognition.compare_faces([wanted_encoding], face_encoding)
        name = "Inconnu"

        if True in matches:
            name = "RECHERCHÉ !"  # Afficher un message d'alerte

        # Dessiner un cadre autour du visage
        color = (0, 255, 0) if name == "RECHERCHÉ !" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Ajouter le nom en dessous du visage
        cv2.putText(frame, name, (left, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Afficher l'image
    cv2.imshow("Face Recognition", frame)

    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Fermer la webcam
cap.release()
cv2.destroyAllWindows()
