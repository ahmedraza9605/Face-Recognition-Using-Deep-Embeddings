import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis


app = FaceAnalysis(name="buffalo_l")

app.prepare(ctx_id=-1, det_size=(640, 640))


database = {}

faces_folder = "faces"

print("Loading database...\n")

for person in os.listdir(faces_folder):

    person_path = os.path.join(faces_folder, person)

    if os.path.isdir(person_path):

        embeddings = []

        for img_name in os.listdir(person_path):

            img_path = os.path.join(person_path, img_name)

            img = cv2.imread(img_path)

            if img is None:
                continue

            detected_faces = app.get(img)

            if len(detected_faces) > 0:

                embedding = detected_faces[0].embedding
                embeddings.append(embedding)

        if len(embeddings) > 0:

            database[person] = embeddings

            print(f"{person} loaded with {len(embeddings)} images")

print("\nDatabase Ready")
print(database.keys())

def cosine_similarity(a, b):

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)

cv2.setWindowProperty(
    "Face Recognition",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

print("\nStarting Webcam...\n")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    detected_faces = app.get(frame)

    for face in detected_faces:

        embedding = face.embedding

        best_name = "Unknown"
        best_score = -1

        for name, emb_list in database.items():

            scores = []

            for db_emb in emb_list:

                score = cosine_similarity(
                    embedding,
                    db_emb
                )

                scores.append(score)

            avg_score = np.mean(scores)

            if avg_score > best_score:

                best_score = avg_score
                best_name = name

        if best_score < 0.30:
            best_name = "Unknown"

        # DEBUG
        print(best_name, round(best_score, 2))

        x1, y1, x2, y2 = face.bbox.astype(int)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        text = f"{best_name} ({best_score:.2f})"

        cv2.putText(
            frame,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Recognition", frame)


    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

