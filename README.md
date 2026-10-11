# Real-Time Face Recognition System Using Deep Face Embeddings and Cosine Similarity

**Muhammad Ahmed Raza**  
Department of Computer Science, Abasyn University, Islamabad, Pakistan  
Email: razaanjum9605@gmail.com

## Abstract

This project presents a real-time face recognition system using deep learning and computer vision techniques. The system uses the InsightFace model to detect faces and extract facial embeddings from images. It can recognize multiple faces, label known individuals, and classify unknown faces when no sufficiently similar match is found. The system demonstrates the use of facial biometrics for recognition and security-related applications.

**Keywords:** Face Recognition, Face Embeddings, Cosine Similarity, Buffalo-L Model, InsightFace, OpenCV

## 1. Project Structure

The project uses a `faces` directory to store reference images of known individuals.

```text
Project/
├── main.py
└── faces/
    ├── Ahmed/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── img3.jpg
    ├── Hamza/
    │   ├── img1.jpg
    │   └── img2.jpg
    └── Ali/
        ├── img1.jpg
        └── img2.jpg
```

- `main.py`: Runs the real-time face recognition system.
- `faces/`: Contains reference images of known individuals.
- Individual subfolders: Use the person's name as the folder name and store their reference images inside it.

## 2. Required Software and Libraries

Install Python and use Visual Studio Code or another Python-compatible IDE.

Check your Python version:

```bash
python --version
```

Check your pip version:

```bash
pip --version
```

### Install the required libraries

Run the following commands in the terminal:

```bash
pip install opencv-python
pip install numpy
pip install insightface
pip install onnxruntime
```

Verify that the libraries are installed:

```bash
pip list
```

**Note:** InsightFace and its dependencies must be compatible with your Python version and operating system. Python 3.10 or another supported version is recommended based on package compatibility.

## 3. Loading the AI Model

The system uses the **Buffalo-L model** provided by InsightFace.

This model includes components for face detection and face recognition. It detects faces in an image and generates a numerical representation of each detected face.

## 4. Database Creation

The system creates an in-memory database of face embeddings from the images stored in the `faces` directory.

The process is:

1. Read the images from each person's folder.
2. Detect faces in the images using InsightFace.
3. Extract the face embedding for each detected face.
4. Associate each embedding with the corresponding person's name.
5. Store the embeddings in memory for later comparison.

This database allows the system to compare faces captured by the webcam against known individuals.

## 5. Real-Time Processing

The system uses a webcam to recognize faces in real time.

The process follows these steps:

1. Capture a frame from the webcam.
2. Detect all faces in the frame.
3. Extract an embedding for each detected face.
4. Compare each live embedding with the stored database embeddings.
5. Calculate the similarity scores.
6. Identify the best matching person if the similarity exceeds the configured threshold.
7. Display the person's name or label the face as unknown.

## 6. Feature Extraction Using Face Embeddings

InsightFace extracts distinguishing numerical features from a detected face. Instead of comparing raw images directly, the system compares these numerical representations, known as **face embeddings**.

An illustrative embedding may look like this:

```text
[0.21, 1.34, -0.76, 0.55, ...]
```

The recognition model typically produces a **512-dimensional embedding** for each face using the selected Buffalo-L recognition model.

These values represent learned facial features and patterns. They are not separate, directly readable measurements of the eyes, nose, or facial proportions.

## 7. Similarity Calculation Using Cosine Similarity

Cosine similarity measures the similarity between two numerical vectors.

The formula is:

\[
\text{Cosine Similarity}(A,B)=
\frac{A\cdot B}{\|A\|\|B\|}
\]

Where:

- \(A\) = embedding extracted from the live face.
- \(B\) = embedding stored in the database.
- \(A\cdot B\) = dot product of the two embeddings.
- \(\|A\|\) and \(\|B\|\) = magnitudes of the vectors.

### Interpretation

For normalized embeddings, cosine similarity commonly ranges from -1 to 1.

| Similarity Score | General Interpretation |
|---|---|
| Close to 1.0 | Highly similar embeddings |
| Around 0.8 | Strong similarity |
| Around 0.5 | Moderate similarity |
| Around 0.0 | Little or no directional similarity |
| Below 0.0 | Opposite vector directions |

These values are illustrative, not universal recognition rules. The correct threshold depends on the model, preprocessing, and testing data.

## 8. Recognition Example

Suppose the system captures Ahmed's face through the webcam and compares it with the stored embeddings.

| Person | Similarity Score |
|---|---:|
| Ahmed | 0.89 |
| Hamza | 0.43 |
| Ali | 0.21 |

Ahmed has the highest similarity score. If the score also satisfies the configured recognition threshold, the system identifies the person as Ahmed.

Example output:

```text
Ahmed (0.89)
```

## 9. Unknown Face Handling

The system uses a similarity threshold to prevent every detected face from being assigned to a known person.

For example, suppose the configured threshold is `0.30`.

| Person | Similarity Score |
|---|---:|
| Ahmed | 0.25 |
| Hamza | 0.18 |
| Ali | 0.12 |

The highest score is `0.25`, which is below the threshold of `0.30`. The system therefore labels the face as:

```text
Unknown Person
```

**Important:** The threshold of `0.30` is an example. It should be selected through testing with genuine and different-person face comparisons. A low threshold can cause false matches, while a high threshold can reject genuine matches.

## 10. Displaying Recognition Results

The system displays the recognition results directly on the webcam frame.

For each detected face, it draws a bounding rectangle and displays the predicted name and similarity score.

Example:

```text
+------------------+
|                  |
|    Detected Face |
|                  |
+------------------+
Stalin (0.61)
```

Here, `Stalin` is the predicted identity, and `0.61` is the similarity score. The system should display this identity only if the configured threshold is satisfied; otherwise, it should display `Unknown Person`.

The similarity score is not necessarily a calibrated probability or a percentage confidence.

## 11. Complete System Flow

```text
Reference Images
       |
       v
Face Detection
       |
       v
Feature Extraction
       |
       v
Store Face Embeddings
       |
       v
Capture Webcam Frame
       |
       v
Detect Live Faces
       |
       v
Extract Live Embeddings
       |
       v
Calculate Cosine Similarity
       |
       v
Find Highest Similarity Score
       |
       v
Compare with Threshold
       |
       +----------------------+
       |                      |
       v                      v
Threshold Satisfied     Below Threshold
       |                      |
       v                      v
Recognized Person       Unknown Person
```

## 12. Applications

Potential applications of real-time face recognition include:

- Attendance management systems.
- Access control and identity verification.
- Employee verification systems.
- Security monitoring.
- Smart surveillance systems.
- Academic research in computer vision and facial biometrics.

These are potential applications; the project itself does not automatically provide production-grade security or authentication.

## 13. Privacy and Responsible Use

Face embeddings and face images can be sensitive biometric data. Use images only with appropriate permission, protect stored data, and avoid collecting or identifying people without authorization. Any real-world deployment should consider privacy, consent, security, and applicable laws.

## 14. Conclusion

This project implements real-time face recognition using InsightFace, the Buffalo-L model, OpenCV, and cosine similarity. It first processes reference images to create a database of face embeddings. During operation, the webcam captures live frames, detects faces, extracts embeddings, and compares them with the stored representations.

The system selects the best matching identity when the similarity score meets the configured threshold. Otherwise, it labels the detected face as unknown. The project demonstrates the practical use of deep learning and computer vision in facial recognition and provides a foundation for further experimentation in biometric systems.

## Author

**Muhammad Ahmed Raza**  
Department of Computer Science  
Abasyn University, Islamabad, Pakistan  
Email: raza anjum9605@gmail.com
