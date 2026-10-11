# Real-Time Face Recognition System Using Deep Face Embeddings and Cosine Similarity

**Author:** Muhammad Ahmed Raza  
**Department:** Department of Computer Science, Abasyn University, Islamabad, Pakistan  
**Email:** raza anjum9605@gmail.com

## Abstract

This project presents a real-time face recognition system using deep learning and computer vision techniques. The system uses the InsightFace library and its Buffalo-L model to detect faces and extract facial embeddings from images. It can recognize multiple faces in a live webcam feed, label known individuals, and classify unknown faces when no match satisfies the configured similarity threshold.

The system compares facial embeddings using cosine similarity to identify the closest match from a database of known individuals. This project demonstrates the practical application of deep learning and computer vision in facial recognition.

**Keywords:** Face Recognition, Face Embeddings, Cosine Similarity, Buffalo-L Model, InsightFace, OpenCV.

## 1. Project Structure

The project stores reference images of known individuals in separate folders. Each folder is named after the corresponding person.

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

**File and folder descriptions:**

- `main.py`: Runs the real-time face recognition system.
- `faces/`: Contains reference images used to identify known individuals.
- `Ahmed/`, `Hamza/`, and `Ali/`: Contain images of the respective individuals.

You can add more people by creating additional folders inside the `faces/` directory and placing their reference images inside them.

## 2. Required Software and Libraries

The project requires Python and the following libraries:

- OpenCV
- NumPy
- InsightFace
- ONNX Runtime

### Check Python and pip versions

Open the terminal in Visual Studio Code and run:

```bash
python --version
```

Check the pip version:

```bash
pip --version
```

Use a Python version compatible with the installed InsightFace and ONNX Runtime packages. Python 3.10 is a reasonable starting point, but compatibility depends on the package versions and operating system.

### Install the required libraries

Run these commands in the terminal:

```bash
pip install opencv-python
pip install numpy
pip install insightface
pip install onnxruntime
```

### Verify the installation

To check the installed packages, run:

```bash
pip list
```

Make sure OpenCV, NumPy, InsightFace, and ONNX Runtime appear in the package list.

## 3. Loading the AI Model

The system uses the **Buffalo-L model** provided by InsightFace.

The model provides face detection and face recognition capabilities. It detects faces in images and generates numerical embeddings that represent facial characteristics.

During initialization, the model is loaded and prepared for processing images from the reference database and the live webcam feed.

## 4. Database Creation

The system creates an in-memory database of facial embeddings using the images stored in the `faces/` directory.

The process involves the following steps:

1. Read the reference images from each person's folder.
2. Detect faces in each image using InsightFace.
3. Extract a facial embedding for each detected face.
4. Associate each embedding with the corresponding person's name.
5. Store the embeddings and associated names in memory.

This database is then used to compare faces detected by the webcam against the known individuals.

The database described here is stored in memory during execution; it is not necessarily a permanent database saved to disk.

## 5. Real-Time Face Recognition

The system uses a webcam to capture live video frames and recognize faces.

The processing steps are:

1. Capture a frame from the webcam using OpenCV.
2. Detect all visible faces in the frame.
3. Extract an embedding for each detected face.
4. Compare each live embedding with the stored reference embeddings.
5. Calculate cosine similarity scores.
6. Identify the best-matching individual if the score satisfies the configured threshold.
7. Display the person's name or label the face as `Unknown Person`.

This process repeats as new webcam frames are captured, allowing the system to recognize multiple faces in real time.

## 6. Feature Extraction Using Face Embeddings

After detecting a face, InsightFace extracts a numerical representation called a **face embedding**.

Instead of directly comparing two facial images pixel by pixel, the system compares their embeddings.

An illustrative embedding looks like this:

```text
[0.21, 1.34, -0.76, 0.55, ...]
```

The selected Buffalo-L recognition model typically generates a 512-dimensional embedding for each face.

These numerical values represent learned facial patterns. They are not direct measurements of individual features such as the eyes, nose, or face geometry.

The system uses these embeddings to compare a live face with the reference faces stored in its database.

## 7. Similarity Calculation Using Cosine Similarity

Cosine similarity measures the similarity between two numerical vectors by calculating the cosine of the angle between them.

The formula is:

$$
\operatorname{CosineSimilarity}(A,B)=\frac{A\cdot B}{\|A\|\|B\|}
$$

**Where:**

- \(A\) = embedding extracted from the live face.
- \(B\) = embedding stored in the database.
- \(A\cdot B\) = dot product of the two embeddings.
- \(\|A\|\) = magnitude (Euclidean norm) of vector \(A\).
- \(\|B\|\) = magnitude (Euclidean norm) of vector \(B\).

### Interpretation of Similarity Scores

For nonzero vectors, cosine similarity ranges from -1 to 1.

| Similarity Score | General Interpretation |
|---|---|
| Close to 1.0 | Highly similar vector directions |
| Around 0.8 | Strong similarity |
| Around 0.5 | Moderate similarity |
| Around 0.0 | Little or no directional similarity |
| Below 0.0 | Opposite vector directions |

These values are illustrative. A similarity score is not automatically a probability or percentage confidence. Actual face recognition performance depends on the model, image quality, and selected threshold.

## 8. Face Recognition Example

Suppose the webcam captures Ahmed's face. The system extracts an embedding and compares it with the embeddings stored in the database.

Example similarity scores:

| Person | Similarity Score |
|---|---:|
| Ahmed | 0.89 |
| Hamza | 0.43 |
| Ali | 0.21 |

Ahmed has the highest similarity score.

If the score also satisfies the configured recognition threshold, the system identifies the face as Ahmed.

Example output:

```text
Ahmed (0.89)
```

The displayed score represents cosine similarity, not a calibrated probability that the identification is correct.

## 9. Unknown Face Handling

The system uses a similarity threshold to avoid assigning every detected face to a known individual.

For example, suppose the configured threshold is `0.30`.

The system compares a live face with the reference embeddings and obtains the following scores:

| Person | Similarity Score |
|---|---:|
| Ahmed | 0.25 |
| Hamza | 0.18 |
| Ali | 0.12 |

The highest score is `0.25`, which is below the threshold of `0.30`.

Therefore, the system labels the face as:

```text
Unknown Person
```

**Note:** The threshold of `0.30` is an example and should not be treated as a universally suitable value. The appropriate threshold must be selected through testing with genuine and different-person face comparisons.

A threshold that is too low may cause false matches, while a threshold that is too high may reject genuine matches.

## 10. Displaying Recognition Results

The system displays recognition results directly on the webcam frame.

For each detected face, it draws a bounding box around the face and displays the predicted identity and similarity score.

Example:

```text
+----------------------+
|                      |
|     Detected Face    |
|                      |
+----------------------+
Stalin (0.61)
```

In this example:

- **Name:** Stalin
- **Similarity score:** 0.61

The system should display a known person's name only when the highest similarity score satisfies the configured threshold. Otherwise, it should display `Unknown Person`.

OpenCV is used to display the webcam frames and draw the bounding boxes and labels.

## 11. Complete System Flow

The complete face recognition process follows this workflow:

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

Real-time face recognition systems can be applied in several areas, including:

- Attendance management systems.
- Access control and identity verification.
- Employee verification systems.
- Security monitoring.
- Smart surveillance systems.
- Computer vision research and experimentation.
- Biometric recognition applications.

These are potential applications. Using this project in a real-world security or authentication system would require additional testing, safeguards, and privacy protections.

## 13. Privacy and Responsible Use

Facial images and face embeddings can contain sensitive biometric information.

When using this system:

- Obtain appropriate permission before collecting or processing people's facial images.
- Protect stored images and embeddings from unauthorized access.
- Avoid using the system to identify people without authorization.
- Evaluate recognition errors before using the system for important decisions.
- Follow applicable privacy and data protection laws.

The system is intended for learning, research, and authorized applications.

## 14. Conclusion

This project implements a real-time face recognition system using InsightFace, the Buffalo-L model, OpenCV, and cosine similarity.

The system first processes reference images to create an in-memory database of face embeddings. During operation, the webcam captures live frames, detects faces, extracts embeddings, and compares them with the stored reference embeddings.

Cosine similarity is used to identify the closest matching individual. If the highest similarity score meets the configured threshold, the system displays the predicted person's name and similarity score. Otherwise, it labels the face as `Unknown Person`.

This project demonstrates how deep learning and computer vision can be combined to build a real-time face recognition application and provides a foundation for further experimentation with facial biometrics.

## Author

**Muhammad Ahmed Raza**  
Department of Computer Science  
Abasyn University, Islamabad, Pakistan  
Email: raza anjum9605@gmail.com
