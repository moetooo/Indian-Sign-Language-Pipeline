# ISL Detection — Exhaustive Viva Question Bank

This document contains a comprehensive, depth-first collection of potential viva questions, ranging from absolute basics (Machine Learning 101) to advanced project-specific implementations, Base Paper comparisons, and Future Scope (Dynamic ISL).

---

## 🟢 Category 1: Absolute Basics (ML & DL Fundamentals)

### 1. What is Machine Learning (ML) vs. Deep Learning (DL)?
*   **ML:** Algorithms that parse data, learn from it, and apply what they’ve learned to make informed decisions (e.g., SVM, Random Forest). Requires feature engineering.
*   **DL:** A subset of ML inspired by the structure of the human brain (Artificial Neural Networks). Capable of automatic feature extraction from raw data (e.g., CNNs on pixels).
*   **Project Context:** We used Deep Learning (a Multi-Layer Perceptron / MLP network), but we *also* performed manual feature engineering (Kinematics) instead of relying purely on the network to find invisible geometric patterns in raw 1D coordinates.

### 2. What is an Artificial Neural Network (ANN)? What is an MLP?
*   **ANN:** A computing system made up of interconnected nodes (neurons) organized in layers (Input, Hidden, Output).
*   **MLP (Multi-Layer Perceptron):** A standard feedforward ANN with at least three layers. "Feedforward" means data only flows in one direction (input to output).
*   **Why we used it:** Perfect for taking a flat, 1D tabular vector (our 38 or 144 features) and mapping it non-linearly to 26 different alphabet classes.

### 3. What is an Epoch? What is a Batch?
*   **Epoch:** One complete pass of the *entire* training dataset through the neural network.
*   **Batch:** The number of samples processed before the model updates its internal weights. We used a batch size of 128. If we have 1000 samples, and a batch size of 100, we have 10 batches per epoch.

### 4. What are Activation Functions? Why do we need them?
*   They introduce *non-linearity* into the network. Without them, a multi-layer network is mathematically identical to a single linear regression model (a straight line).
*   **ReLU (Rectified Linear Unit):** Used in our hidden layers. Function: `f(x) = max(0, x)`. It solves the "vanishing gradient" problem and is computationally very fast.
*   **Softmax:** Used in our *output layer*. It converts raw scores (logits) into a probability distribution (values between 0 and 1 that sum up to 1). If the 'A' node outputs 0.95, it means the model is 95% confident the sign is 'A'.

### 5. What is the difference between Sigmoid and Softmax?
*   **Sigmoid:** Used for binary classification (Yes/No, Cat/Dog). *Note: The base paper used Sigmoid for multi-class, which is technically incorrect.*
*   **Softmax:** Used for multi-class classification (predicting A-Z, 26 mutually exclusive classes).

### 6. What is a Loss Function? Which one did you use?
*   It measures how wrong the model’s predictions are. The optimizer tries to minimize this loss.
*   **Our choice:** `Sparse_Categorical_Crossentropy`. Used when classes are mutually exclusive (it can only be 'A' or 'B', not both) and labels are provided as integers (0, 1, 2...) rather than one-hot encoded vectors ([1,0,0...], [0,1,0...]).

### 7. What is an Optimizer? Which one did you use?
*   The algorithm used to change the attributes of the neural network (weights and biases) to reduce the loss.
*   **Our choice:** `Adam` (Adaptive Moment Estimation). It combines the best properties of the AdaGrad and RMSProp algorithms to provide an optimization algorithm that handles sparse gradients on noisy problems well. Learning rate was set to `0.001`.

### 8. What is Overfitting vs. Underfitting? How did you solve it?
*   **Underfitting:** Model is too simple; cannot learn the training data patterns (high training loss, high validation loss).
*   **Overfitting:** Model memorizes the training data but fails on new, unseen data (low training loss, but validation loss starts increasing).
*   **How we solved Overfitting:**
    1.  **Dropout:** Randomly deactivated 30% or 20% of neurons during training so the network doesn't rely too heavily on specific paths.
    2.  **Early Stopping:** Monitored the validation loss, and stopped training if it didn't improve for 5 epochs.
    3.  **Batch Normalization:** Normalized the outputs of hidden layers, smoothing the loss landscape.
    4.  **Massive Dataset:** 106,000+ samples naturally prevents memorization compared to small datasets.

---

## 🔵 Category 2: Project Implementation & Data (Phase 1)

### 9. What is MediaPipe? Why not YOLO or OpenCV Haar Cascades?
*   **MediaPipe:** A Google framework that provides pre-trained ML models for human pose, face, and hand tracking. Crucially, it returns **3D topological landmarks (x, y, z coordinates)** representing skeleton joints.
*   **YOLO/Haar Cascades:** These are Object Detectors. They return a bounding box (e.g., "A hand is inside these 4 corners"). They do not tell us *how* the fingers are bent inside that box.
*   To recognize sign language, we need the angles of the joints, not just a box around the hand.

### 10. How many landmarks does MediaPipe capture?
*   **Face:** 468 points (We discarded these as they are irrelevant for hand signs)
*   **Pose (Body):** 33 points (We retained only the upper 6: shoulders, elbows, wrists)
*   **Hands:** 21 points per hand (Left and Right).
*   Each point has 3 coordinates (X, Y, Z). Our bare minimum raw feature set became 144 values ( [21 + 21 + 6] * 3 ).

### 11. Where did you get your dataset?
*   We did not rely on a small set of webcam photos.
*   **Source 1:** Kaggle "Indian Sign Language Gesture Landmarks" (78,000 samples). This is highly standardized.
*   **Source 2:** The "Gesture Speech" image dataset containing 31,200 raw photos of hands. We ran `image_pipeline.py` using MediaPipe Hands to extract 28,600 valid landmark rows.
*   This dual-source approach ensures massive variety in hand shapes, preventing the model from memorizing one specific person's hand.

---

## 🟡 Category 3: Kinematic Feature Engineering (Phase 2)

### 12. What does "Kinematic" mean in this context?
Kinematics is the study of motion/geometry without considering forces. In our project, it means extracting the pure *geometric shape* of the hand from the raw skeleton points provided by MediaPipe.

### 13. Why couldn't you just feed raw coordinates to the MLP?
Raw coordinates (x, y) represent *where* the hand is on the screen, not just *what* the hand is doing.
*   If you make an 'A' in the top left corner, the (x,y) values are small.
*   If you make an 'A' in the bottom right corner, the (x,y) values are large.
*   To the MLP, these look like completely completely different sequences of numbers, confusing the model.

### 14. Explain the three kinematic transformations.
1.  **Translation (Centering):** We subtract the wrist's X,Y,Z coordinates from all other finger coordinates. The wrist becomes (0,0,0). Effectively, we moved the hand to the center of a virtual 3D space, removing **position dependence**.
2.  **Normalization:** We take the absolute maximum value across all centered coordinates and divide everything by that max. The hand now fits perfectly inside a 1x1x1 box. This removes **scale (distance/size) dependence**.
3.  **Joint Angles:** We calculate the angles of bend between finger bones (e.g., MCP, PIP, DIP joints) using dot-products of 3D vectors. An angle of a bent index finger is the exact same whether the hand is upside down, sideways, or straight. This removes **rotation dependence**.

### 15. What is the `angles_only` model and why is it special?
Instead of feeding 144 coordinates, we only feed the **38 calculated joint and spread angles** into the neural network (15 joint bends + 4 inter-finger spreads per hand). It achieves 99.75% accuracy. It proves that pure hand geometry (the angles of the fingers) is the true signal for sign language classification, and spatial coordinates are mostly noise.

---

## 🔴 Category 4: The Base Paper Comparison (The "Gotcha" Questions)

### 16. The base paper used CNN. Why didn't you?
*   **Their approach is mathematically flawed.** CNNs utilize convolutional kernels that scan across 2D spatial grids (pixels) looking for adjacent local patterns like edges.
*   MediaPipe outputs a flat 1D array of floats (e.g., [x1, y1, z1, x2, y2, z2...]). There is no spatial grid here. `x1` is not "next to" `x2` in a pixel grid format.
*   Applying a 1D or 2D Convolution over an unordered list of spatial coordinates does not extract meaningful geometric features, it just arbitrarily mashes numbers together. An MLP is the scientifically correct architecture for Dense Tabular/Vector data.

### 17. The base paper got 97.75% static accuracy using 1,662 features. You got 99.9% using 38 features. Why is yours better?
*   **The Curse of Dimensionality & Noise:** The paper fed 468 face landmarks (1,404 numbers) into a model trying to predict static hand letters. This is pure noise that the model has to learn to ignore.
*   **Lack of invariance:** The paper fed raw coordinates. If a user stood slightly to the left during testing compared to training, the model would fail. Our 38 kinematic features are pure, noise-less representations of hand shape.
*   **Dataset Size:** They trained on ~1,800 records. We trained on ~106,000 records. Our model generalized perfectly; theirs likely overfitted their specific webcam setup.

### 18. Why didn't you use LSTM for the Static recognition part?
*   LSTM (Long Short-Term Memory) is designed to process **Time-Series / Sequential Data** (e.g., video frames over time, spoken audio).
*   A static sign (like the letter 'A') is evaluated on a *single frame*. There is no "past" or "future" context needed to know an 'A' is an 'A'.
*   The base paper used LSTM on static signs and got a poor 85.57% accuracy because forcing a temporal model to learn a non-temporal task causes unnecessary difficulty in optimization.

### 19. If you aren't doing Dynamic Gestures yet, how are you handling real-time video?
Real-time webcam inference is a stream of frames. We run the *Static* classifier on every single frame individually. To prevent the text from flickering rapidly (e.g., predicting 'A', then 'E', then 'A' in the span of 3 frames), we built a **Stability Logic Pipeline**:
1.  **Buffer:** We keep the last 5 predictions in memory.
2.  **Majority Vote:** A letter is only considered a valid prediction if it appears in at least 3 out of those 5 frames (60%).
3.  **Confidence:** The average softmax confidence of those predictions must be above 85%.

---

## 🟣 Category 5: Dynamic ISL (Part 2 / Future Scope)

### 20. How will you scale this to recognize full words and dynamic phrases?
This requires moving from Static classification to Continuous Translation (Phases 5-7).
*   **Data Shape Change:** Instead of training on `[106000, 38]` (Samples, Features), we must train on `[N_Videos, 30_Frames, 38_Features]`.
*   **Model Change:** Here, we *will* use an LSTM or Transformer, because a sign like "Hello" involves moving a hand from the forehead outward over time.
*   **Kinematics Applied Over Time:** A major flaw in the base paper was using raw coordinates for their LSTM. We will apply our Phase 2 kinematic centering/angles to *every single frame of the 30-frame sequence*. The LSTM will learn how finger angles change over time, regardless of where the person stands.

### 21. How will you isolate words continuously (Word Boundary Detection)?
If someone is continuously speaking ISL, how does the model know when one word ends and another begins?
*   Instead of looking at isolated, pre-trimmed gesture clips (like the base paper), a real-time system needs a **sliding window** (a `deque` of the last 30 frames).
*   We will implement **Neutral Pose Detection**. Signers naturally drop their hands to a resting position between distinct phrases or words. Detecting this neutral position will trigger the model to output the captured buffer, acting like a "spacebar" for the sentence generator.

### 22. What are the modern alternatives to LSTM for dynamic sign language?
Since the base paper (2023), Sequential Deep Learning has shifted. If asked about state-of-the-art:
*   **DTW (Dynamic Time Warping):** An algorithmic (non-AI) approach that measures similarity between two temporal sequences, resistant to varying speeds of signing. Excellent baseline.
*   **Transformers (e.g., Spatial-Temporal Graph Convolutional Networks or ViT):** Transformers use "Attention" mechanisms to look at the entire 30-frame sequence at once to determine which frames matter most, outperforming LSTMs on long-range context.

---

## 🛠️ Category 6: Engineering & Coding Specifics

### 23. Why did you use `StandardScaler` from scikit-learn?
Even though the data was normalized to [-1, 1] during kinematic engineering, Neural Networks train best when features have a mean of 0 and standard deviation of 1. `StandardScaler` (Z-score normalization) centers the distribution. **Crucial:** We saved the scaler object as an `.pkl` file. During real-time inference, the incoming webcam frame is passed through *the exact same applied scaling transformations* as the data the model was trained on.

### 24. What are those `.h5` files in the models directory?
They are Hierarchical Data Format 5 (HDF5) files. Originally designed by the NCSA, Keras/TensorFlow uses them to save both the Model Architecture (layers, nodes) and the corresponding trained Weights/Biases in a single binary file.

### 25. Explain your Ablation Study.
An ablation study is a scientific experiment where components of a system are systematically altered or removed to understand their contribution to the overall outcome.
*   We trained **7 separate MLP models**.
*   We varied the feature sets: `raw_coordinates (144)`, `centered_kinematics (182)`, and `angles_only (38)`.
*   We varied the datasets: Kaggle vs. Image-extracted.
*   This allowed us to conclusively state: *"Adding angle features maintains >99% accuracy on a different dataset, proving rotation invariance, while reducing computational load."*

### 26. What was the output activation math for your Multi-Class setup?
The output layer consists of 26 neurons. We use the **Softmax function**: `e^(z_i) / sum(e^(z_j))`
This takes the raw numerical outputs (logits) of the 26 nodes, exponentiates them (making them all positive), and divides by the sum of all exponentiated outputs. This forces the output array to sum to 1.0, representing a strict probability distribution across the 26 classes.
