"""
MINOR PROJECT: CNN IMAGE CLASSIFICATION AND STREAMLIT APPLICATION
Student: Partha Gogoi
Institute: NERIST
Batch: B AI
Dataset: MNIST
Streamlit app topic: Student Marks Calculator

Install:
    pip install tensorflow streamlit numpy pillow

Run:
    streamlit run Partha_Gogoi_CNN_Streamlit_Project.py
"""

# ============================================================
# PART A - CNN IMAGE CLASSIFICATION USING MNIST
# ============================================================
import os
import numpy as np
import tensorflow as tf

MODEL_PATH = "mnist_cnn_model.keras"

def build_cnn_model():
    """Create and compile a CNN for 28x28 grayscale digit images."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_and_evaluate_cnn():
    """Load MNIST, preprocess it, train the CNN, and evaluate test accuracy."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    model = build_cnn_model()
    model.fit(
        x_train, y_train,
        epochs=8,
        batch_size=128,
        validation_split=0.1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2, restore_best_weights=True
            )
        ],
        verbose=1
    )
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    model.save(MODEL_PATH)
    print(f"MNIST test accuracy: {accuracy * 100:.2f}%")
    return model, loss, accuracy

def get_cnn_model():
    """Load an existing trained model, or train one if it is not available."""
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    model, _, _ = train_and_evaluate_cnn()
    return model

# ============================================================
# PART B - STREAMLIT APPLICATION: STUDENT MARKS CALCULATOR
# ============================================================
def run_streamlit_app():
    import streamlit as st

    st.set_page_config(page_title="Student Marks Calculator", page_icon="🎓")
    st.title("🎓 Student Marks Calculator")
    st.write(
        "Enter a student's name and marks in five subjects. "
        "The app calculates total, average, percentage, and grade."
    )

    student_name = st.text_input("Student name")
    st.subheader("Enter marks (0 to 100)")
    subject_names = ["English", "Mathematics", "Science", "Computer", "Social Science"]
    marks = []
    for subject in subject_names:
        value = st.number_input(
            f"{subject} marks",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
            key=subject
        )
        marks.append(value)

    if st.button("Calculate Result"):
        if not student_name.strip():
            st.error("Please enter the student's name.")
        elif len(marks) != 5 or any(mark < 0 or mark > 100 for mark in marks):
            st.error("Invalid marks. Each subject must be between 0 and 100.")
        else:
            total = sum(marks)
            average = total / len(marks)
            percentage = (total / 500) * 100

            if percentage >= 90:
                grade = "A+"
            elif percentage >= 80:
                grade = "A"
            elif percentage >= 70:
                grade = "B"
            elif percentage >= 60:
                grade = "C"
            elif percentage >= 50:
                grade = "D"
            else:
                grade = "F"

            st.success(f"Result calculated for {student_name.strip()}.")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Marks", f"{total:.1f}/500")
            col2.metric("Average", f"{average:.2f}")
            col3.metric("Percentage", f"{percentage:.2f}%")
            st.subheader(f"Grade: {grade}")
            if grade == "F":
                st.warning("The student needs improvement.")
            else:
                st.info("Congratulations on completing the assessment!")

            st.write("### Subject-wise marks")
            st.table({
                "Subject": subject_names,
                "Marks": [f"{mark:.1f}" for mark in marks]
            })

if __name__ == "__main__":
    # Streamlit sets this environment variable when launched with streamlit run.
    # CNN training/evaluation functions above can also be run independently.
    run_streamlit_app()
