import numpy as np
import pandas as pd
import time
from PIL import Image, ImageOps
import tensorflow as tf

class FruitClassifier:
    """Loads TFLite model and runs predictions."""
    def __init__(self, model_path="model_unquant.tflite", labels_path="labels.txt"):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        with open(labels_path, "r") as f:
            self.labels = [line.strip().split(" ", 1)[1] for line in f.readlines()]

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        data = np.asarray(image, dtype=np.float32)
        data = (data / 127.5) - 1
        data = np.expand_dims(data, axis=0)
        self.interpreter.set_tensor(self.input_details[0]['index'], data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        index = np.argmax(output)
        return self.labels[index], float(output[0][index])


class AnalyticsLogger:
    """Logs detection events to a CSV."""
    def __init__(self):
        self.logs = []

    def log_event(self, label, confidence):
        self.logs.append({
            "timestamp": time.time(),
            "detected_fruit": label,
            "confidence_score": round(confidence, 4)
        })

    def save_to_csv(self, filename="session_log.csv"):
        df = pd.DataFrame(self.logs)
        df.to_csv(filename, index=False)
        print(f"Saved to {filename}")
        return df


if __name__ == "__main__":
    print("Loading model...")
    classifier = FruitClassifier()
    print("Model loaded!")
    logger = AnalyticsLogger()

    test_images = [
    "test_fruit.jpg",
    "test_banana.jpg",
    "test_banana1.jpg",
    "test_banana2.jpg",
    "test_orange.jpg",
    "test_orange1.jpg",
    "test_orange2.jpg",
]

    for image in test_images:
        label, confidence = classifier.predict(image)
        logger.log_event(label, confidence)
        print(f"Detected: {label} ({confidence:.2%} confidence)")

    logger.save_to_csv()