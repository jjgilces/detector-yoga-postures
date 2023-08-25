import gradio as gr
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
from gradio import components,themes  


class ImprovedTransferNetVGG19(models.Model):
    def __init__(self, base_model_weights=None):
        super(ImprovedTransferNetVGG19, self).__init__()
        self.base_model = tf.keras.applications.VGG19(include_top=False, weights=base_model_weights, input_shape=(224, 224, 3))
        for layer in self.base_model.layers:
            layer.trainable = False  # Freeze VGG19 layers initially
        self.flatten = layers.Flatten()
        
        l1lambda = 0.0001
        # Dense layers with reduced neurons
        self.fc1 = layers.Dense(1024, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(l1lambda))
        self.fc1_bn = layers.BatchNormalization()
        self.fc1_dropout = layers.Dropout(0.4)

        self.fc2 = layers.Dense(512, activation='relu')
        self.fc2_bn = layers.BatchNormalization()
        self.fc2_dropout = layers.Dropout(0.4)
        
        self.fc3 = layers.Dense(128, activation='relu')
        self.fc3_bn = layers.BatchNormalization()
        self.fc3_dropout = layers.Dropout(0.4)

        self.fc4 = layers.Dense(4, activation='softmax')

    def call(self, x):
        x = self.base_model(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = self.fc1_dropout(x)
        
        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = self.fc2_dropout(x)

        x = self.fc3(x)
        x = self.fc3_bn(x)
        x = self.fc3_dropout(x)

        x = self.fc4(x)
        return x
    def unfreeze_last_layers(self, num_layers=6):
        """Desbloquea las últimas num_layers del modelo base."""
        for layer in self.base_model.layers[-num_layers:]:
            layer.trainable = True

    def get_config(self):
        return {"base_model_name": "VGG19"}

    @classmethod
    def from_config(cls, config):
        if config["base_model_name"] == "VGG19":
            return cls(base_model_weights="imagenet")



model = ImprovedTransferNetVGG19()
model(tf.zeros([1, 224, 224, 3]))  # Esto construirá el modelo
model.unfreeze_last_layers(num_layers=6) #modelo fine tunning
model.load_weights('v5.h5')

def predict(img):
    if img is None:
        return "Sin imagen", "Por favor, sube una imagen"
    if img.shape[0] == 0 or img.shape[1] == 0:
        return "Sin imagen", "Por favor, sube una imagen"
    # Cambia el tamaño de la imagen al tamaño de entrada esperado y prepárala para la predicción
    img = tf.image.resize(img, (224, 224))
    img = img / 255.0  # Escalar los valores de píxeles al rango [0, 1]
    img = tf.expand_dims(img, axis=0)
    
    # Aquí, suponemos que tienes una lista de nombres de clases en el orden apropiado
    class_names = ["Downdog", "Plank", "Tree", "Warrior"]
    
    # Haz predicciones
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0]
    prob = predictions[0][predicted_class] * 100
    probability_str = f"{prob:.2f}%"
    # Encuentra la etiqueta de clase con la mayor probabilidad
    label = class_names[predicted_class]  
    probabilities = [f"{prob*100:.2f}%" for prob in predictions[0]]
    labeled_probs = list(zip(class_names, probabilities))
    return label, probability_str

css = """
body {
    background-color: white;
}
.gradio-content {
    background-color: white;
}
"""

# Define y lanza la interfaz
interface = gr.Interface(
    fn=predict, 
    inputs=components.Image(shape=(224, 224)), 
    outputs=[
        components.Label(num_top_classes=4, label="Pose"), 
        components.Textbox(label="Probabilidad")
    ],
    live=True, 
    theme=gr.themes.Soft(primary_hue=gr.themes.colors.red, secondary_hue=gr.themes.colors.pink),
    css=css,
    title="Yoga Pose Classifier",
    examples=[
        os.path.join(os.path.dirname(__file__), "img/downdog.jpg"),
        os.path.join(os.path.dirname(__file__), "img/plank.jpg"),
        os.path.join(os.path.dirname(__file__), "img/tree.jpg"),
        os.path.join(os.path.dirname(__file__), "img/warrior.jpg"),
    ],
)

interface.launch()
