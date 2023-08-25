import Augmentor
import os

# Parámetros
TARGET_IMAGES_PER_CLASS = 2000
SOURCE_DIR = 'data'
AUGMENTED_DIR = 'FINALDATA'

# Función para aumentar imágenes de una clase específica
def augment_class_images(class_name, num_required_images):
    class_path = os.path.join(SOURCE_DIR, class_name)
    
    # Verificar si el directorio de salida existe, si no, crearlo.
    if not os.path.exists(AUGMENTED_DIR):
        os.makedirs(AUGMENTED_DIR)
    
    num_existing_images = len(os.listdir(class_path))
    num_images_to_generate = num_required_images - num_existing_images
    
    p = Augmentor.Pipeline(source_directory=class_path, output_directory=AUGMENTED_DIR)
    
    # Desactivar la creación de la subcarpeta 'output'
    p._disable_output_directory_creation()  # Esto evita que Augmentor cree la subcarpeta 'output'
    
    # Operaciones de aumentación
    p.random_contrast(probability=0.5, min_factor=0.8, max_factor=1.2)
    p.random_brightness(probability=0.5, min_factor=0.8, max_factor=1.2)
    p.random_color(probability=0.5, min_factor=0.8, max_factor=1.2)
    p.flip_left_right(probability=0.5)
    p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
    p.zoom_random(probability=0.5, percentage_area=0.9)
    
    # Añadiendo operaciones adicionales para mejorar la detección de pies y manos.
    p.flip_top_bottom(probability=0.2)
    p.random_distortion(probability=0.5, grid_width=4, grid_height=4, magnitude=8)
    
    # Generar imágenes aumentadas
    p.sample(num_images_to_generate)

# Aumentar imágenes para cada clase
augment_class_images('downdog', TARGET_IMAGES_PER_CLASS)
augment_class_images('plank', TARGET_IMAGES_PER_CLASS)
augment_class_images('tree', TARGET_IMAGES_PER_CLASS)
augment_class_images('warrior', TARGET_IMAGES_PER_CLASS)
