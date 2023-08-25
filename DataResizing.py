import cv2

image_path = '/Users/johanjairgilcesreyes/Desktop/ESPOL/IA/Proyecto/YOGA/DATA/downdog/1.jpg'
img = cv2.imread(image_path)

# Calculate the aspect ratio
aspect_ratio = img.shape[1] / img.shape[0]

# Desired height
new_height = 250
new_width = int(new_height * aspect_ratio)

# Resize the image
resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

# Save or further process the resized image
cv2.imwrite('prueba/resized_image.jpg', resized_img)
