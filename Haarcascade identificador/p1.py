import cv2 as cv

img = cv.imread('imgs/figura.png', 1)
img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

ubb = (0, 60, 60)
uba = (10, 255, 255)
ubb1 = (170, 60, 60)
uba1 = (180, 255, 255)

mascara1 = cv.inRange(hsv, ubb, uba)
mascara2 = cv.inRange(hsv, ubb1, uba1)

mascara = mascara1 + mascara2
resultado = cv.bitwise_and(img, img, mask=mascara)

# Encontrar contornos de las figuras rojas
contornos, _ = cv.findContours(mascara, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

print("Coordenadas de las figuras rojas:")
print("-" * 40)

for i, contorno in enumerate(contornos):
    # Filtrar contornos muy pequeños
    area = cv.contourArea(contorno)
    if area > 500:  # Solo figuras con área significativa
        # Calcular centroide
        M = cv.moments(contorno)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])  # Coordenada X del centroide
            cy = int(M["m01"] / M["m00"])  # Coordenada Y del centroide
        else:
            cx, cy = 0, 0
        
        # Obtener bounding box (rectángulo que contiene la figura)
        x, y, w, h = cv.boundingRect(contorno)
        
        print(f"Figura roja {i+1}:")
        print(f"  Centroide: ({cx}, {cy})")
        print(f"  Esquina superior izquierda: ({x}, {y})")
        print(f"  Ancho: {w}, Alto: {h}")
        print(f"  Esquina inferior derecha: ({x+w}, {y+h})")
        print("-" * 40)

cv.imshow('resultado', resultado)
cv.imshow('mascara1', mascara1)
cv.imshow('mascara2', mascara2)
cv.imshow('img', img)
#cv.imshow('img2', img2)
#cv.imshow('hsv', hsv)

cv.waitKey(0)
cv.destroyAllWindows()