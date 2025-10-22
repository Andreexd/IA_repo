import cv2
import mediapipe as mp
import math
import numpy as np

# Configuración MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Puntos clave para detección de emociones
MOUTH_LEFT = 61      # Esquina izquierda de la boca
MOUTH_RIGHT = 291    # Esquina derecha de la boca
UPPER_LIP = 13       # Labio superior centro
LOWER_LIP = 14       # Labio inferior centro
LEFT_EYE_INNER = 133 # Esquina interna del ojo izquierdo
RIGHT_EYE_INNER = 362 # Esquina interna del ojo derecho
LEFT_EYEBROW = 70    # Punto de la ceja izquierda
RIGHT_EYEBROW = 300  # Punto de la ceja derecha
NOSE_TIP = 1         # Punta de la nariz
CHIN = 18            # Punto del mentón

# Puntos del contorno facial (oval de la cara)
FACE_OVAL_POINTS = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]

def calculate_distance(point1, point2):
    """Calcula la distancia euclidiana entre dos puntos"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def get_landmark_coordinates(landmarks, image_width, image_height, point_idx):
    """Convierte landmarks normalizados a coordenadas de píxeles"""
    landmark = landmarks.landmark[point_idx]
    x = int(landmark.x * image_width)
    y = int(landmark.y * image_height)
    return (x, y)

def detect_emotion(landmarks, image_width, image_height):
    """
    Detecta emociones básicas basándose en landmarks faciales:
    - Felicidad: sonrisa (comisuras hacia arriba)
    - Enojo: cejas bajas y juntas
    - Tristeza: comisuras hacia abajo
    """
    try:
        # Obtener coordenadas de puntos clave
        mouth_left = get_landmark_coordinates(landmarks, image_width, image_height, MOUTH_LEFT)
        mouth_right = get_landmark_coordinates(landmarks, image_width, image_height, MOUTH_RIGHT)
        upper_lip = get_landmark_coordinates(landmarks, image_width, image_height, UPPER_LIP)
        lower_lip = get_landmark_coordinates(landmarks, image_width, image_height, LOWER_LIP)
        left_eyebrow = get_landmark_coordinates(landmarks, image_width, image_height, LEFT_EYEBROW)
        right_eyebrow = get_landmark_coordinates(landmarks, image_width, image_height, RIGHT_EYEBROW)
        left_eye = get_landmark_coordinates(landmarks, image_width, image_height, LEFT_EYE_INNER)
        right_eye = get_landmark_coordinates(landmarks, image_width, image_height, RIGHT_EYE_INNER)
        nose_tip = get_landmark_coordinates(landmarks, image_width, image_height, NOSE_TIP)
        
        # Calcular características faciales
        mouth_width = calculate_distance(mouth_left, mouth_right)
        mouth_height = calculate_distance(upper_lip, lower_lip)
        
        # Altura de la boca respecto al centro
        mouth_center_y = (upper_lip[1] + lower_lip[1]) / 2
        mouth_corners_y = (mouth_left[1] + mouth_right[1]) / 2
        mouth_curve = mouth_center_y - mouth_corners_y  # Positivo = sonrisa, negativo = tristeza
        
        # Distancia entre cejas (para detectar enojo)
        eyebrow_distance = calculate_distance(left_eyebrow, right_eyebrow)
        
        # Altura de cejas respecto a los ojos
        left_brow_eye_distance = left_eyebrow[1] - left_eye[1]
        right_brow_eye_distance = right_eyebrow[1] - right_eye[1]
        avg_brow_distance = (left_brow_eye_distance + right_brow_eye_distance) / 2
        
        # Normalizar usando el ancho de la boca como referencia
        if mouth_width > 0:
            mouth_curve_ratio = mouth_curve / mouth_width
            
            # Umbrales para detección de emociones
            if mouth_curve_ratio > 0.05:  # Sonrisa pronunciada
                return "felicidad"
            elif mouth_curve_ratio < -0.03:  # Comisuras hacia abajo
                return "tristeza"
            elif avg_brow_distance < -15 and eyebrow_distance < mouth_width * 0.8:  # Cejas bajas y juntas
                return "enojo"
        
        return "neutral"
    
    except Exception as e:
        print(f"Error en detección de emoción: {e}")
        return "neutral"

def draw_face_contour(image, landmarks, emotion, image_width, image_height):
    """Dibuja los puntos del contorno facial según la emoción detectada"""
    # Colores para cada emoción
    emotion_colors = {
        "felicidad": (0, 255, 0),    # Verde
        "enojo": (0, 0, 255),        # Rojo
        "tristeza": (255, 0, 0),     # Azul
        "neutral": (255, 255, 255)   # Blanco
    }
    
    color = emotion_colors.get(emotion, (255, 255, 255))
    
    # Dibujar puntos del contorno facial
    for point_idx in FACE_OVAL_POINTS:
        try:
            x, y = get_landmark_coordinates(landmarks, image_width, image_height, point_idx)
            cv2.circle(image, (x, y), 3, color, -1)
        except:
            continue
    
    return image

def main():
    # Inicializar captura de video
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara")
        return
    
    # Configurar MediaPipe Face Mesh
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        print("Presiona 'q' para salir")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo leer el frame")
                break
            
            # Voltear la imagen horizontalmente (efecto espejo)
            frame = cv2.flip(frame, 1)
            
            # Convertir BGR a RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Procesar con MediaPipe
            results = face_mesh.process(rgb_frame)
            
            # Obtener dimensiones de la imagen
            image_height, image_width = frame.shape[:2]
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Detectar emoción
                    emotion = detect_emotion(face_landmarks, image_width, image_height)
                    
                    # Dibujar contorno facial con color según emoción
                    frame = draw_face_contour(frame, face_landmarks, emotion, image_width, image_height)
                    
                    # Mostrar la emoción detectada
                    cv2.putText(frame, f"Emocion: {emotion}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Mostrar instrucciones
            cv2.putText(frame, "Verde: Felicidad | Rojo: Enojo | Azul: Tristeza", 
                       (10, image_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Mostrar frame
            cv2.imshow('Detector de Emociones - MediaPipe', frame)
            
            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Limpiar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
