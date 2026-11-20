import argparse
from collections import deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp

from capturar_hola import COLUMNAS, LETRAS, dibujar_confirmacion, extraer_coordenadas
from entrenar_hola import MODELO, normalizar


def reconocer(ruta_modelo, indice, umbral):
    artefacto = joblib.load(ruta_modelo)
    if artefacto["columnas"] != COLUMNAS[5:]:
        raise ValueError("El modelo usa un orden de coordenadas incompatible.")
    modelo = artefacto["modelo"]
    lateralidad = artefacto["mano"]
    lado = {"derecha": "Right", "izquierda": "Left"}[lateralidad]
    manos = mp.solutions.hands
    dibujo = mp.solutions.drawing_utils
    historial = deque(maxlen=5)
    avance = ""
    ventana = "Reconocimiento HOLA"
    camara = cv2.VideoCapture(indice)
    try:
        if not camara.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara {indice}.")
        with manos.Hands(max_num_hands=2, min_detection_confidence=0.75) as detector:
            while True:
                correcto, imagen = camara.read()
                if not correcto:
                    raise RuntimeError("La cámara dejó de entregar fotogramas.")
                imagen = cv2.flip(imagen, 1)
                resultado = detector.process(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
                detectadas = resultado.multi_hand_landmarks or []
                mensaje = f"Muestra solo la mano {lateralidad}" if detectadas else "Sin mano"
                prediccion = None
                if len(detectadas) == 1 and resultado.multi_handedness:
                    mano = detectadas[0]
                    dibujo.draw_landmarks(imagen, mano, manos.HAND_CONNECTIONS)
                    if resultado.multi_handedness[0].classification[0].label == lado:
                        puntos = extraer_coordenadas(mano)
                        mensaje = "Encuadra toda la mano"
                        if puntos is not None:
                            try:
                                vector = normalizar(puntos)
                            except ValueError:
                                pass
                            else:
                                puntajes = modelo.predict_proba([vector])[0]
                                mejor = puntajes.argmax()
                                mensaje = "Sin reconocimiento estable"
                                if puntajes[mejor] >= umbral:
                                    prediccion = modelo.classes_[mejor]
                if prediccion is None:
                    historial.clear()
                else:
                    historial.append(prediccion)
                    esperada = LETRAS[len(avance)]
                    if len(historial) == historial.maxlen and all(letra == esperada for letra in historial):
                        avance += esperada
                        historial.clear()
                if avance:
                    dibujar_confirmacion(cv2, imagen, avance)
                if avance == LETRAS:
                    dibujar_confirmacion(cv2, imagen, "COMPLETADO", margen_inferior=66)
                    cv2.imshow(ventana, imagen)
                    break
                mensaje = f"Siguiente: {LETRAS[len(avance)]} | {mensaje}"
                cv2.putText(imagen, mensaje, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
                cv2.putText(imagen, mensaje, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.imshow(ventana, imagen)
                if cv2.waitKey(1) & 0xFF == 27 or cv2.getWindowProperty(ventana, cv2.WND_PROP_VISIBLE) < 1:
                    return
        while cv2.waitKey(30) & 0xFF != 27 and cv2.getWindowProperty(ventana, cv2.WND_PROP_VISIBLE) >= 1:
            pass
    finally:
        camara.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconoce H, O, L y A en la cámara. Escape para salir.")
    parser.add_argument("--modelo", type=Path, default=MODELO)
    parser.add_argument("--camara", type=int, default=0)
    parser.add_argument("--umbral", type=float, default=0.75)
    args = parser.parse_args()
    if not 0 <= args.umbral <= 1:
        parser.error("--umbral debe estar entre 0 y 1.")
    try:
        reconocer(args.modelo, args.camara, args.umbral)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Error de reconocimiento: {error}\n")
