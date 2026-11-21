import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from time import monotonic
from uuid import uuid4


LETRAS = "HOLA"
CSV_PREDETERMINADO = Path(__file__).resolve().parent / "datos" / "hola.csv"
COLUMNAS = ["etiqueta", "participante_id", "sesion_id", "toma_id", "lateralidad"] + [
    f"{eje}{indice}" for indice in range(21) for eje in "xyz"
]


def preparar_csv(ruta):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("a+", encoding="utf-8", newline="") as archivo:
        archivo.seek(0)
        lector = csv.reader(archivo)
        encabezado = next(lector, None)
        if encabezado is None:
            csv.writer(archivo).writerow(COLUMNAS)
            return Counter()
        if encabezado != COLUMNAS:
            raise ValueError(f"El encabezado de {ruta} no coincide con el esquema de captura.")
        conteos = Counter()
        for fila in lector:
            if len(fila) != len(COLUMNAS) or fila[0] not in tuple(LETRAS):
                raise ValueError(f"Fila incompatible en {ruta}, línea {lector.line_num}.")
            conteos[fila[0]] += 1
        return conteos


def extraer_coordenadas(mano):
    if len(mano.landmark) != 21:
        return None
    valores = []
    for punto in mano.landmark:
        if not (0 <= punto.x <= 1 and 0 <= punto.y <= 1 and math.isfinite(punto.z)):
            return None
        valores.extend((punto.x, punto.y, punto.z))
    return valores


def guardar_muestra(ruta, letra, participante, sesion, lateralidad, coordenadas):
    with ruta.open("a", encoding="utf-8", newline="") as archivo:
        csv.writer(archivo).writerow(
            [letra, participante, sesion, uuid4().hex, lateralidad, *coordenadas]
        )


def dibujar_confirmacion(cv2, imagen, texto, margen_inferior=24):
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    escala = 1.1
    ancho, alto = cv2.getTextSize(texto, fuente, escala, 2)[0]
    posicion = (max(0, (imagen.shape[1] - ancho) // 2), max(alto, imagen.shape[0] - margen_inferior))
    cv2.putText(imagen, texto, posicion, fuente, escala, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(imagen, texto, posicion, fuente, escala, (255, 255, 255), 2, cv2.LINE_AA)


def capturar(args, conteos):
    import cv2
    import mediapipe as mp

    ventana = "Captura HOLA"
    camara = cv2.VideoCapture(args.camara)
    lado = {"derecha": "Right", "izquierda": "Left"}[args.mano]
    letra = ""
    estado = "Selecciona H, O, L o A"
    ultima_captura = float("-inf")
    letra_guardada = ""
    try:
        if not camara.isOpened():
            raise RuntimeError(
                f"No se pudo abrir la cámara {args.camara}. Prueba Cámara de Windows "
                "y usa --camara 1 si tu webcam tiene otro índice."
            )

        manos = mp.solutions.hands
        dibujo = mp.solutions.drawing_utils
        with manos.Hands(max_num_hands=2, min_detection_confidence=0.75) as detector:
            while True:
                correcto, imagen = camara.read()
                if not correcto:
                    raise RuntimeError("La cámara dejó de entregar fotogramas.")

                imagen = cv2.flip(imagen, 1)
                resultado = detector.process(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
                detectadas = resultado.multi_hand_landmarks or []
                coordenadas = None
                mensaje_mano = f"Muestra solo la mano {args.mano}"
                if len(detectadas) == 1 and resultado.multi_handedness:
                    mano = detectadas[0]
                    dibujo.draw_landmarks(imagen, mano, manos.HAND_CONNECTIONS)
                    if resultado.multi_handedness[0].classification[0].label == lado:
                        coordenadas = extraer_coordenadas(mano)
                        mensaje_mano = "Mano lista" if coordenadas else "Encuadra toda la mano"

                resumen = "  ".join(f"{clase}: {conteos[clase]}" for clase in LETRAS)
                lineas = (
                    "H / O / L / A: elegir | ESPACIO: guardar | ESC: salir",
                    f"Letra: {letra or '-'} | {mensaje_mano}",
                    resumen,
                    estado,
                )
                for i, linea in enumerate(lineas, start=1):
                    posicion = (10, 22 * i)
                    cv2.putText(imagen, linea, posicion, cv2.FONT_HERSHEY_SIMPLEX, 0.49, (0, 0, 0), 3)
                    cv2.putText(imagen, linea, posicion, cv2.FONT_HERSHEY_SIMPLEX, 0.49, (255, 255, 255), 1)
                if monotonic() - ultima_captura < 1:
                    dibujar_confirmacion(cv2, imagen, f"{letra_guardada} guardada")
                cv2.imshow(ventana, imagen)
                tecla = cv2.waitKey(1) & 0xFF
                if tecla == 27 or cv2.getWindowProperty(ventana, cv2.WND_PROP_VISIBLE) < 1:
                    return
                seleccion = chr(tecla).upper()
                if seleccion in tuple(LETRAS):
                    letra = seleccion
                    estado = f"Forma {letra} y pulsa ESPACIO"
                elif tecla == 32:
                    if not letra or coordenadas is None:
                        estado = "Elige letra y muestra una mano valida"
                    elif monotonic() - ultima_captura >= 1:
                        guardar_muestra(args.csv, letra, args.participante, args.sesion, args.mano, coordenadas)
                        conteos[letra] += 1
                        ultima_captura = monotonic()
                        letra_guardada = letra
                        estado = f"{letra} guardada. Relaja la mano antes de repetir."
    finally:
        camara.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Captura las letras H, O, L y A con MediaPipe.")
    parser.add_argument("--participante", default="p01", help="Identificador de la persona (p01 por defecto).")
    parser.add_argument("--sesion", default="s01", help="Identificador de la sesión (s01 por defecto).")
    parser.add_argument("--mano", choices=("derecha", "izquierda"), default="derecha")
    parser.add_argument("--camara", type=int, default=0, help="Índice de cámara de OpenCV.")
    parser.add_argument("--csv", type=Path, default=CSV_PREDETERMINADO, help="Ruta del CSV de salida.")
    parser.add_argument("--solo-inicializar", action="store_true", help="Prepara el encabezado sin abrir la cámara.")
    args = parser.parse_args()

    try:
        conteos = preparar_csv(args.csv)
        print(f"CSV: {args.csv.resolve()} | {sum(conteos.values())} muestras existentes.", flush=True)
        if not args.solo_inicializar:
            capturar(args, conteos)
    except (OSError, RuntimeError, ValueError, ImportError, AttributeError) as error:
        parser.exit(1, f"Error de captura: {error}\n")


if __name__ == "__main__":
    main()
