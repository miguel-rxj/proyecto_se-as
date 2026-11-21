import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

from capturar_hola import COLUMNAS, CSV_PREDETERMINADO, LETRAS


MODELO = CSV_PREDETERMINADO.with_name("modelo_hola.joblib")


def normalizar(coordenadas):
    puntos = np.asarray(coordenadas, dtype=float).reshape(21, 3)
    puntos = puntos - puntos[0]
    escala = np.linalg.norm(puntos[9])
    if not np.isfinite(puntos).all() or escala <= 1e-8:
        raise ValueError("Coordenadas inválidas o escala de mano casi nula.")
    return (puntos / escala).ravel()


def leer_muestras(ruta):
    muestras = []
    descartadas = 0
    vistas = set()
    with ruta.open(encoding="utf-8", newline="") as archivo:
        lector = csv.DictReader(archivo)
        if lector.fieldnames != COLUMNAS:
            raise ValueError("El encabezado del CSV no coincide con el del capturador.")
        for fila in lector:
            try:
                letra = fila["etiqueta"]
                grupo = (fila["participante_id"], fila["sesion_id"])
                toma = fila["toma_id"]
                lado = fila["lateralidad"]
                puntos = tuple(float(fila[columna]) for columna in COLUMNAS[5:])
                if (
                    letra not in tuple(LETRAS) or lado not in ("derecha", "izquierda")
                    or not all(grupo) or not toma or None in fila or puntos in vistas
                ):
                    raise ValueError("Fila inválida o duplicada.")
                caracteristicas = normalizar(puntos)
            except (ValueError, TypeError):
                descartadas += 1
                continue
            vistas.add(puntos)
            muestras.append((letra, grupo, toma, lado, caracteristicas))
    return muestras, descartadas


def separar(muestras, clases):
    participantes = {grupo[0] for _, grupo, *_ in muestras}
    sesiones = {grupo for _, grupo, *_ in muestras}
    if len(sesiones) > 1:
        por_persona = len(participantes) > 1
        grupos = [grupo[0] if por_persona else grupo for _, grupo, *_ in muestras]
        ids = {grupo: indice for indice, grupo in enumerate(dict.fromkeys(grupos))}
        particiones = GroupShuffleSplit(n_splits=100, test_size=0.2, random_state=42)
        for entrenamiento, prueba in particiones.split(clases, groups=[ids[g] for g in grupos]):
            if set(clases[entrenamiento]) == set(clases[prueba]) == set(LETRAS):
                unidad = "participante" if por_persona else "sesión"
                return entrenamiento, prueba, f"Separación por {unidad} completo."
        raise ValueError("No hay una separación por grupos con las cuatro letras en ambos conjuntos.")

    entrenamiento, prueba = [], []
    for letra in LETRAS:
        indices = np.flatnonzero(clases == letra)
        limite = len(indices) - max(1, int(np.ceil(0.2 * len(indices))))
        entrenamiento.extend(indices[:limite])
        prueba.extend(indices[limite:])
    return entrenamiento, prueba, "Evaluación preliminar. Último 20 % de cada letra de la única sesión."


def entrenar(ruta_csv, ruta_modelo):
    muestras, descartadas = leer_muestras(ruta_csv)
    conteos = Counter(muestra[0] for muestra in muestras)
    if any(conteos[letra] < 2 for letra in LETRAS):
        raise ValueError("Se necesitan al menos dos muestras válidas de cada letra.")
    lados = {muestra[3] for muestra in muestras}
    if len(lados) != 1:
        raise ValueError("Usa un dataset con un solo lado de mano, igual que en la captura.")
    clases = np.asarray([muestra[0] for muestra in muestras])
    datos = np.stack([muestra[4] for muestra in muestras])
    entrenamiento, prueba, metodo = separar(muestras, clases)
    modelo = RandomForestClassifier(n_estimators=200, random_state=42)
    modelo.fit(datos[entrenamiento], clases[entrenamiento])
    predicciones = modelo.predict(datos[prueba])
    orden = list(LETRAS)
    metricas = classification_report(clases[prueba], predicciones, labels=orden, output_dict=True, zero_division=0)
    informe = {
        "separacion": metodo,
        "muestras_validas": dict(conteos),
        "filas_descartadas": descartadas,
        "entrenamiento": dict(Counter(clases[entrenamiento])),
        "prueba": dict(Counter(clases[prueba])),
        "exactitud": accuracy_score(clases[prueba], predicciones),
        "f1_macro": metricas["macro avg"]["f1-score"],
        "orden_matriz": orden,
        "matriz_confusion": confusion_matrix(clases[prueba], predicciones, labels=orden).tolist(),
        "metricas_por_letra": {letra: metricas[letra] for letra in orden},
        "tomas_entrenamiento": [muestras[i][2] for i in entrenamiento],
        "tomas_prueba": [muestras[i][2] for i in prueba],
        "scikit_learn": sklearn.__version__,
    }
    ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modelo": modelo, "mano": lados.pop(), "columnas": COLUMNAS[5:]}, ruta_modelo)
    ruta_informe = ruta_modelo.with_suffix(".metricas.json")
    ruta_informe.write_text(json.dumps(informe, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metodo)
    print(f"Entrenamiento: {len(entrenamiento)} | Prueba: {len(prueba)} | Descartadas: {descartadas}")
    print(f"Exactitud: {informe['exactitud']:.2%} | F1 macro: {informe['f1_macro']:.4f}")
    print(f"Modelo: {ruta_modelo}\nMétricas: {ruta_informe}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena el Random Forest para H, O, L y A.")
    parser.add_argument("--csv", type=Path, default=CSV_PREDETERMINADO)
    parser.add_argument("--modelo", type=Path, default=MODELO)
    args = parser.parse_args()
    try:
        entrenar(args.csv, args.modelo)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error de entrenamiento: {error}\n")
