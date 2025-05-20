import cv2
import numpy as np
#import matplotlib.pyplot as plt
#import os
import keras
import clingo
from math import sqrt
from functools import cmp_to_key
from sklearn.cluster import DBSCAN

def mejorar_contraste_saturacion(img, alpha=1.1, beta=10, saturacion=1.5):

    # --- Paso 1: Aumentar contraste (RGB) ---
    # Nuevo píxel = alpha * píxel + beta
    img_contraste = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # --- Paso 2: Aumentar saturación (HSV) ---
    hsv = cv2.cvtColor(img_contraste, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Incrementar saturación
    hsv[...,1] *= saturacion
    hsv[...,1] = np.clip(hsv[...,1], 0, 255)

    img_saturada = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return img_saturada

def keep_black(image):
    # Load image and convert to HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define black color range in HSV
    lower_black = np.array([0, 0, 0])        # Min (H=any, S=0, V=0)
    upper_black = np.array([180, 255, 50])   # Max (H=any, S=255, V=50)

    no_black_pixels = np.where(
    (image_hsv[:, :, 2] > 100)  # V (brillo) mayor que umbral
       # Opcional: también considerar saturación
    )

    image_res = image.copy()
    image_res[no_black_pixels] = [255.0,255.0,255.0]

    return cv2.cvtColor(image_res, cv2.COLOR_RGB2GRAY)

def adapta_img(img, size=(200,200)):
    img_res = cv2.resize(img.copy(), size, interpolation = cv2.INTER_CUBIC)
    return np.reshape(keep_black(img_res), (size[0], size[1], 1))

def extrae_horizontales(bwimage):
  hedges = bwimage.copy().astype("uint8")
  kernel = np.ones((1,7),np.uint8)
  hedges = cv2.dilate(hedges,kernel,iterations = 7)
  hedges = cv2.medianBlur(hedges, 7)
  kernel = np.ones((1,9),np.uint8)
  hedges = cv2.erode(hedges,kernel,iterations = 7)
  hedges = cv2.medianBlur(hedges, 9)
  return hedges


def extrae_verticales(bwimage):
  vedges = bwimage.copy().astype("uint8")
  kernel = np.ones((7,1),np.uint8)
  vedges = cv2.dilate(vedges,kernel,iterations = 7)
  vedges = cv2.medianBlur(vedges, 7)
  kernel = np.ones((9,1),np.uint8)
  vedges = cv2.erode(vedges,kernel,iterations = 7)
  vedges = cv2.medianBlur(vedges, 9)
  return vedges

def point_less(p1,p2):
    [y1,x1] = p1
    [y2,x2] = p2
    if (y2 - y1) > 30:
        return -1
    if (abs (y2 - y1)) <= 30 and ((x2 - x1) > 30):
        return -1
    return 1

def extrae_esquinas(rgbimage):
  bwimage = keep_black(rgbimage)
  vedges = extrae_verticales(bwimage)
  hedges = extrae_horizontales(bwimage)
  interseccion = cv2.bitwise_or(vedges, hedges)
  # Umbralizar la imagen para detectar los puntos (suponiendo que los puntos son negros sobre fondo blanco)
  _, thresh = cv2.threshold(interseccion, 100, 255, cv2.THRESH_BINARY_INV)

  # Encontrar coordenadas de los puntos negros
  coords = np.column_stack(np.where(thresh > 0))

  # CON DBSCAN (CON ESTO, PODRÍAMOS OBTENER EL TAMAÑO DEL TABLERO SIN CONOCERLO ANTICIPADAMENTE)
  # Aplicar DBSCAN
  dbscan = DBSCAN(eps=30, min_samples=1)  # Ajustar eps según la separación de los puntos
  labels = dbscan.fit_predict(coords)

  # Obtener los centroides de los clusters encontrados
  unique_labels = set(labels) - {-1}  # Excluir ruido (-1)
  centroids = np.array([coords[labels == k].mean(axis=0) for k in unique_labels])
  centroids = np.vectorize(lambda x: int(round(x)))(centroids)

  centroids_sort = sorted(centroids.tolist(), key=cmp_to_key(point_less))
  n = int(sqrt(len(centroids)))
  d={(j,i):[centroids_sort[n*i+j],centroids_sort[n*i+j+1],
            centroids_sort[n*(i+1)+j],centroids_sort[n*(i+1)+j+1]
          ] for i in range(n-1) for j in range(n-1)}
  return (n,d)


def extrae_valores(rgbimage, esquinas, n):
  # EXTRACCIÓN DEL NÚMERO DE CADA CELDA
  # USO DE UN MODELO (CNN) PARA LA CLASIFICACIÓN

  model = keras.models.load_model("printed_digits_detector.h5")
  celdas = dict()

  for i in range (n-1):
      for j in range (n-1):
          y_inicio , x_inicio = esquinas[(i,j)][0]
          y_fin, x_fin = esquinas[(i,j)][-1]
          recorte = adapta_img(rgbimage[int(y_inicio):int(y_fin), int(x_inicio):int(x_fin),:])
          clase = np.argmax(model.predict(np.array([recorte]),verbose = 0)[0])
          celdas[(i,j)] = clase
  return celdas

from skimage.color import deltaE_ciede2000, rgb2lab

def color_distance_ciede2000(color1, color2):
    # Convertir de RGB a LAB
    color1_lab = rgb2lab(np.array([[color1]], dtype=np.uint8) / 255.0)
    color2_lab = rgb2lab(np.array([[color2]], dtype=np.uint8) / 255.0)

    # Calcular la diferencia de color
    return deltaE_ciede2000(color1_lab[0, 0], color2_lab[0, 0])

import cv2
import numpy as np

def rgb_to_lab(rgb):
    rgb_array = np.uint8([[rgb]])
    lab = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2Lab)
    return lab[0][0]

def delta_e_cielab(color1, color2):
    lab1 = rgb_to_lab(color1).astype(float)
    lab2 = rgb_to_lab(color2).astype(float)
    return np.linalg.norm(lab1 - lab2)

def is_adyacent(cpos1,cpos2):
    return (abs(cpos1[0]-cpos2[0]) + abs(cpos1[1]-cpos2[1]))==1

def get_dominant_rgb(image):
    # Separar los canales de color
    r_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    g_hist = cv2.calcHist([image], [1], None, [256], [0, 256])
    b_hist = cv2.calcHist([image], [2], None, [256], [0, 256])

    # Obtener los valores de R, G, B más frecuentes
    r = np.argmax(r_hist)
    g = np.argmax(g_hist)
    b = np.argmax(b_hist)
    return (r,g,b)

def extrae_regiones(rgbimage, celdas, esquinas):
  dominants_rgb = dict()
  for c in celdas:
    y_inicio , x_inicio = esquinas[c][0]
    y_fin, x_fin = esquinas[c][-1]
    image = rgbimage[int(y_inicio)+10:int(y_fin)-10, int(x_inicio)+10:int(x_fin)-10,:]
    dominants_rgb[c] = get_dominant_rgb(image)

  REGIONS = []
  for c in celdas:
      added = False
      for i,r in enumerate(REGIONS):
          if any ((is_adyacent(c,cr) and delta_e_cielab(dominants_rgb[c], dominants_rgb[cr]) < 15) for cr in REGIONS[i]):
              REGIONS[i] += [c]
              added = True
      if not(added):
          REGIONS.append([c])

  return REGIONS



from functools import partial


def resuelve_tablero(esquinas, celdas, regiones, img):

  def on_model_img(model, imagen_res=None):
    NEGRAS = [tuple(x.number for x in atom.arguments) for atom in model.symbols(shown=True) if atom.name == "negra"]
    for (i,j) in NEGRAS:
      y_inicio , x_inicio = esquinas[(i,j)][0]
      y_fin, x_fin = esquinas[(i,j)][-1]
      imagen_res[int(y_inicio):int(y_fin), int(x_inicio):int(x_fin)] = [100,100,100]
    MIENTEN = [tuple(x.number for x in atom.arguments) for atom in model.symbols(shown=True) if atom.name == "miente"]
    for (i,j) in MIENTEN:
      y1, x1 = esquinas[(i,j)][0]
      y2, x2 = esquinas[(i,j)][1]
      y3, x3 = esquinas[(i,j)][2]
      y4, x4 = esquinas[(i,j)][3]
      cv2.line(imagen_res, (x1,y1), (x4,y4), color=(255,0,0), thickness=3)
      cv2.line(imagen_res, (x2,y2), (x3,y3), color=(255,0,0), thickness=3)

  img_res = img.copy()
  on_model = partial(on_model_img, imagen_res=img_res)

  reglas_juego = """% Relación de adyacencia (4-conex).
  ady(X,Y,X1,Y1) :- casilla(X,Y,_), casilla(X1,Y1,_), |X - X1| + |Y - Y1| = 1.

  % R1: Dos casillas adyacentes no pueden estar simultáneamente sombreadas
  :- ady(X,Y,X1,Y1), negra(X,Y), negra(X1,Y1).

  % R2: Todas lxas casillas blancas han de estar conectadas por otras blancas

  % Relación de blanco-conexión
  bcon(X,Y,X1,Y1) :- ady(X,Y,X1,Y1), not negra(X,Y), not negra(X1,Y1).
  bcon(X,Y,X2,Y2) :- bcon(X,Y,X1,Y1), bcon(X1,Y1,X2,Y2).

  :- casilla(X,Y,_), casilla(X1,Y1,_), not negra(X,Y), not negra(X1,Y1), not bcon(X,Y,X1,Y1).

  % R3: En cada región hay, exactamente, un mentiroso.

  % Definición de mentiroso.
  miente(X,Y) :- numero(X,Y,V), V != #count {(X1,Y1) : ady(X,Y,X1,Y1), negra(X1,Y1)}.

  :- casilla(_,_,R), 1 != #count {(X,Y) : miente(X,Y), casilla(X,Y,R)}.

  % Para cada casilla no numerada se ha de elegir si es negra o no.

  0 {negra(X,Y)} 1 :- casilla(X,Y,_).
  - negra(X,Y) :- casilla(X,Y,_), numero(X,Y,_).

  #show negra/2.
  #show miente/2."""

  repr_tablero="% Declaración de casillas \n"

  for ir,r in enumerate(regiones):
      repr_tablero += " ".join(f"casilla({X},{Y},{ir})." for (X,Y) in r) + "\n"


  repr_tablero += "\n % Declaración de los números \n" + " ".join(f"numero({X},{Y},{V})." for ((X,Y),V) in celdas.items() if V < 5)

  # Crear instancia de Clingo
  ctl = clingo.Control()

  # Añadir parte fija (reglas generales)
  ctl.add("base", [], reglas_juego)

  # Añadir parte dinámica (datos específicos)
  ctl.add("base", [], repr_tablero)

  # Ground y resolver
  ctl.ground([("base", [])])
  ctl.solve(on_model=on_model, yield_=False)

  return img_res


def resuelve_tablero2(tablero):

  MODELS = []


  def on_model(model):
    #print("paso.")

    nonlocal MODELS

    NEGRAS = [tuple(x.number for x in atom.arguments) for atom in model.symbols(shown=True) if atom.name == "negra"]
    MIENTEN = [tuple(x.number for x in atom.arguments) for atom in model.symbols(shown=True) if atom.name == "miente"]
    MODELS.append((NEGRAS, MIENTEN))

  #on_model = partial(on_model_img, MODELS)

  reglas_juego = """% Relación de adyacencia (4-conex).
  ady(X,Y,X1,Y1) :- casilla(X,Y,_), casilla(X1,Y1,_), |X - X1| + |Y - Y1| = 1.

  % R1: Dos casillas adyacentes no pueden estar simultáneamente sombreadas
  :- ady(X,Y,X1,Y1), negra(X,Y), negra(X1,Y1).

  % R2: Todas lxas casillas blancas han de estar conectadas por otras blancas

  % Relación de blanco-conexión
  bcon(X,Y,X1,Y1) :- ady(X,Y,X1,Y1), not negra(X,Y), not negra(X1,Y1).
  bcon(X,Y,X2,Y2) :- bcon(X,Y,X1,Y1), bcon(X1,Y1,X2,Y2).

  :- casilla(X,Y,_), casilla(X1,Y1,_), not negra(X,Y), not negra(X1,Y1), not bcon(X,Y,X1,Y1).

  % R3: En cada región hay, exactamente, un mentiroso.

  % Definición de mentiroso.
  miente(X,Y) :- numero(X,Y,V), V != #count {(X1,Y1) : ady(X,Y,X1,Y1), negra(X1,Y1)}.

  :- casilla(_,_,R), 1 != #count {(X,Y) : miente(X,Y), casilla(X,Y,R)}.

  % Para cada casilla no numerada se ha de elegir si es negra o no.

  0 {negra(X,Y)} 1 :- casilla(X,Y,_).
  - negra(X,Y) :- casilla(X,Y,_), numero(X,Y,_).

  #show negra/2.
  #show miente/2."""

  repr_tablero = "% Definición del tablero:\n"

  for i,f in enumerate(tablero):
     for j,c in enumerate(f):
        repr_tablero +=f"casilla({i},{j},{c["region"]}).\n"
        
  for i,f in enumerate(tablero):
     for j,c in enumerate(f):
        if c["valor"] < 5:
          repr_tablero +=f"numero({i},{j},{c["valor"]}).\n"


  #print(repr_tablero)

  # Crear instancia de Clingo
  ctl = clingo.Control()

  # Añadir parte dinámica (datos específicos)
  ctl.add("base", [], repr_tablero)

  # Añadir parte fija (reglas generales)
  ctl.add("base", [], reglas_juego)

  # Ground y resolver
  ctl.ground([("base", [])])
  ctl.solve(on_model=on_model, yield_=False)

  return MODELS
