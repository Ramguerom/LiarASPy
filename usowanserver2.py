# server.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import usowan2 as usowan
import cv2
import numpy as np
import base64

app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['POST'])
def upload_and_process_image():
    data = request.get_json()
    if ('image' not in data) and ('tablero' not in data):
        return jsonify({'error': 'No image/board file in request'}), 400

    if ('image' in data):
        try:
            base64_str = data['image'].split(',')[1]  # quitar "data:image/png;base64,"
            img_data = np.frombuffer(base64.b64decode(base64_str), np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

            if img is None:
                return jsonify({'error': 'Cannot decode image'}), 400
            imagen_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            imagen_bgr = usowan.mejorar_contraste_saturacion(imagen_bgr)
            imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)

            #procesar imagen
            (n,esquinas) = usowan.extrae_esquinas(imagen_rgb)
            celdas = usowan.extrae_valores(imagen_rgb, esquinas, n)
            regiones = usowan.extrae_regiones(imagen_rgb,celdas,esquinas)

            regionesd = dict()
            for ir,r in enumerate(regiones):
                for (X,Y) in r:
                    regionesd[(X,Y)] = ir 

            tablero = [[{'valor':int(celdas[(i,j)]), 'region':int(regionesd[(i,j)])} for i in range(n-1)] for j in range(n-1)]

            return jsonify({'tablero':tablero})


            # Resolver el tablero
            # img_res = usowan.resuelve_tablero(esquinas, celdas, regiones, imagen_rgb)

            # if (img_res == imagen_rgb).all():
            #     return jsonify({'no_solution': 'no_solution'})

            # # Codificar a PNG en memoria
            # _, buffer = cv2.imencode('.png', img_res)

            # # Codificar a base64
            # img_base64 = base64.b64encode(buffer).decode('utf-8')

            # return jsonify({'image': f'data:image/png;base64,{img_base64}'})
        
        
        except Exception as e:

            return jsonify({'error': str(e)}), 500
        

    if ('tablero' in data):
        #print(data)
        try:
            tablero = data["tablero"]
            MODELS = usowan.resuelve_tablero2(tablero)

            #print(MODELS)

            if len(MODELS) > 0:
                return jsonify({"solution":True, 'negras':MODELS[0][0], "mienten":MODELS[0][1]})
            
            else:
                return jsonify({"solution":False})
        
        except Exception as e:

            return jsonify({'error': str(e)}), 500
        
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
