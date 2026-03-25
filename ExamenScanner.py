import cv2
#from google import genai
#from google.genai import types
import numpy as np
import json
import pathlib

class ExamenScanner:
    
    def __init__(self, imagen:str):
        self.imagen = imagen
        
    
    def enhance_image(self):
        
        try:
            
            self.imagen = cv2.imread(self.imagen)
            if self.imagen is None:
                raise FileNotFoundError('imagen no existe')
        except:
            raise FileNotFoundError('La imagen no existe en la ruta especificada')
        gray = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4,4))
        gray = clahe.apply(gray)
        
        
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        sharpened = cv2.addWeighted(gray, 1.3, blur, -0.3, 0)
        
        

        return sharpened
    
    def enhance_image_post(self, image):

        # Verificar si ya es escala de grises ## OPCIONAL, si ya es gris no convertir
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        sharpened = cv2.addWeighted(gray, 2.2, blur, -1.2, 0)

        _, dark_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        sharpened = cv2.bitwise_and(sharpened, sharpened, mask=cv2.bitwise_not(dark_mask))
        sharpened[dark_mask == 255] = 0

        return sharpened
    
    def cut_img(self):
        img = self.enhance_image()
        thresh = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cuadros = []
        h_img, w_img = img.shape[:2]
        area_img = h_img * w_img

        for c in contours:
            area = cv2.contourArea(c)
            
            if area_img * 0.0005 < area < area_img * 0.02:
                x, y, w, h = cv2.boundingRect(c)
                ratio = w / float(h)
                
                if 0.5 < ratio < 2.0:
                    
                    roi = img[y:y+h, x:x+w]
                    if np.mean(roi) < 80:
                        cx = x + w // 2
                        cy = y + h // 2
                        cuadros.append((cx, cy))

        if len(cuadros) < 4:
            raise Exception(f"Solo se detectaron {len(cuadros)} cuadros, se necesitan 4")

        
        if len(cuadros) > 4:
            corners = [
                (0, 0), (w_img, 0),
                (w_img, h_img), (0, h_img)
            ]
            selected = []
            for corner in corners:
                closest = min(cuadros, key=lambda p: (p[0]-corner[0])**2 + (p[1]-corner[1])**2)
                selected.append(closest)
                cuadros.remove(closest)
            cuadros = selected

        
        cuadros = sorted(cuadros, key=lambda p: (p[1], p[0]))
        top = sorted(cuadros[:2], key=lambda p: p[0])
        bottom = sorted(cuadros[-2:], key=lambda p: p[0])
        tl, tr = top
        bl, br = bottom

        pts = np.array([tl, tr, br, bl], dtype="float32")

        widthA = np.linalg.norm(np.array(br) - np.array(bl))
        widthB = np.linalg.norm(np.array(tr) - np.array(tl))
        width = int(max(widthA, widthB))

        heightA = np.linalg.norm(np.array(tr) - np.array(br))
        heightB = np.linalg.norm(np.array(tl) - np.array(bl))
        height = int(max(heightA, heightB))

        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(pts, dst)
        warp = cv2.warpPerspective(img, M, (width, height))
        
        #warp_mejorado = self.enhance_image_post(warp)
        
        cv2.imwrite("corte_y_mejora.jpg", warp)
        return warp
    
    def detect_circles_precise(self):
        try:
            warp = self.cut_img()
            output_color = cv2.cvtColor(warp, cv2.COLOR_GRAY2BGR)
            h_img, w_img = warp.shape[:2]

            # ── 1. Detectar cuadros negros de referencia (encabezado) ──────────
            # Los cuadros pequeños ■ antes de A B C D son anclas de posición
            thresh = cv2.adaptiveThreshold(
                warp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

           
            area_img = h_img * w_img
            cuadros_ref = []
            for c in contours:
                area = cv2.contourArea(c)
                x, y, w, h = cv2.boundingRect(c)
                if area_img * 0.0001 < area < area_img * 0.005:
                    ratio = w / float(h)
                    if 0.6 < ratio < 1.6:
                        roi = warp[y:y+h, x:x+w]
                        if np.mean(roi) < 60: 
                            cuadros_ref.append((x + w//2, y + h//2, w, h))

            print(f"Cuadros de referencia encontrados: {len(cuadros_ref)}")

            
            blur = cv2.GaussianBlur(warp, (5, 5), 0)
            circles = cv2.HoughCircles(
                blur,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=18,
                param1=60,      # antes 50 — más estricto en bordes
                param2=30,      # antes 25 — más estricto en acumulador
                minRadius=10,   # antes 8
                maxRadius=22    # antes 25
            )

            if circles is None:
                print("HoughCircles no encontró círculos")
                cv2.imwrite("precision_deteccion.jpg", output_color)
                return output_color, []

            circles = np.round(circles[0, :]).astype("int")
            print(f"HoughCircles encontró: {len(circles)} círculos")

            
            candidatos_rellenos = []
            candidatos_vacios = []

            for (cx, cy, r) in circles:
                mask_in = np.zeros(warp.shape, dtype="uint8")
                cv2.circle(mask_in, (cx, cy), max(r - 4, 2), 255, -1)
                mean_in = cv2.mean(warp, mask=mask_in)[0]

                
                mask_zona = np.zeros(warp.shape, dtype="uint8")
                cv2.circle(mask_zona, (cx, cy), r * 3, 255, -1)  # zona amplia alrededor
                cv2.circle(mask_zona, (cx, cy), r + 2, 0, -1)     # excluir la burbuja misma
                mean_zona = cv2.mean(warp, mask=mask_zona)[0]
                
                diferencia_local = mean_zona - mean_in
                # Justo antes del if es_rellena, agrega:
                if 25 < diferencia_local < 40 and mean_in < 150:
                    print(f"CASI RELLENA: cx:{cx} cy:{cy} | mean_in:{mean_in:.0f} | diff_local:{diferencia_local:.0f}")
                es_rellena = diferencia_local > 36 and mean_in < 150
                

                if es_rellena:
                    candidatos_rellenos.append((cx, cy, r, mean_in))
                    cv2.circle(output_color, (cx, cy), r, (0, 255, 0), 2)
                else:
                    candidatos_vacios.append((cx, cy, r, mean_in))
                    cv2.circle(output_color, (cx, cy), r, (100, 100, 100), 1)

                # Mostrar valor
                cv2.putText(output_color, f"{mean_in:.0f}",
                        (cx - 12, cy - r - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                        (0, 200, 0) if es_rellena else (80, 80, 80), 1)

            print(f"Burbujas rellenas detectadas: {len(candidatos_rellenos)}")
            print(f"Burbujas vacías detectadas: {len(candidatos_vacios)}")

            
            if candidatos_rellenos:
                vals = [c[3] for c in candidatos_rellenos]
                print(f"Mean interior rellenas — min:{min(vals):.0f} max:{max(vals):.0f} avg:{np.mean(vals):.0f}")
            if candidatos_vacios:
                vals = [c[3] for c in candidatos_vacios]
                print(f"Mean interior vacías  — min:{min(vals):.0f} max:{max(vals):.0f} avg:{np.mean(vals):.0f}")

            burbujas_finales = [
                {"posicion": (cx, cy), "id": i}
                for i, (cx, cy, r, _) in enumerate(candidatos_rellenos)
            ]

            cv2.imwrite("precision_deteccion.jpg", output_color)
            return output_color, burbujas_finales

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return None, []
        
    """   
    def analizar_hoja_respuestas(self):
        
        
        try:
            client = genai.Client(api_key=self.apikey)
            image_path = pathlib.Path('precision_deteccion.jpg')
            if not image_path.is_file():
                raise FileNotFoundError(f"No se encontró el archivo de imagen en {image_path}")
            image_bytes = image_path.read_bytes()
            
            image_part = types.Part.from_bytes(
            data=image_bytes, 
            mime_type="image/jpeg")
            
            prompt_texto = 
            Eres un lector experto de hojas de respuestas tipo scantron.
            Analiza esta imagen de una hoja de respuestas (OMR).
            
            Tu tarea es extraer dos cosas:
            1. El 'ID ALUMNO' (la cuadrícula de identificación). Identifica qué número (1-10) ,
            la segunda columna identifica qué número (1-10), la tercera y cuarta son unidas (01-20).
            las columnas 3 y 4 se cuentan de hacia abajo es decir la 4 comensaria en 11
            IMPORTANTE:
            - Comiensan en 1.
            
            2. Las respuestas marcadas para cada pregunta numerada. Indica la letra (A, B, C, D) de la burbuja rellenada Analiza la imagen y extrae TODAS las preguntas con su respuesta marcada.

            REGLAS:
            - Burbuja marcada = OSCURA/GRIS/CIRCULO VERDE/RELLENA
            
            IMPORTANTE: si no tiene marca OSCURA/GRIS/CIRCULO VERDE/RELLENA, tienes que ser muy preciso marcar null
            - Sin marca: null
            - Doble marca: null

            Devuelve el resultado EXCLUSIVAMENTE en formato JSON válido, sin texto explicativo adicional, ni bloques de código de Markdown.
            El formato JSON debe ser:
            {
            "informacion_estudiante": {
                "id_alumno": "XXXX"
            },
            "respuestas": {
            "1": "A",
            "2": "C",
            "3": null
            ...
            }
            }
            

            response = client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=[image_part, prompt_texto],
            config=types.GenerateContentConfig(
                response_mime_type="application/json", ))


            try:
                print("\nAnálisis completado con éxito.")
                result = json.loads(response.text)
                #print(result)
                
                with open("resultado_analisis.json", "w") as f:
                    json.dump(result, f, indent=4)
                
                
                
                
            except json.JSONDecodeError as e:
                print(f"\nError: La API no devolvió un JSON válido. Respuesta recibida:\n{response.text}")
                print(f"Detalle del error: {e}")
            

        except FileNotFoundError:
            print(f"Error: No se encontró el archivo de imagen en ")
            
        except Exception as e:
            print(f"Ocurrió un error inesperado: {e}")
        """
    
    


    def run(self):
        self.detect_circles_precise()

    

       
        

if __name__ == '__main__':
    
    # PONER EL API KEY_____
    
    APIKEY:str = ''
    result = ExamenScanner(imagen='7.jpeg').run()