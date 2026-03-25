# 👁️ VisionOMR

**VisionOMR** es un sistema avanzado de Reconocimiento Óptico de Marcas (OMR) desarrollado en Python, diseñado para procesar hojas de respuesta de exámenes de forma automática y precisa. Utiliza técnicas de **visión por computador con OpenCV** y está preparado para integrarse con modelos de **IA (Gemini)** para la interpretación final de resultados.

---

## 🚀 Descripción

VisionOMR permite tomar una imagen (foto o escaneo) de una hoja de respuestas, corregir su perspectiva, detectar burbujas marcadas y extraer información estructurada como respuestas e identificación del estudiante.

El sistema está optimizado para trabajar en condiciones reales: fotos con celular, iluminación variable, sombras y ruido.

---

## ✨ Características

- 📐 **Corrección automática de perspectiva**
  - Detecta 4 puntos de anclaje (cuadros negros)
  - Aplica transformación para alinear la hoja

- 🧪 **Preprocesamiento robusto**
  - Escala de grises
  - CLAHE (mejora de contraste)
  - Gaussian Blur
  - Sharpening

- ⚪ **Detección precisa de burbujas**
  - Uso de `HoughCircles`
  - Filtros por radio y distancia
  - Evaluación de contraste local

- 🧠 **Clasificación inteligente**
  - Diferencia entre burbujas llenas y vacías
  - Basado en intensidad interna vs entorno

- 📊 **Debug visual**
  - Genera imágenes con círculos detectados
  - Muestra valores de intensidad

- 🤖 **Integración opcional con Gemini**
  - Extrae ID del alumno
  - Interpreta respuestas
  - Genera JSON estructurado

---

## 🧠 Flujo del Sistema

1. Carga de imagen
2. Mejora de imagen (`enhance_image`)
3. Detección de contornos y esquinas
4. Corrección de perspectiva (`cut_img`)
5. Detección de burbujas (`detect_circles_precise`)
6. Clasificación (rellena / vacía)
7. (Opcional) Análisis con IA

---

## 🛠️ Requisitos

- Python 3.8+
- OpenCV
- NumPy

Opcional:
- Google Generative AI (Gemini)

---

## 📦 Instalación

Clonar el repositorio:

```bash
git clone https://github.com/tu-usuario/VisionOMR.git
cd VisionOMR
pip install opencv-python numpy
```

