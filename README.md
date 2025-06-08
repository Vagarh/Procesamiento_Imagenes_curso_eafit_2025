# Visión por Computador – Talleres EAFIT 2025

Este repositorio agrupa los cinco talleres del curso **SI7004_6621_2561: Visión por Computador** de la Maestría en Ciencias de Datos (Universidad EAFIT). Cada carpeta incluye los notebooks ejecutados y los datos mínimos necesarios para reproducir los resultados.

## Estructura

El repositorio agrupa cinco talleres de Visión por Computador, cada uno con su propósito, metodología y resultados principales:

1. **Taller 01 – Clasificación de Enfermedades de Mango**  
   Desarrollo de un clasificador SVM para detectar distintas patologías en hojas de mango.  
   **Resultados clave:**
   - **Accuracy:** 77,12 %  
   - **F1-Score global:** 0.772  
   - **Precisión:** > 90 % en clases seleccionadas  

   **Reporte de clasificación**  
   | Enfermedad         | Precisión | Recall | F1-score | Soporte |
   |--------------------|----------:|-------:|---------:|--------:|
   | Anthracnose        |      0.64  |  0.68  |   0.66   |      93 |
   | Bacterial Canker   |      0.79  |  0.92  |   0.85   |      91 |
   | Cutting Weevil     |      0.99  |  0.97  |   0.98   |     101 |
   | Die Back           |      0.93  |  0.86  |   0.89   |      90 |
   | Gall Midge         |      0.61  |  0.72  |   0.66   |      86 |
   | Healthy            |      0.69  |  0.74  |   0.71   |     104 |
   | Powdery Mildew     |      0.85  |  0.71  |   0.78   |     112 |
   | Sooty Mould        |      0.73  |  0.62  |   0.67   |     123 |
   | **Macro avg**      |      0.78  |  0.78  |   0.77   |     800 |
   | **Weighted avg**   |      0.78  |  0.77  |   0.77   |     800 |

2. **Taller 02 – Clasificación con AutoGluon**  
   Implementación de un pipeline automático con AutoGluon que aprovecha features extraídos por una CNN para clasificación de imágenes.

3. **Taller 03 – ViTT: Recuperación Texto → Imagen**  
   Fine-tuning ligero de un transformer multimodal para mejorar la búsqueda texto → imagen.  
   **Mejoras tras 3 épocas:**
   - **Recall@5 global:** 74 % → 87 %  
   - **Clase “hot dog”:** 42 % → 78 %  

   ![Ejemplo Taller 03](https://github.com/Vagarh/Procesamiento_Imagenes_curso_eafit_2025/blob/18657a98c0e7bdf23d499cb8ae6d3b0b8feac0b9/Imagenes/Taller_03.png)

4. **Taller 04 – Modelos Fundacionales y Tareas de Pretexto**  
   Exploración de arquitecturas auto-supervisadas (p. ej., masked autoencoders) y diseño de tareas de pretexto para aprender representaciones robustas en visión por computador.

5. **Taller 05 – Detección de Aviones y Coordenadas**  
   Entrenamiento de un modelo de detección de objetos para localizar aviones en imágenes aéreas.  
   **Rendimiento en validación:**
   - **IoU:** entre 0.82 y 0.91  

   ![Ejemplo Taller 05](https://github.com/Vagarh/Procesamiento_Imagenes_curso_eafit_2025/blob/18657a98c0e7bdf23d499cb8ae6d3b0b8feac0b9/Imagenes/Taller_5.png)

## Guía de uso

1. **Clonar el repositorio**  
   ```bash
   git clone https://github.com/Vagarh/Procesamiento_Imagenes_curso_eafit_2025.git
   cd Procesamiento_Imagenes_curso_eafit_2025
