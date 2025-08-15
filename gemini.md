# 🤖 Instrucciones para el Asistente de IA (Gemini)
## Misión Actual: Análisis y Refactorización del Proyecto Python/Jupyter

Hola Gemini. Este archivo define el estado actual de mi proyecto y mis objetivos para refactorizarlo y alinearlo con las mejores prácticas en el contexto de proyectos de Python y Jupyter Notebooks. Úsalo como tu guía principal para analizar el código y proporcionar sugerencias de mejora.

🤖 Directiva para Asistente de IA: Análisis y Refactorización de Código
Misión: Mejora de Calidad y Arquitectura de Software para Proyectos Python/Jupyter
Resumen: Este documento sirve como directiva central para el asistente de IA. Su propósito es definir el contexto, los objetivos y las tareas solicitadas para el análisis y refactorización de un proyecto de software basado en Python y Jupyter Notebooks. Úsalo como la fuente principal de verdad para entender el estado actual y los objetivos de mejora.

PARTE I: CONTEXTO DEL PROYECTO (ESTADO ACTUAL)
1. Descripción General
Nombre del Proyecto: "Procesamiento de Imágenes - Curso EAFIT 2025"

Propósito Principal: "Colección de notebooks y scripts para el procesamiento y análisis de imágenes, incluyendo tareas de clasificación y detección."

Dominio de Negocio: "Visión por Computadora, Aprendizaje Automático, Educación"

2. Stack Tecnológico
Lenguaje/Framework Principal: "Python, Jupyter Notebooks"

Librerías Clave: "NumPy, Pandas, scikit-learn, OpenCV, TensorFlow/Keras, Matplotlib, Seaborn"

Testing: "N/A (o aserciones básicas dentro de notebooks)"

Herramientas Clave: "Jupyter, Git"

3. Estructura del Repositorio
Describe la disposición actual de los directorios para establecer un punto de partida.

/
├── legacy/             # Contiene notebooks y experimentos más antiguos
├── notebooks/          # Contiene notebooks actuales/activos
├── Imagenes/           # Imágenes de ejemplo
├── README.md
├── gemini.md
└── .git/...

4. Convenciones y Estándares (Si existen)
Estilo de Código: "Informal, varía por notebook."

Convenciones de Nomenclatura: "Informal, varía por notebook."

Gestión de Dependencias: "No gestionadas explícitamente (e.g., requirements.txt ausente o incompleto)."

PARTE II: OBJETIVOS DE LA REFACTORIZACIÓN
🎯 5. Misión Principal
El objetivo central es auditar y refactorizar el código Python y los Jupyter notebooks para mejorar su calidad, modularidad, reusabilidad y mantenibilidad. Buscamos transformar el proyecto en un sistema más robusto y alineado con las mejores prácticas de desarrollo de software para proyectos de ciencia de datos y aprendizaje automático.

⚠️ 6. Diagnóstico: Puntos de Dolor y Deuda Técnica
Lista de problemas conocidos o sospechados que motivan esta iniciativa.

Ejemplo 1 (Modularidad): "Lógica de procesamiento de imágenes y modelos mezclada directamente en los notebooks, dificultando la reusabilidad."

Ejemplo 2 (DRY): "Falta de modularización: funciones y clases útiles no están encapsuladas para ser importadas y reutilizadas."

Ejemplo 3 (Consistencia): "Manejo inconsistente de datos y rutas de archivos."

Ejemplo 4 (Testing): "Ausencia de pruebas unitarias o de integración para la lógica crítica."

Ejemplo 5 (Dependencias): "Dependencias no gestionadas explícitamente (e.g., requirements.txt missing or incomplete)."

Ejemplo 6 (Legibilidad): "Notebooks largos y difíciles de seguir, con múltiples pasos de experimentación y visualización."

🏆 7. Criterios de Éxito y Objetivos Específicos
Definición clara de cómo se ve el estado final deseado.

Ejemplo 1 (Modularidad): "Extraer funciones y clases reutilizables de los notebooks a módulos Python (.py files)."

Ejemplo 2 (Estructura): "Implementar una estructura de proyecto más organizada (e.g., src directory for modules, data for datasets, notebooks for analysis)."

Ejemplo 3 (Dependencias): "Establecer un sistema de gestión de dependencias (e.g., requirements.txt)."

Ejemplo 4 (Calidad): "Aumentar la cobertura de pruebas para la lógica de procesamiento y modelado."

Ejemplo 5 (Mantenibilidad): "Mejorar la legibilidad y mantenibilidad de los notebooks, enfocándolos en la narrativa y el análisis."

📐 8. Principios Guía para el Análisis
Principios de diseño que deben guiar todas las sugerencias y refactorizaciones.

SOLID:
SRP (Single Responsibility): Cada clase o módulo debe tener una única razón para cambiar.
OCP (Open/Closed): Las entidades de software deben estar abiertas a la extensión, pero cerradas a la modificación.
LSP (Liskov Substitution): Los subtipos deben ser sustituibles por sus tipos base.
ISP (Interface Segregation): Es mejor tener muchas interfaces específicas que una sola de propósito general.
DIP (Dependency Inversion): Los módulos de alto nivel no deben depender de los de bajo nivel; ambos deben depender de abstracciones.

DRY (Don't Repeat Yourself): Evitar la duplicación de código mediante abstracciones.

KISS (Keep It Simple, Stupid): Preferir soluciones sencillas y claras sobre las complejas.

Bajo Acoplamiento y Alta Cohesión: Minimizar las dependencias entre módulos y asegurar que el contenido de un módulo esté fuertemente relacionado.

PARTE III: COLABORACIÓN Y MODO DE TRABAJO
📋 9. Tareas Solicitadas
Cuando se solicite asistencia, las tareas principales serán:

Análisis de Arquitectura:
Evaluar la estructura actual del proyecto y proponer mejoras (ej: modularización, organización de directorios).
Identificar violaciones de los límites entre módulos y sugerir cómo corregirlas.
Señalar módulos con alto acoplamiento o baja cohesión.

Revisión de Código (Code Review):
Analizar fragmentos o archivos de código (incluyendo notebooks) en busca de "code smells" y antipatrones.
Sugerir aplicaciones concretas de los principios guía (SOLID, DRY, etc.) sobre el código proporcionado.
Recomendar patrones de diseño (ej: Factory, Strategy para modelos/preprocesamiento) para resolver problemas específicos.

Generación de Código Refactorizado:
A partir de un bloque de código (de notebook o script), generar una versión mejorada, explicando claramente los cambios y los beneficios obtenidos.
Asistir en la creación de nuevas estructuras (clases, funciones, módulos Python) y en la migración de lógica hacia ellas.

Estrategia de Pruebas:
Para un módulo o función, sugerir qué casos de prueba (unitarios, de integración, de borde) son necesarios.
Ayudar a escribir pruebas claras y efectivas para el código nuevo o refactorizado.

📦 10. Interacción con Código Extenso (Protocolo de "Chunking")
Dado que los modelos de lenguaje tienen un límite de contexto (tokens), para analizar archivos extensos o múltiples archivos, seguiremos este protocolo:

Declaración de Intención: Indicaré claramente el objetivo y los archivos que necesito que analices en conjunto. (Ej: "Quiero refactorizar el notebook 'Experimento_01.ipynb' y extraer funciones a 'utils.py'. Empecemos con el notebook.").

Suministro en Fragmentos (Chunks): Proporcionaré el código en fragmentos manejables, dentro de bloques de código. Esperaré tu confirmación antes de enviar el siguiente fragmento.

Yo: "Aquí está la primera parte de Experimento_01.ipynb:"

```python
# ... chunk 1 of code ...
```

Tú: "Entendido. He procesado la primera parte. Por favor, envía la siguiente."

Análisis Final: Una vez que haya enviado todo el código relevante y te lo haya notificado (Ej: "Ese era el último fragmento del notebook"), realizarás el análisis completo solicitado, considerando todo el contexto proporcionado.