# 🤖 Instrucciones para el Asistente de IA (Gemini)
## Misión Actual: Análisis y Refactorización del Proyecto

Hola Gemini. Este archivo define el estado actual de mi proyecto y mis objetivos para refactorizarlo y alinearlo con las mejores prácticas. Úsalo como tu guía principal para analizar el código y proporcionar sugerencias de mejora.

---

### **PARTE I: CONTEXTO DEL PROYECTO (ESTADO ACTUAL)**

#### **1. Resumen del Proyecto**

* **Nombre del Proyecto:** [Ej: "API de Gestión de Inventarios v1"]
* **Propósito Principal:** [Ej: "Provee endpoints para controlar el stock de productos, proveedores y movimientos de almacén."]

#### **2. Pila Tecnológica**

* **Lenguaje/Framework:** [Ej: Python con FastAPI]
* **Base de Datos:** [Ej: PostgreSQL con SQLAlchemy]
* **Testing:** [Ej: Pytest]
* **Otros:** [Ej: Docker, Alembic para migraciones]

#### **3. Estructura Actual del Repositorio**

*Aquí describe la estructura actual, incluso si crees que es incorrecta. Es mi punto de partida.*

```
/
├── app/
│   ├── crud/         # Lógica de acceso a datos (CRUD)
│   ├── models/       # Modelos de SQLAlchemy
│   ├── schemas/      # Esquemas Pydantic (DTOs)
│   ├── api/          # Endpoints de la API
│   │   └── v1/
│   │       ├── endpoints/
│   │       │   ├── products.py
│   │       │   └── users.py
│   ├── core/         # Configuración
│   └── main.py       # Arranque de la app
├── tests/            # Pruebas
└── ...
```

#### **4. Estándares de Código Actuales (Si existen)**

* **Estilo de Código:** [Ej: "Se intenta seguir PEP 8, usando el formateador Black."]
* **Convenciones:** [Ej: "No hay una convención estricta, pero se usa snake_case."]

---

### **PARTE II: OBJETIVOS DE LA REFACTORIZACIÓN**

#### **🎯 5. Objetivo Principal de la Mejora**

Mi meta es analizar la estructura y el código de este proyecto para identificar áreas de mejora, reducir la deuda técnica y aplicar patrones de diseño y prácticas recomendadas. Busco que el código sea más **mantenible, escalable y robusto.**

#### **⚠️ 6. Puntos de Dolor y Deuda Técnica Identificada**

*Aquí lista los problemas que ya conoces o sospechas que existen.*
* **Ejemplo 1:** "El archivo `products.py` en los endpoints tiene demasiada lógica de negocio, debería estar en una capa de servicio."
* **Ejemplo 2:** "Hay mucho código duplicado en las funciones CRUD para diferentes modelos."
* **Ejemplo 3:** "El manejo de errores es inconsistente en toda la API."
* **Ejemplo 4:** "Faltan pruebas para los casos de uso más complejos."

#### **🏆 7. Objetivos Específicos de la Refactorización**

*Aquí define cómo se ve el "éxito".*
* **Ejemplo 1:** "Implementar una capa de servicios (`/app/services/`) para abstraer la lógica de negocio de los controladores/endpoints."
* **Ejemplo 2:** "Crear un repositorio genérico o una clase base para reducir la duplicación en el CRUD."
* **Ejemplo 3:** "Centralizar el manejo de excepciones con un middleware de FastAPI."
* **Ejemplo 4:** "Aumentar la cobertura de pruebas al 80%, enfocándose en la capa de servicios."

#### **📐 8. Principios y Métricas de Calidad a Seguir**

*Pídele a la IA que se enfoque en estos principios durante el análisis.*
* **Principios SOLID:** Especialmente el de Responsabilidad Única (SRP) y Inversión de Dependencias (DIP).
* **Principio DRY (Don't Repeat Yourself):** Identificar y eliminar código repetido.
* **Principio KISS (Keep It Simple, Stupid):** Simplificar funciones y algoritmos complejos.
* **Acoplamiento y Cohesión:** Buscar un bajo acoplamiento y una alta cohesión en los módulos.

---

### **PARTE III: CÓMO PUEDES AYUDAR**

#### **🔍 9. Tareas de Análisis y Asistencia Solicitadas**

Cuando te pida ayuda, enfócate en estas tareas:

1.  **Análisis Estructural:**
    * Revisa la estructura de carpetas actual y sugiere una organización más limpia y escalable (ej: arquitectura hexagonal, por capas, etc.).
    * Identifica módulos que están demasiado acoplados.

2.  **Revisión de Código (`Code Review`):**
    * Analiza archivos específicos que te proporcione y busca "code smells" (malos olores en el código).
    * Sugiere cómo aplicar los principios (SOLID, DRY) al código existente.
    * Propón patrones de diseño que podrían mejorar una sección del código (ej: Patrón Repositorio, Inyección de Dependencias, etc.).

3.  **Generación de Código Refactorizado:**
    * Cuando te pida refactorizar una función, provéeme la versión mejorada explicando los cambios realizados.
    * Ayúdame a crear las nuevas capas (como los servicios) y a mover la lógica correspondiente.

4.  **Estrategia de Pruebas:**
    * Sugiéreme qué tipos de pruebas (unitarias, de integración) faltan para un módulo específico.
    * Ayúdame a escribir pruebas para el código ya refactorizado.

Gracias por tu ayuda para mejorar la calidad de este proyecto.