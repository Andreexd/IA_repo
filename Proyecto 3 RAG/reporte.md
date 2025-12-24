
## CONSTRUCCIÓN DEL CORPUS

Se construyó un corpus compuesto por documentos académicos y filosóficos en formato PDF, correspondientes a obras y textos de los autores mencionados. La selección de los documentos se realizó considerando su relevancia para analizar identidad, sentido, tecnología y autonomía.

Además, datos fueron extraídos de diversas fuentes en internet para la recolección de opiniones de relevancia respecto al tema.

---

## USO DEL SISTEMA RAG

![Configuración de AnythingLLM](imagenes/ConfigAnythingllm.png)

### ¿Qué es RAG?

**RAG (Retrieval Augmented Generation)** es una técnica que combina búsqueda de información con generación de texto. El sistema busca documentos relevantes en una base de datos y luego genera respuestas basadas únicamente en esa información recuperada, evitando alucinaciones del modelo.

### Implementación

Para la práctica se utilizó la herramienta AnythingLLM, configurada con un modelo de lenguaje de la familia Qwen. El sistema implementa un enfoque RAG, el cual funciona de la siguiente manera:

1. Los documentos son cargados y fragmentados en unidades de texto.
2. Cada fragmento se transforma en una representación vectorial (embedding) que captura su significado semántico.
3. Estos vectores se almacenan en una base de datos vectorial.
4. Cuando el usuario realiza una pregunta, esta también se transforma en un vector.
5. El sistema recupera los fragmentos más cercanos semánticamente a la pregunta.
6. El modelo generativo produce una respuesta utilizando únicamente la información recuperada, permitiendo además la inclusión de citas.

### Ejemplos de Uso

![Ejemplo de pregunta 1](imagenes/pregunta.png)

![Ejemplo de pregunta 2](imagenes/pregunta1.png)

---

## 1. LIMPIEZA DE DATOS (1_limpiar_datos.py)

### Objetivo
Preprocesar el dataset sintético eliminando inconsistencias y duplicados para obtener datos de calidad.

### Proceso
1. **Carga de datos:** Lectura del archivo CSV original (5,000 registros)
2. **Validación:** Eliminación de registros con campos nulos en columnas críticas (texto, tema)
3. **Eliminación de duplicados:** Identificación y remoción de textos repetidos usando `drop_duplicates()`
4. **Análisis estadístico:** Generación de distribución por tema y sentimiento
5. **Exportación:** Guardado del dataset limpio y reporte JSON con métricas
---

## 2. CONVERSIÓN A FORMATO JSONL (2_convertir_a_jsonl.py)

### Objetivo
Transformar el dataset limpio a formato JSONL optimizado para sistemas de Retrieval Augmented Generation (RAG).

### Proceso
1. **Carga del dataset limpio:** Lectura de los registros validados
2. **Estructuración:** Creación de documentos JSON con:
   - Campo `content` para el texto principal
   - Objeto `metadata` con atributos contextuales (tema, sentimiento, fecha, métricas)
   - Campos adicionales para indexación (`title`, `date`)
3. **Serialización JSONL:** Un documento JSON por línea para procesamiento eficiente
4. **Generación alternativa:** Archivo JSON estándar como respaldo

### Resultados
- **Formato de salida:** JSONL (optimizado para LLMs)
- **Estructura:** JSON anidado con metadata completa
- **Archivos generados:** `dataset_para_llm.jsonl`, `dataset_para_llm.json`

---

