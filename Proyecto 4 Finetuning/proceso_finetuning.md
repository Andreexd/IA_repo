## Preparación y Preprocesamiento del Dataset

### Descripción del Dataset

Para el entrenamiento del Tutor Inteligente de Algoritmos se diseñó un dataset educativo enfocado en la enseñanza de algoritmos y estructuras de datos. El contenido fue elaborado manualmente en formato Markdown y consiste en diálogos pedagógicos simulados entre un estudiante y un tutor experto.

Cada ejemplo del dataset sigue una estructura clara de **pregunta–respuesta**, donde el estudiante plantea una duda relacionada con algoritmos y el tutor proporciona una explicación detallada, paso a paso, con un enfoque didáctico. Este formato permite entrenar al modelo bajo un esquema de *instruction-following*, alineado con los objetivos del proyecto.

El archivo original del dataset se almacenó como:

- `algoritmos_contenido.md`

ubicado en el directorio `data/raw`.

---

### Proceso de Limpieza y Extracción de Ejemplos

Para transformar el contenido en un formato adecuado para fine-tuning supervisado, se desarrolló un script de preprocesamiento en Python. El objetivo principal de este proceso fue:

- Extraer automáticamente pares **instruction / output** desde el archivo Markdown.
- Eliminar información irrelevante o redundante.
- Normalizar la estructura de los ejemplos.
- Preparar los datos para su posterior entrenamiento.

El script identifica las preguntas del estudiante mediante la etiqueta `**Estudiante:**` y las respuestas del tutor mediante `**Tutor:**`. A partir de estas marcas, se recorren las líneas del archivo y se construyen ejemplos estructurados.

Cada ejemplo generado sigue el siguiente esquema:

```json
{
  "instruction": "Pregunta del estudiante",
  "input": "",
  "output": "Respuesta del tutor"
}
```

---

### División del Dataset

Los ejemplos fueron divididos aleatoriamente en tres conjuntos:

- **Train**: 12 ejemplos (80%)
- **Validation**: 1 ejemplo (10%)
- **Test**: 1 ejemplo (10%)

Los archivos generados fueron guardados en formato JSONL en el directorio `data/processed`:

- `train.jsonl`
- `val.jsonl`
- `test.jsonl`

**Nota**: El tamaño reducido del dataset (14 ejemplos en total) se debe a la naturaleza experimental del proyecto. En un escenario de producción se recomienda utilizar al menos 100-1000 ejemplos para obtener resultados más robustos.

---

## Entrenamiento del Modelo de Lenguaje

### Selección del Modelo Base

Para el desarrollo del Tutor Inteligente de Algoritmos se seleccionó como modelo base **Phi-3 Mini Instruct (4k)** de Microsoft (`microsoft/Phi-3-mini-4k-instruct`). Este modelo fue elegido debido a su buen equilibrio entre tamaño, capacidad de razonamiento y eficiencia computacional, lo que lo hace adecuado para tareas educativas y de fine-tuning en entornos con recursos limitados.

El modelo base ya se encuentra instruido para seguir indicaciones (*instruction-tuned*), lo cual facilita su adaptación a un dominio específico como la enseñanza de algoritmos.

---

### Formato de Entrenamiento y Construcción del Prompt

Los datos preprocesados en formato JSONL fueron cargados utilizando la librería `datasets`. Cada ejemplo se transformó en un prompt estructurado que sigue un formato claro de instrucción y respuesta, como se muestra a continuación:

```
### Instrucción:
{pregunta del estudiante}

### Respuesta:
{respuesta del tutor}</s>
```

Este formato permite al modelo distinguir claramente entre la consulta del usuario (instrucción) y la respuesta esperada, facilitando el aprendizaje supervisado.

---

### Configuración de LoRA (Low-Rank Adaptation)

Para realizar el fine-tuning de manera eficiente se utilizó la técnica **LoRA**, que permite ajustar el modelo con un número reducido de parámetros entrenables, manteniendo congelados los pesos del modelo base.

Parámetros de configuración:

- **r (rank)**: 8
- **lora_alpha**: 16
- **lora_dropout**: 0.05
- **bias**: "none"
- **task_type**: "CAUSAL_LM"
- **target_modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

Esta configuración permite entrenar aproximadamente el 2-3% del total de parámetros del modelo, reduciendo significativamente los requerimientos de memoria y tiempo de entrenamiento.

---

### Resultados del Entrenamiento

El modelo se entrenó durante 3 épocas completas. Los pesos finales fueron guardados en el directorio `models/lora`, incluyendo:

- `adapter_model.safetensors`: Pesos LoRA entrenados
- `adapter_config.json`: Configuración de LoRA
- Archivos del tokenizador

Los adaptadores LoRA tienen un tamaño aproximado de pocos MB, a diferencia del modelo base completo que pesa varios GB.

---

## Inferencia y Uso del Modelo Entrenado

### Carga del Modelo Ajustado

Una vez finalizado el proceso de entrenamiento, el modelo fue utilizado en modo inferencia para evaluar su comportamiento como tutor de algoritmos. Para ello, se cargó el modelo base **Phi-3 Mini Instruct (4k)** junto con los pesos ajustados mediante LoRA.

El proceso de carga consiste en:
- Inicializar el tokenizador correspondiente al modelo base.
- Cargar el modelo preentrenado.
- Integrar los pesos LoRA entrenados.
- Configurar el modelo en modo evaluación (`eval`).

La inferencia se ejecutó en CPU, utilizando precisión FP32 para garantizar estabilidad durante la generación de respuestas.

---
