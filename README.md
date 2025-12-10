# Fine-tuning-llm-con-Unsloth
Paso a paso como hacer fine tuning de un modelo llm con unsloth

# Pasos

## Instalación de librerias - Paso 1

Primero, instala un instalador de paquetes rápido llamado uv, luego verifica si se está ejecutando en un entorno de Google Colab o si torch no está presente. Basado en estas verificaciones, instala torch, triton, bitsandbytes, transformers, y las librerías unsloth y unsloth_zoo, que son esenciales para operaciones eficientes de modelos de lenguaje grandes (LLM) aceleradas por GPU.

- uv
- torch
- triton
- bitsandbytes
- transformers
- unsloth
- unsloth_zoo

## Unsloth - Paso 2

Este bloque de código se encarga de cargar un modelo de lenguaje grande (LLM) pre-entrenado, específicamente el modelo gpt-oss-20b, utilizando la biblioteca Unsloth. Primero, importa FastLanguageModel y torch. Luego, define max_seq_length para la longitud máxima de la secuencia y dtype para el tipo de datos, aunque aquí se deja como None para autodetección. También se muestra una lista de modelos pre-cuantizados en 4 bits que Unsloth soporta, lo cual ayuda a reducir la memoria y acelerar la descarga. Finalmente, el código carga el modelo y su tokenizador asociado desde Hugging Face, aplicando una cuantización de 4 bits (load_in_4bit = True) para optimizar el uso de la memoria, lo que es crucial para modelos grandes.

Por defecto:
- max_seq_length = 1024  // depende de la aplicación y del recurso disponible ya que usa GPU Vrams
- dtype = None  // se autodetecta depende principalmente de tu hardware (GPU) y de un balance entre memoria, velocidad y precisión que desees para tu modelo.

En este paso se debe seleccionar el modelo a usar, en este caso se usa gpt-oss-20b, pero puedes elegir otro de la lista disponible

fourbit_models = [
    "unsloth/gpt-oss-20b-unsloth-bnb-4bit", # 20B model using bitsandbytes 4bit quantization
    "unsloth/gpt-oss-120b-unsloth-bnb-4bit",
    "unsloth/gpt-oss-20b", # 20B model using MXFP4 format
    "unsloth/gpt-oss-120b",
]

> Warning: pendiente saber como usar full_finetuning con unsloth

## LoRA adding - Paso 3

Este bloque de código es fundamental para la técnica de finetuning eficiente conocida como LoRA (Low-Rank Adaptation). En lugar de entrenar todas las capas del modelo completo (que podría ser muy costoso en tiempo y recursos de memoria), LoRA añade unas pequeñas "adaptaciones" o "capas" nuevas al modelo existente. Solo estas capas pequeñas se entrenan, lo que reduce drásticamente los recursos necesarios.

Se configura:

Variables:
1. r: por defecto es 8, y un valor más alto significa que el modelo dedicará más recursos a razonar sobre la tarea, lo que puede mejorar la calidad de las respuestas pero también aumentará el tiempo de respuesta. Sus valores típicos son 4, 8, 16, 32, 64, 128
2. target_modules: son las capas del modelo donde se aplicará el esfuerzo de razonamiento. En este caso, se aplicará a las capas 'q_proj' y 'v_proj' del modelo, que son cruciales para la atención y la generación de texto.
3. Lora_alpha: es un factor de escalado que ayuda a estabilizar el entrenamiento cuando se aplican adaptaciones LoRA. Por defecto es 16.
4. lora_dropout: es la tasa de abandono aplicada a las capas LoRA para prevenir el sobreajuste. Por defecto es 0.1 (10%). se coloca en 0.
5. bias: se establece en "none" para evitar modificar los sesgos del modelo original durante el entrenamiento LoRA.
6. use_gradient_checkpointing: se establece en True para reducir el uso de memoria durante el entrenamiento, lo que es especialmente útil para modelos grandes.
7. random_state: se establece en 42 para asegurar la reproducibilidad de los resultados durante el entrenamiento. esta en 3407
8. use_rslora: se establece en True para utilizar la variante RS-LoRA, que es una versión optimizada de LoRA que puede ofrecer mejores resultados en ciertos escenarios. Esta en False
9. loftq_config: se establece en None, lo que significa que no se está utilizando ninguna configuración específica de LoFT-Q en este caso.

## Reasoning Effort - Paso 4

Este bloque de texto explica el concepto de 'Reasoning Effort' (Esfuerzo de Razonamiento) en los modelos gpt-oss de OpenAI. Esta característica te permite controlar el nivel de 'pensamiento' que el modelo dedica a una tarea, equilibrando la calidad de la respuesta con la velocidad. Básicamente, puedes ajustar cuántos tokens utiliza el modelo para 'pensar' antes de generar su respuesta final.

Tiene 3 niveles de esfuerzo:
1. Low (Bajo): Utiliza 4 tokens para razonar. Es rápido pero puede sacrificar algo de calidad en la respuesta.
2. Medium (Medio): Utiliza 8 tokens para razonar. Ofrece un equilibrio entre velocidad y calidad.
3. High (Alto): Utiliza 16 tokens para razonar. Proporciona la mejor calidad de respuesta, pero es más lento.

el mensaje actual es:

```python
messages = [
    {"role": "user", "content": "Solve x^5 + 3x^4 - 10 = 3."},
]
```

segun mi caso lo cambio a:
```python
messages = [
    # El mensaje del sistema es clave para establecer las expectativas del formato de salida.
    {"role": "system", "content": "You are a helpful assistant that configures bots. Always respond with the bot's configuration in a valid JSON format."},
    
    # El mensaje del usuario contiene tu 'data' o prompt específico.
    {"role": "user", "content": "I need a configuration for a simple weather forecast bot. It should ask the user for their location and then provide the current temperature and a brief weather description. If the location is not provided, it should ask again. Ensure the output is a JSON object with keys like 'bot_name', 'description', 'dialog_flow', and 'error_handling'."},
]
```
