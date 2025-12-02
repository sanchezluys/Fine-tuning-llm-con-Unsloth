# Fine-tuning-llm-con-Unsloth
Paso a paso como hacer fine tuning de un modelo llm con unsloth

# Pasos

## Instalación de librerias

Primero, instala un instalador de paquetes rápido llamado uv, luego verifica si se está ejecutando en un entorno de Google Colab o si torch no está presente. Basado en estas verificaciones, instala torch, triton, bitsandbytes, transformers, y las librerías unsloth y unsloth_zoo, que son esenciales para operaciones eficientes de modelos de lenguaje grandes (LLM) aceleradas por GPU.

- uv
- torch
- triton
- bitsandbytes
- transformers
- unsloth
- unsloth_zoo

## Unsloth

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

## LoRA adding

Este bloque de código es fundamental para la técnica de finetuning eficiente conocida como LoRA (Low-Rank Adaptation). En lugar de entrenar todas las capas del modelo completo (que podría ser muy costoso en tiempo y recursos de memoria), LoRA añade unas pequeñas "adaptaciones" o "capas" nuevas al modelo existente. Solo estas capas pequeñas se entrenan, lo que reduce drásticamente los recursos necesarios.

Se configura:

- r: el rango de las matrices de adaptación, que controla la capacidad del modelo para aprender nuevas representaciones.
- lora_alpha: un factor de escalado que ayuda a estabilizar el entrenamiento.
- lora_dropout: la tasa de abandono para prevenir el sobreajuste durante el entrenamiento.
- target_modules: las capas específicas del modelo original donde se aplicarán las adaptaciones LoRA.

## Reasoning Effort

Este bloque de texto explica el concepto de 'Reasoning Effort' (Esfuerzo de Razonamiento) en los modelos gpt-oss de OpenAI. Esta característica te permite controlar el nivel de 'pensamiento' que el modelo dedica a una tarea, equilibrando la calidad de la respuesta con la velocidad. Básicamente, puedes ajustar cuántos tokens utiliza el modelo para 'pensar' antes de generar su respuesta final.