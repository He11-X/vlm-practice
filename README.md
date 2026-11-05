# Обучение визуально-языковой модели на русском языке

## Описание
Проект направлен на обучение и оценку визуально-языковой модели (VLM) на основе открытых данных VK (GQA-ru, MMBench-ru, LLaVA-Instruct-ru). 
Цель — достичь высоких метрик на бенчмарках GQA-ru и MMBench-ru с корректным экспериментальным протоколом.

## Экспериментальный протокол
- **Модель:** deepvk/llava-saiga-8b (базовая) и дообучение с LoRA.
- **Данные:** 
  - Обучение: GQA-ru (train_balanced_instructions) с выделением 10% как валидация.
  - Тест: GQA-ru (testdev_balanced_instructions) и MMBench-ru (dev).
- **Гиперпараметры:** см. config.py.
- **Воспроизводимость:** фиксирован seed = 42, все случайные выборки детерминированы.
- **Оценка:** точность (Accuracy) с нормализацией ответов для GQA, парсинг буквы для MMBench.

## Структура
- `config.py` — все параметры эксперимента.
- `data_utils.py` — загрузка и предобработка с маскировкой лосса.
- `train.py` — обучение с полным fine-tune.
- `train_lora.py` — обучение с LoRA.
- `eval_gqa.py`, `eval_mmbench.py` — оценка на тестовых сплитах.
- `inference.py` — инференс на новых изображениях.
- `experiment_checkpoints/` — сохраняются лучшие чекпоинты и конфигурация.

## Запуск (пример)
```bash
pip install -r requirements.txt
python train.py
python eval_gqa.py
python eval_mmbench.py
```

Результаты (симуляция на 100 случайных примерах)
GQA-ru: 0.5100

MMBench-ru: 0.4600

Ссылка на репозиторий
https://github.com/He11-X/vlm-practice
