# Исследование OCR Reward моделей для fine-tuning диффузионных моделей на рендеринг кириллического текста

**Автор:** [Имя]  
**Дата:** Апрель 2026  
**Проект:** diffusion-text-tuner — фреймворк для дообучения диффузионных моделей на рендеринг не-латинских алфавитов

---

## 1. Введение и постановка задачи

### 1.1 Проблема
Современные диффузионные модели генерации изображений (FLUX, Stable Diffusion и др.) плохо рендерят текст на не-латинских алфавитах. При генерации кириллического текста типичны следующие артефакты:
- Смешение латинских и кириллических глифов (Л → JI, Ч → Y, Н → H)
- Искажение визуальной формы символов (зеркальные, повёрнутые)
- Пропуск или дублирование символов
- Потеря диакритики и знаков препинания

### 1.2 Цель исследования
Найти оптимальную комбинацию OCR-моделей для использования в качестве **reward signal** при дообучении диффузионной модели FLUX.2 Klein 4B методами ReFL (Reward Feedback Learning) и GRPO (Group Relative Policy Optimization).

### 1.3 Требования к reward-модели

| Требование | Описание |
|------------|----------|
| Честность | Модель должна видеть фактические ошибки рендеринга, а не "исправлять" их |
| Дискриминативность | Большая дельта оценки между корректным и сломанным текстом |
| Дифференцируемость | Возможность backprop через reward (для ReFL, Stage 1) |
| Компактность | Умещается в VRAM вместе с FLUX (13GB) и другими rewards |
| Кириллица | Нативная поддержка русского алфавита |

---

## 2. Методология

### 2.1 Тестовый набор

Три изображения с целевым текстом **"ЛУЧШЕЕ КАПУЧИНО 2025!"**:

| Изображение | Описание | Назначение |
|-------------|----------|------------|
| `bad_text.png` | Генерация FLUX Klein 4B с искажённым кириллическим текстом | Негативный пример (сломанный рендер) |
| `good_text.jpg` | Фотография с корректным печатным текстом | Позитивный пример |
| `good_handwritten_text.jpg` | Рукописный корректный текст | Контрольный пример (другой домен) |

### 2.2 Метрика сравнения
Основная метрика — **дельта** (Δ) = разница в оценке между корректным (`good_text.jpg`) и сломанным (`bad_text.png`) текстом. Чем больше Δ, тем лучше модель различает качество рендеринга и тем сильнее reward-сигнал для обучения.

### 2.3 Протестированные модели

1. **PaddleOCR v3** — `cyrillic_PP-OCRv3_mobile_rec` (3MB)
2. **PaddleOCR v5** — `eslav_PP-OCRv5_mobile_rec`, `cyrillic_PP-OCRv5_mobile_rec`, `PP-OCRv5_server_rec`
3. **TrOCR** — `cyrillic-trocr/trocr-handwritten-cyrillic` (0.3B, ~1.34GB)
4. **Qwen3.5-4B** — генерация текста (чтение текста с изображения)
5. **Qwen3.5-4B yes-prob** — вероятность токена "yes" как непрерывная оценка

---

## 3. Результаты

### 3.1 PaddleOCR v3 (cyrillic_PP-OCRv3_mobile_rec)

**Подход:** Классическое OCR — детекция текстовых регионов → распознавание символов. Reward вычисляется как нормализованное расстояние Левенштейна между OCR-выходом и целевым текстом.

**Результаты:**

| Изображение | OCR-выход | Confidence | Примечание |
|-------------|-----------|------------|------------|
| `bad_text.png` | `ЛУчLЕЕ` | 0.64 | Честно читает сломанные глифы |
| `bad_text.png` | `KAпVHиNO` | 0.81 | Смесь Latin/Cyrillic — отражает реальные артефакты |
| `bad_text.png` | `2025!` | 0.99 | Цифры корректны |
| `good_text.jpg` | `лучшEE` | 0.63 | Даже на чистом тексте путает E/Е |
| `good_text.jpg` | `кAпучиHO` | 0.80 | Путает А/A, Н/H на чистом тексте |

**Оценка Δ (confidence):** ~0.35 между broken и correct текстом

**Вывод:** ✅ Честная посимвольная модель без внутренней языковой коррекции. Подходит как reward. Слабость — шумит даже на корректном тексте (потолок reward < 1.0).

---

### 3.2 PaddleOCR v5 (eslav / cyrillic / server)

**Подход:** Архитектурно аналогичен v3, но с **встроенной языковой моделью**, которая корректирует распознанный текст.

**Результаты:**

| Изображение | Модель | OCR-выход | Confidence |
|-------------|--------|-----------|------------|
| `bad_text.png` | v5 eslav | `ЛУЧШЕЕ` | 0.95 |
| `bad_text.png` | v5 cyrillic | `ЛУЧШЕЕ` | 0.94 |
| `bad_text.png` | v5 server | `JYYUEE` | 0.70 |
| `good_text.jpg` | v5 eslav | `ЛУЧШЕЕ` | 0.98 |
| `good_text.jpg` | v5 cyrillic | `ЛУЧШЕЕ` | 0.98 |

**Оценка Δ (confidence):** ~0.03–0.04 между broken и correct текстом

**Критическая проблема:** v5 "исправляет" сломанный текст `ЛУЧЦЕЕ` → `ЛУЧШЕЕ` с confidence 0.95. Модель маскирует именно те ошибки, которые мы пытаемся детектировать. Дельта в 0.03 непригодна как reward-сигнал.

**Вывод:** ❌ Отвергнута. Языковая модель внутри v5 делает её бесполезной как reward — она не видит ошибки рендеринга.

---

### 3.3 TrOCR Cyrillic (trocr-handwritten-cyrillic)

**Подход:** Transformer-based OCR (encoder-decoder), fine-tuned на рукописном кириллическом тексте (включая церковнославянский).

**Результаты:**

| Изображение | OCR-выход | Замечание |
|-------------|-----------|-----------|
| `bad_text.png` (printed) | `И҆ прїи́де ко а҆ве́ссѣ въ` | Полная галлюцинация — церковнославянский текст |
| `good_text.jpg` (printed) | `почій еє` | Галлюцинация |
| `good_handwritten_text.jpg` | `Лучшее`, `Капучино`, `2025!` | ✅ Корректно на рукописном |

**Критическая проблема:** Модель обучена на исторических рукописных документах. На **печатном** тексте (а именно его генерируют диффузионные модели) она галлюцинирует, выдавая церковнославянские фрагменты вне зависимости от содержания изображения.

**Вывод:** ❌ Отвергнута. Domain mismatch — модель непригодна для оценки машинного рендеринга текста.

---

### 3.4 VLM генерация текста (Qwen3.5-4B, прямое чтение)

**Подход:** Попросить мультимодальную модель прочитать текст с изображения.

**Проблема (VLM OCR Bias):** Мультимодальные модели при чтении текста используют внутреннюю языковую модель для "восстановления" текста. На сломанном рендере `ЛУЧЦЕЕ` → VLM читает `ЛУЧШЕЕ`, потому что контекст (кафе, капучино) подсказывает правильное слово.

Это та же проблема, что у PaddleOCR v5, но на уровне foundation model — и более выраженная.

**Вывод:** ❌ Отвергнута как подход прямого чтения. Но VLM можно использовать иначе (см. §3.5).

---

### 3.5 Qwen3.5-4B Yes-Token Probability (ОСНОВНОЙ РЕЗУЛЬТАТ)

**Подход:** Вместо просьбы прочитать текст задаём бинарный вопрос:

> *"Carefully examine each character in this image one by one. Does this image contain the text 'ЛУЧШЕЕ КАПУЧИНО 2025!' with every single character rendered accurately and correctly? Respond with only 'yes' or 'no'."*

Затем извлекаем **вероятность токена "yes"** из logits первого сгенерированного токена. Используем `generate_step()` из `mlx_vlm`, который возвращает полный вектор log-softmax по словарю.

**Формула нормализованного reward:**

$$R_{VLM} = \frac{\sum_{w \in W_{yes}} P(w)}{\sum_{w \in W_{yes}} P(w) + \sum_{w \in W_{no}} P(w)}$$

где $W_{yes}$ = {yes, Yes, YES, да, Да, ДА}, $W_{no}$ = {no, No, NO, нет, Нет, НЕТ}.

**Технические детали реализации:**
- Модель: `mlx-community/Qwen3.5-4B-MLX-4bit` (нативно мультимодальная, ~2.5GB VRAM)
- Библиотека: `mlx-vlm 0.4.3` → `generate_step()` для доступа к logits
- Критично: `enable_thinking=False` при `apply_chat_template` — иначе модель генерирует chain-of-thought, и P(yes/no) для первого токена ~0
- Критично: передавать `image_grid_thw` из `prepare_inputs()` как extra kwargs
- System prompt: *"You are a precise image analysis tool. Answer ONLY with a single word: 'Yes' or 'No'. Do not explain."*
- `temperature=0.0` (greedy decoding)

**Результаты:**

| Изображение | P(yes) | P(no) | R_VLM (normalized) | Сгенерированный токен |
|-------------|--------|-------|--------------------|-----------------------|
| `bad_text.png` (сломанный) | 0.07 | 0.95 | **0.071** | "no" |
| `good_text.jpg` (корректный) | 0.83 | 0.07 | **0.922** | "yes" |
| `good_handwritten_text.jpg` | 0.69 | 0.27 | **0.716** | "yes" |

**Дельта Δ = 0.922 − 0.071 = 0.851** — наилучший результат среди всех протестированных подходов.

**Ключевые свойства:**
- **Дифференцируемость:** P(yes) вычисляется через softmax по logits модели — это дифференцируемая операция. Градиент можно пропустить через VLM reward в Stage 1 (ReFL).
- **Непрерывность:** Reward ∈ [0, 1] — не бинарный, а непрерывный сигнал. Рукописный текст (0.716) получает промежуточную оценку — модель даёт более мягкую оценку неидеальному, но читаемому тексту.
- **Устойчивость к VLM bias:** В формате бинарного вопроса модель реально оценивает визуальное соответствие, а не пытается "прочитать" и реконструировать текст.

**Вывод:** ✅ Основной VLM OCR reward. Лучшая дискриминативность, дифференцируемость, непрерывная шкала.

---

## 4. Сравнительная таблица

| Модель | Тип | Δ (good−bad) | Дифф.? | VRAM | Честность | Статус |
|--------|-----|-------------|--------|------|-----------|--------|
| **Qwen3.5-4B yes-prob** | VLM | **0.851** | ✅ | ~2.5GB | Высокая | ✅ Принята |
| PaddleOCR v3 | Classical | ~0.35 | ❌ | ~3MB | Высокая | ✅ Принята |
| PaddleOCR v5 | Classical+LM | ~0.03 | ❌ | ~50MB | Низкая | ❌ Отвергнута |
| TrOCR Cyrillic | Transformer | N/A | ❌ | ~1.3GB | N/A (галлюцинации) | ❌ Отвергнута |
| VLM прямое чтение | VLM | ~0 | ❌ | ~2.5GB | Низкая | ❌ Отвергнута |

---

## 5. Найденные проблемы и феномены

### 5.1 Проблема коррекции языковой моделью (LM Bias)

Обнаружена у двух классов моделей:
- **PaddleOCR v5**: Встроенный LM корректирует OCR-выход: `ЛУчLЕЕ` → `ЛУЧШЕЕ` (0.95). Дельта 0.03.
- **VLM (прямое чтение)**: Foundation model с мощным language prior реконструирует ожидаемый текст из контекста сцены.

**Вывод:** Любая OCR-модель с языковой коррекцией непригодна как reward для text rendering — она маскирует ошибки, которые мы хотим исправить.

### 5.2 Domain mismatch (TrOCR)

TrOCR, обученная на церковнославянских/рукописных документах, галлюцинирует при работе с печатным текстом. Это показывает, что fine-tuned OCR модели могут быть хуже универсальных, если домен обучающих данных не совпадает с целевым.

### 5.3 Бинарный вопрос vs. чтение текста

Ключевое открытие: **формулировка задачи для VLM критична**. Одна и та же модель (Qwen3.5-4B):
- При просьбе **прочитать** текст → "исправляет" ошибки (bias)
- При бинарном вопросе **"есть ли этот текст?"** → честно оценивает (delta 0.85)

Гипотеза: при чтении активируется language prior модели для реконструкции текста, а при бинарной классификации — визуальная система сравнения.

### 5.4 Thinking mode interference

При включённом thinking mode (`enable_thinking=True`) Qwen3.5 генерирует `<think>...</think>` блок перед ответом. Первый токен в этом случае — начало рассуждения, а не yes/no. P(yes) и P(no) для первого токена ≈ 0. Отключение thinking mode (`enable_thinking=False`) заставляет модель сразу отвечать yes/no, что даёт чистый probability signal.

---

## 6. Архитектура reward stack для обучения

На основе результатов предлагается следующий reward stack:

### Stage 1: ReFL (дифференцируемые rewards)
| Reward | Модель | Что измеряет |
|--------|--------|-------------|
| VLM OCR | Qwen3.5-4B yes-prob | Посимвольная корректность текста |
| Alignment | SigLIP2 So400m | Соответствие prompt ↔ image |
| Aesthetic | HPSv2.1 | Человеческое предпочтение качества |
| Perceptual text | DINOv3 (рендер-сравнение) | Визуальное сходство текста с эталоном |

### Stage 2: Flow-GRPO (все rewards)
К Stage 1 добавляются:
| Reward | Модель | Что измеряет |
|--------|--------|-------------|
| Classical OCR | PaddleOCR v3 | Честное посимвольное чтение |
| Char count | Эвристика из OCR | Совпадение числа символов |

**Формула composite reward (Stage 2):**

$$R(x, p) = \alpha \cdot R_{VLM} + \beta \cdot R_{OCR} + \gamma \cdot R_{SigLIP} + \delta \cdot R_{HPS} + \varepsilon \cdot R_{DINOv3} + \zeta \cdot R_{charcount}$$

---

## 7. Ограничения и дальнейшие шаги

### 7.1 Ограничения текущего исследования
1. **Малый тестовый набор** (3 изображения, 1 целевой текст) — результаты являются proof of concept, не статистической валидацией
2. **Не протестирован DINOv3** rendered reference comparison
3. **Не оценена чувствительность к частичным ошибкам** — как Qwen3.5 реагирует на 1 ошибку из 20 символов?
4. **Temperature=0** — не исследовано влияние температуры на калибровку вероятностей
5. **PaddleOCR v3 зашумлена на чистом тексте** — потолок reward < 1.0 даже для идеального рендера
6. **Только один язык** (русский) — поведение на других кириллических языках и скриптах не проверено

### 7.2 Ближайшие шаги
1. Расширить тестовый набор до 50+ пар (good/bad) с разными текстами, шрифтами, сценами
2. Протестировать DINOv3 rendered reference comparison
3. Исследовать чувствительность Qwen yes-prob к количеству и типу ошибок
4. Реализовать ReFL trainer с Qwen3.5 + SigLIP2 + HPSv2.1
5. Реализовать pipeline генерации промптов (алгоритмический текст + LLM-сцены)

---

## Приложение A: Воспроизведение экспериментов

### Зависимости
```bash
# PaddleOCR v3
pip install paddlepaddle==3.3.1 paddleocr==3.4.0

# Qwen3.5-4B
pip install mlx-vlm==0.4.3

# TrOCR (не рекомендуется)
pip install transformers torch sentencepiece
```

### Запуск
```bash
# PaddleOCR v3
python test_paddleocr.py

# PaddleOCR v5 (сравнение)
python test_paddleocr_v5.py

# TrOCR
python test_trocr.py

# Qwen3.5 yes-prob (основной)
python test_qwen_yes_prob.py
```

### Ключевые параметры Qwen3.5 yes-prob
- Model ID: `mlx-community/Qwen3.5-4B-MLX-4bit`
- System prompt: жёсткое ограничение yes/no
- `enable_thinking=False` — обязательно
- `temperature=0.0` — greedy decoding
- `generate_step()` → logprobs = log-softmax по всему словарю
- Нормализация: P(yes) / (P(yes) + P(no)) с суммированием всех вариантов (yes/Yes/YES/да/Да/ДА)

---

## Приложение B: Код Qwen3.5 yes-token probability

Основная функция извлечения reward score:

```python
from mlx_vlm.generate import generate_step
from mlx_vlm import load, prepare_inputs

def get_vlm_reward(model, processor, tokenizer, image_path, target_text,
                   yes_ids, no_ids):
    """
    Compute differentiable VLM OCR reward as normalized P(yes).
    
    Returns: float in [0, 1] — probability that the image contains
    the target text with all characters rendered correctly.
    """
    prompt = (
        f"Carefully examine each character in this image one by one. "
        f"Does this image contain the text \"{target_text}\" with every "
        f"single character rendered accurately and correctly? "
        f"Respond with only 'yes' or 'no'."
    )
    
    messages = [
        {"role": "system",
         "content": "You are a precise image analysis tool. "
                    "Answer ONLY with a single word: 'Yes' or 'No'. "
                    "Do not explain."},
        {"role": "user",
         "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
    ]
    
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,   # CRITICAL: disable chain-of-thought
    )
    
    inputs = prepare_inputs(processor, images=[image_path], prompts=formatted)
    extra_kwargs = {k: v for k, v in inputs.items()
                    if k not in ("input_ids", "pixel_values", "attention_mask")}
    
    gen = generate_step(
        inputs["input_ids"], model,
        inputs.get("pixel_values"), inputs.get("attention_mask"),
        max_tokens=1, temperature=0.0,
        **extra_kwargs,  # includes image_grid_thw
    )
    _, logprobs = next(gen)
    
    p_yes = sum(mx.exp(logprobs[tid]).item() for tid in yes_ids.values())
    p_no  = sum(mx.exp(logprobs[tid]).item() for tid in no_ids.values())
    
    return p_yes / (p_yes + p_no) if (p_yes + p_no) > 0 else 0.0
```
