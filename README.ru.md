# mi50-qwen3.8-fast

**Qwen3.8-27B на двух AMD Instinct MI50 (gfx906) — ~48 ток/с вместо 19.4 на стоковом llama.cpp.**
Та же модель, тот же файл Q4_0, вывод побитово тот же. Пропатченный llama.cpp в Docker, установка
одной командой.

[English version](README.md)

| Конфигурация (2× MI50 16 ГБ, по 85 Вт) | Генерация, запрос агента | Примечание |
|---|---|---|
| llama.cpp b10621, ROCm 6.3, `-sm layer`, без MTP | 19.4 ток/с | типичная отправная точка |
| b11235, `-sm tensor` + MTP, ROCm 6.3 | 39–45 | зависает примерно раз на 8 длинных запросов |
| b11235, `-sm tensor` + MTP, **ROCm 7.14** | 40–44 | 0 зависаний на 30+ длинных запросах |
| **+ заплаты кернелов gfx906 (этот репозиторий)** | **48** | вывод побитово тот же, KLD в пределах шума |
| vLLM (форк gfx906), GPTQ-Int4, TP=2 + MTP | 34.7 | промпт тоже медленнее, контекст до 64K |

Подробности и как мерилось каждое число — [docs/ru/results.md](docs/ru/results.md).

## Быстрый старт

Нужно: Linux со встроенным в ядро драйвером `amdgpu` (любой современный дистрибутив), Docker,
~30 ГБ на диске, одна или несколько MI50/MI60/Radeon VII.

```bash
git clone https://github.com/JeanP00l/mi50-qwen3.8-fast && cd mi50-qwen3.8-fast
./install.sh        # проверка системы, сборка образа (20–40 мин один раз), загрузка ~17.4 ГБ моделей
./run.sh            # сервер на http://127.0.0.1:8080 (API OpenAI и Anthropic)
```

Проверить (в другом терминале): `./bench.sh`

`run.sh` сам выбирает режим:

| Карты | Что происходит |
|---|---|
| 2 и больше | две карты, `-sm tensor` + спекуляция MTP — измеренная конфигурация |
| 1 × 32 ГБ | одна карта + MTP (должно работать; автором не измерено) |
| 1 × 16 ГБ | отказ: файл 27B Q4_0 весит 16.1 ГБ. Варианты — [docs/ru/single-gpu.md](docs/ru/single-gpu.md) |

Полезное: `API_KEY=secret HOST=0.0.0.0 PORT=8080 CTX=65536 ./run.sh`; `./run.sh --dry-run`
печатает точную команду `docker run`. Все настройки — в начале `run.sh`.

С Claude Code (сервер запущен как `API_KEY=secret ./run.sh`):
```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_AUTH_TOKEN=secret \
       ANTHROPIC_MODEL=qwen3.8 ANTHROPIC_DEFAULT_OPUS_MODEL=qwen3.8 \
       ANTHROPIC_DEFAULT_SONNET_MODEL=qwen3.8 ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen3.8 \
       CLAUDE_CODE_SUBAGENT_MODEL=qwen3.8 CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
       NO_PROXY=127.0.0.1,localhost
claude
```
Один агент на сервер: при `--parallel 1` один разговор переиспользует кэш промпта (40–140 новых
токенов за ход, меньше секунды); два агента вытесняют друг друга, и каждый ход перечитывает 20–60K.

## Что внутри

| | |
|---|---|
| `docker/Dockerfile` | llama.cpp тег `b11235` на ROCm 7.14 с поддержкой gfx906 ([mixa3607/rocm-gfx906](https://hub.docker.com/r/mixa3607/rocm-gfx906)) + 3 заплаты |
| `patches/gfx906-mmvq-q4_0-multicol.py` | умножение Q4_0 на 2–8 столбцах: проверка черновиков MTP (1 + 3 токена) 51 → 43 мс |
| `patches/gfx906-gdn-multicolumn.py` | кернел Gated DeltaNet (гибриды Qwen3.5/3.6/3.8): одна волна на 4 столбца |
| `patches/gfx906-mmvf-few-rows.py` | F32-умножение для матриц с малым числом строк (`ssm_alpha/beta`): 48 блоков на 60 CU, карта почти пустая |
| `templates/qwen3.8-agent.jinja` | шаблон чата, работающий с Claude Code (системные сообщения посреди разговора, разумный effort) |
| `scripts/check-system.sh` | проверка только чтением: драйвер, Docker, карты, **крупный BAR**, **P2P между картами** |

## Что дало больше всего

1. **ROCm 7.14 вместо 6.3** для `-sm tensor`: на 6.3 взаимная блокировка в all-reduce, при сбросе карты машина умирала.
2. **BIOS: Above 4G Decoding + окно MMIO выше 4G ниже 16 ТиБ** — у Vega 20 DMA 44-битная; при апертуре
   на 56 ТиБ нет P2P, и `-sm tensor` вдвое медленнее. [docs/ru/hardware.md](docs/ru/hardware.md)
3. **`-sm tensor` + MTP (черновик 3)**: dense 27B упирается в память, разрез tensor читает веса из обеих HBM.
4. **`--cache-reuse 0 --cache-ram 0 --ctx-checkpoints 0`**: без них gfx906 ловит
   `Memory access fault` под нагрузкой. [docs/ru/troubleshooting.md](docs/ru/troubleshooting.md)
5. **Q4_0, а не Q4_K_M**: у K-квантов нет быстрого кернела под gfx906.
6. Заплаты отсюда: +13 % сверху, вывод тот же. [docs/ru/how-it-works.md](docs/ru/how-it-works.md)

## На чём проверено

2× MI50 16 ГБ, Xeon E5-2666 v3 (X99), Ubuntu 24.04, ядро 6.8, Docker 29, лимит 85 Вт на карту.
Другие дистрибутивы должны работать — с хоста берутся только драйвер ядра и Docker. Отчёты приветствуются.

## Лицензия

MIT. llama.cpp — MIT; шаблон чата — производный от шаблона Qwen (Apache-2.0) с исправлениями Unsloth.
