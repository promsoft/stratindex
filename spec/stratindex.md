# Задача

Есть метрика - индекс стратификации см например тут https://www150.statcan.gc.ca/n1/en/catalogue/12-001-X197600200002   есть
R- библиотка для расчета его https://cran.r-project.org/web/packages/strat/

Давай портируем библиотеку на python, целевая версия - python 3.12 и опубликуем пакет в pypi

# Уточняющие вопросы и ответы (зафиксированы 2026-07-12)

**В1. Какую метрику портируем?** R-пакет `strat` реализует индекс **Zhou (2012)** «A Nonparametric Index of Stratification» (Sociological Methodology 42(1): 365–389), а статья Statcan 1976 (Gray, «Stratification index: Methodology and analysis») описывает другую метрику — индекс эффективности стратификации выборки.
**Ответ:** портируем R-пакет `strat` (Zhou 2012); ссылку на Statcan в исходной постановке считаем неточностью описания.

**В2. Какая лицензия?** Исходный R-пакет — GPL ≥3, в репозитории лежал Apache 2.0; прямой порт кода — производная работа.
**Ответ:** GPL-3.0-or-later (LICENSE заменён).

**В3. Как реализовать O(n²)-ядро (в оригинале C++/Rcpp)?** Варианты: чистый NumPy с блочной векторизацией / Cython/C-расширение / Numba.
**Ответ:** NumPy с блочной векторизацией — без компиляции, простая сборка wheel, единственная runtime-зависимость numpy. На 14k строк — секунды.

**В4. Как публиковать на PyPI?** Варианты: GitHub Actions + Trusted Publishing / вручную twine.
**Ответ:** GitHub Actions + PyPI Trusted Publishing, релиз по тегу `v*` (нужна разовая настройка trusted publisher в аккаунте PyPI).

**Решения без отдельного вопроса** (по умолчанию, согласованы в плане):
- Имя пакета на PyPI: **`stratindex`** — свободно; `strat` и `pystrat` заняты.
- Целевая версия Python: `requires-python >=3.12`.
- Датасет `cpsmarch2015` конвертируется в csv.gz и включается в пакет (данные CPS — публичные данные правительства США).
- Валидация: R локально не установлен → ручные примеры + сверка с наивной реализацией + золотые значения из R через CI-job (docker r-base).

# План реализации

## Этап 1. Скелет пакета — ✅ сделано
- [x] `pyproject.toml` (hatchling, метаданные, GPL-3, `requires-python >=3.12`)
- [x] src-layout `src/stratindex/`
- [x] Замена LICENSE Apache 2.0 → GPL-3.0-or-later
- [x] `requirements.in` → `requirements.txt` (uv pip-compile), зависимости установлены
- [x] `.gitignore`: симлинк `venv`, `.python-version`

## Этап 2. Ядро (порт 1:1 по поведению) — ✅ сделано
- [x] `_utils.py`: `clean()` (валидация, complete cases, нормировка весов к сумме n), `wtd_rank()` — порт `Hmisc::wtd.rank(normwt=TRUE)` (midranks по cumsum весов)
- [x] `_kernel.py`: блочно-векторизованные эквиваленты `strat_cpp` / `strat_cpp_by` (пары с `r_j > r_i` и `y_j ≠ y_i`, вклад `sign(y_j−y_i)·w_i·w_j`)
- [x] `results.py`: датаклассы `StratResult` / `SrankResult` с `__str__` как print-методы R
- [x] `core.py`: публичные `srank()` и `strat()` (декомпозиция between/within, SE по Goodman–Kruskal 1963)
- [x] Экспорты в `__init__.py`, smoke-прогон: сверка с наивной O(n²) реализацией сошлась до 1e-12

## Этап 3. Датасет — ✅ сделано
- [x] Конвертация `cpsmarch2015.rda` → `cpsmarch2015.csv.gz` (`scripts/convert_dataset.py`, pyreadr, воспроизводимый gzip)
- [x] Включение в пакет (`src/stratindex/data/`) + `load_cpsmarch2015()` (dict of ndarrays или pandas DataFrame)
- [x] Прогон примера из R-документации: `strat(income, big_class, weights=weight, group=education)` → strat=0.4128, se=0.01296, ~6 с на 14 358 строк

## Этап 4. Тесты — ✅ сделано (26 тестов)
- [x] Ручные маленькие примеры (значения посчитаны по определению): идеальная стратификация = ±1, пример 2×2 со strat=0.5
- [x] Сверка векторизованного ядра с наивной построчной реализацией (перевод C++-цикла) на случайных данных: веса, ties, группы, размеры блока 1/3/64/4096, сходимость 1e-12
- [x] Краевые случаи: NA (drop + ошибки валидации), одна страта → NaN (как 0/0 в R), `ordered=True`, группа с одним уровнем → без декомпозиции; тождества декомпозиции (веса суммируются в 1, overall = взвешенная сумма компонент)
- [x] Прогон на `cpsmarch2015`: regression-snapshot (strat=0.41275, se=0.01296) — до сверки с R в CI
- [x] Найдено и исправлено: точность `deno_between` в блочном ядре (поэлементная разность вместо разности сумм); prank может слегка превышать 1 (свойство Hmisc::wtd.rank, воспроизведено 1:1)

## Этап 5. Документация и CI — ✅ сделано
- [x] README (EN): формула, пример на `cpsmarch2015`, таблица соответствия R-API, поведенческие заметки
- [x] GitHub Actions `ci.yml`: pytest (3.12, 3.13) + ruff check/format
- [x] Золотые значения из R: сгенерированы локально в docker (rocker/r2u, CRAN strat 0.1) → `tests/data/r_golden.json`; `tests/test_r_golden.py` сверяет 6 кейсов + srank (rtol 1e-9, совпадение ~1e-11); CI-job `r-crosscheck` перегенерирует и сверяет fixture с CRAN
- [x] `publish.yml`: тег `v*` → build + smoke-тест wheel + PyPI (trusted publishing, env `pypi`); workflow_dispatch → TestPyPI (env `testpypi`)
- [x] CLAUDE.md обновлён реальными командами и архитектурой

## Этап 6. Релиз 0.1.0
- [ ] Сборка `uv build`, прогон на TestPyPI
- [ ] Настройка trusted publisher в аккаунте PyPI (нужно участие владельца аккаунта)
- [ ] Тег `v0.1.0`, публикация на PyPI

# Справка: устройство оригинального R-пакета

- `clean()`: complete cases по outcome/strata/weights; веса нормируются к сумме n; `prank = wtd_rank(outcome, w)/n`; strata/group → факторы.
- `srank()`: по стратам — доля населения (`share`) и средневзвешенный перцентильный ранг (`s_prank`).
- `strat()`: строки сортируются по `s_prank` страты (или по порядку страт при `ordered=TRUE`); индекс = взвешенная доля согласованных пар минус несогласованных среди пар из разных страт с различающимся outcome (аналог Somers' D); SE — аппроксимация Goodman & Kruskal (1963): `se = sqrt((1−strat²)·n/deno)`. При заданном `group` — декомпозиция на within/between и индексы по группам.
- Датасет `cpsmarch2015`: 14 358 мужчин 35–64 из March CPS 2015 (income, big_class, micro_class, education, weight).
