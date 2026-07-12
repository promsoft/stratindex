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

## Этап 3. Датасет
- [ ] Конвертация `cpsmarch2015.rda` → `cpsmarch2015.csv.gz` (pyreadr, dev-скрипт)
- [ ] Включение в пакет + `load_cpsmarch2015()`

## Этап 4. Тесты
- [ ] Ручные маленькие примеры (значения посчитаны по определению)
- [ ] Сверка векторизованного ядра с наивной O(n²) реализацией на случайных данных (веса, ties, группы, малые размеры блока)
- [ ] Краевые случаи: NA, одна страта, `ordered=True`, вырожденные группы
- [ ] Прогон на `cpsmarch2015`

## Этап 5. Документация и CI
- [ ] README (EN): формула, пример на `cpsmarch2015`, соответствие R-API
- [ ] GitHub Actions: тесты + ruff на push
- [ ] CI-job с r-base (docker) — золотые значения из оригинального R-пакета, фиксация в fixtures
- [ ] Workflow публикации на тег `v*` (trusted publishing)
- [ ] Обновить CLAUDE.md реальными командами

## Этап 6. Релиз 0.1.0
- [ ] Сборка `uv build`, прогон на TestPyPI
- [ ] Настройка trusted publisher в аккаунте PyPI (нужно участие владельца аккаунта)
- [ ] Тег `v0.1.0`, публикация на PyPI

# Справка: устройство оригинального R-пакета

- `clean()`: complete cases по outcome/strata/weights; веса нормируются к сумме n; `prank = wtd_rank(outcome, w)/n`; strata/group → факторы.
- `srank()`: по стратам — доля населения (`share`) и средневзвешенный перцентильный ранг (`s_prank`).
- `strat()`: строки сортируются по `s_prank` страты (или по порядку страт при `ordered=TRUE`); индекс = взвешенная доля согласованных пар минус несогласованных среди пар из разных страт с различающимся outcome (аналог Somers' D); SE — аппроксимация Goodman & Kruskal (1963): `se = sqrt((1−strat²)·n/deno)`. При заданном `group` — декомпозиция на within/between и индексы по группам.
- Датасет `cpsmarch2015`: 14 358 мужчин 35–64 из March CPS 2015 (income, big_class, micro_class, education, weight).
