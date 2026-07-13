# Backlog

Идеи для следующих версий `stratindex` (кандидаты на 0.2.0+). Зафиксировано 2026-07-12, после релиза 0.1.0.

**Статус:** весь бэклог реализован (ночь 12→13.07.2026) и **выпущен 13.07.2026 как 0.2.0**: https://pypi.org/project/stratindex/0.2.0/, GitHub release v0.2.0. Тесты: 63, CI зелёный, документация: https://promsoft.github.io/stratindex/. Проверено чистой установкой с PyPI (пример: strat=0.4128 за 0.14 с, bootstrap-SE и HTML-repr работают).

## Поддержка pandas Categorical с сохранением порядка уровней — ✅ сделано (0.2.0)
- [x] `Categorical.categories` уважается как порядок уровней для strata и group (в т.ч. `pd.Series` с dtype `category`); неиспользуемые категории отбрасываются с сохранением порядка — семантика R `factor()`; NA-коды (−1) → drop для strata / ошибка для group
- [ ] параметр `levels=` для явного порядка без pandas — отложено (Categorical покрывает основной сценарий)

## Ускорение ядра — ✅ сделано (0.2.0)
- [x] O(n log n): числитель — взвешенный подсчёт инверсий деревом Фенвика с блоками равных r; знаменатель — включения-исключения по блокам ties (T − A − B + AB); группы — независимый расчёт по подмножествам, between = total − within. Полный датасет с декомпозицией: 5.9 с → 0.15 с (~40×). Блочная O(n²)-версия сохранена как эталон в тестах (`pair_sums_blocked`)
- [ ] numba/Cython extra и распараллеливание блоков — отпали за ненадобностью после алгоритмического ускорения (python-цикл BIT ~0.1 с на 14k строк)

## Приём DataFrame напрямую — ✅ сделано (0.2.0)
- [x] `strat(df, outcome="income", strata="big_class", weights="weight", group="education")` — первый аргумент DataFrame или любой Mapping (в т.ч. dict из `load_cpsmarch2015()`); строковые keyword-аргументы разрешаются как имена колонок; `group_name` автоматически = имя колонки group; pandas не стал зависимостью (duck-typing по `.columns`/Mapping); позиционные вызовы массивами полностью совместимы

## Прочее (мелочи)
- [x] `repr` результатов в Jupyter: `_repr_html_` с таблицами (overall + декомпозиция + страты, экранирование меток) — ✅ 0.2.0
- [x] Бутстреп-SE: `strat(..., se_method="bootstrap", n_boot=200, random_state=...)`; ранги и порядок страт пересчитываются в каждой реплике; по умолчанию остаётся аппроксимация Goodman–Kruskal — ✅ 0.2.0
- [x] Документация: mkdocs + material + mkdocstrings, деплой на GitHub Pages (https://promsoft.github.io/stratindex/, workflow `docs.yml`); badges CI/PyPI/docs/license в README — ✅ 0.2.0
