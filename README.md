# Atmospheric Particle Sandbox (Python)

Интерактивный прототип агентно-ориентированного моделирования атмосферы вокруг 3D-объекта.
Частицы отображаются как 3D-сферы и взаимодействуют:
- с другими частицами (сферические столкновения + spatial hashing),
- с объектом-препятствием,
- с полем среды (ветер, турбулентность, drag, buoyancy, gravity).

---

## 1. Быстрый старт

```bash
cd /Users/demn/particle-sandbox
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python main.py
```

По умолчанию загружается `assets/model_cube.obj`.

---

## 2. Что стало визуально понятнее

Чтобы сразу было видно, где объект и как ориентирована сцена:
- объект окрашен с направленной подсветкой и оттенками по нормалям граней;
- поверх сцены рисуется **желтый контур bounding-box** объекта;
- добавлены оси мира:
  - X = красная,
  - Y = зеленая,
  - Z = синяя;
- добавлена сетка-плоскость (grid) как ориентир масштаба и положения.

Если нет внешней модели, режим fallback явно виден в панели как `Mode: fallback cube`.

---

## 3. Готовые модели для загрузки

В `assets/` добавлены несколько моделей:
- `assets/model_cube.obj`
- `assets/model_cone.obj`
- `assets/model_sphere.obj`

Примеры:

```bash
python main.py --model assets/model_cube.obj
python main.py --model assets/model_cone.obj
python main.py --model assets/model_sphere.obj
```

Также можно загружать свои модели:

```bash
python main.py --model /path/to/model.obj
python main.py --model /path/to/model.glb
```

Для `.dae` и `.3ds` нужен Assimp:

```bash
brew install assimp
```

---

## 4. Управление

- `ЛКМ + drag` - orbit-камера.
- `ЛКМ click по объекту` - локальный импульс частиц в точке клика.
- `ПКМ` или `F` / `Enter` - порыв по направлению камеры.
- `Space` - пауза.
- `R` - сброс частиц.
- `Стрелки` или `W/A/S/D/Q/E` - поворот камеры (удобно с тачпадом).
- Скролл или `+/-` - zoom.
- `H` - скрыть/показать встроенную панель.
- `Esc` - выход.

---

## 5. Параметры симуляции (UI и CLI)

Через панель `Flow Controls` (ImGui внутри окна или fallback Tk/Web):
- `Wind X/Y/Z`
- `Turbulence`
- `Drag`
- `Buoyancy`
- `Gravity`
- `Particle Size`
- `Kick Strength`
- `Max dt`
- `Pause`, `Gust`, `Reset Particles`

Основные CLI-параметры:
- `--particles`
- `--size`
- `--wind WX WY WZ`
- `--drag --turbulence --buoyancy --gravity`
- `--kick-strength --max-dt`
- `--model`
- `--particle-config`

---

## 6. Конфигурация частиц из JSON

Можно задать начальные частицы в файле:

```bash
python main.py --particle-config assets/particle_config_example.json
```

Поддерживаемые поля частицы:
- `position: [x, y, z]` или `x/y/z`
- `velocity: [vx, vy, vz]` или `vx/vy/vz`
- `mass`
- `radius`

---

## 7. Проверка работоспособности (headless)

```bash
python main.py --verify --particles 600 --verify-steps 480
```

Ожидаемый результат:
- `Self-check OK: ...`

---

## 8. Архитектура проекта

- `main.py` - окно, ввод, камера, UI, цикл рендера/симуляции.
- `engine/loader.py` - загрузка 3D-моделей (включая fallback через Assimp).
- `engine/particles.py` - система частиц, шаг симуляции, инстансный рендер сфер.
- `engine/physics.py` - силы среды, collisions, spatial hash, self-check.
- `shaders/model.*` - шейдеры объекта.
- `shaders/particle.*` - шейдеры частиц.
- `shaders/line.*` - шейдеры визуальных направляющих (grid/axes/box).
- `assets/` - модели и примеры конфигурации.

---

## 9. Подробное НТО-описание

Полный подробный документ:
- `NTO_REPORT.md`
