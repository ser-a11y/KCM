# supports_data.py

"""
Типы систем координат:
========================
БСК (базовая СК) — система координат, жёстко связанная с телом.
  Оси: X — продольная, Y — поперечная, Z — вверх (перпендикулярно опорной плоскости).
  В БСК задаются координаты опор (поле x, y, z).
  Углы наклона БСК относительно горизонта:
    alpha — поворот вокруг оси Y (тангаж),
    beta  — поворот вокруг оси X (крен).

ГСК (глобальная СК) — неподвижная система координат.
  Ось Z направлена вертикально вверх (против силы тяжести).
  В ГСК задаются компоненты сил (Fx, Fy, Fz) и моментов (Tx, Ty, Tz).
  Сила тяжести в ГСК: g = (0, 0, -g_val).

Переход БСК -> ГСК:
  r_гск = R @ r_бск
  где R = Rx(beta) @ Ry(alpha)
  Ry(alpha) — поворот вокруг Y на alpha,
  Rx(beta)  — поворот вокруг X на beta.

Переход ГСК -> БСК:
  r_бск = R.T @ r_гск
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

# ==================== НАСТРОЙКИ ЗАДАЧИ ====================

# Масса тела (кг)
MASS = 100.0

# Углы наклона БСК (градусы)
ALPHA_DEG = 30 # тангаж (вокруг Y)
BETA_DEG  = 0   # крен (вокруг X)

# Известная координата ЦМ в БСК (одна из трёх, остальные две вычисляются)
# Задаётся в виде: ('x'/'y'/'z', значение_в_метрах)
KNOWN_CM_COORD = ('z', -0.25)  # например, известна Z координата ЦМ = 0.0 м
# KNOWN_CM_COORD = ('x', 0.15)  # или известна X
# KNOWN_CM_COORD = ('y', 0.0)   # или известна Y

# ==================== ОПОРЫ ====================
@dataclass
class Support:
    """
    Опора.
    Координаты (x, y, z) — в БСК (метры).
    Силы (Fx, Fy, Fz) и моменты (Tx, Ty, Tz) — в ГСК (Н, Н·м).
    Коэф. трения (fx, fy, fz) и диаметры шарнира (Dx, Dy, Dz) — в (м).    
    None означает "неизвестно, требуется найти".
    """    
    name: str
    x: float
    y: float
    z: float
    Fx: Optional[float] = None
    Fy: Optional[float] = None
    Fz: Optional[float] = None
    Tx: Optional[float] = None
    Ty: Optional[float] = None
    Tz: Optional[float] = None
    fx: Optional[float] = None
    fy: Optional[float] = None
    fz: Optional[float] = None
    Dx: Optional[float] = None
    Dy: Optional[float] = None
    Dz: Optional[float] = None


# Пример: тело на 4-х весах (известны только глобальные Fz)
supports = [
    Support("S1", x= 0.8, y= 0.8, z=0.0, Fz=245.25),
    Support("S2", x=-0.8, y= 0.8, z=0.0, Fz=245.25),
    Support("S3", x=-0.8, y=-0.8, z=0.0, Fz=245.25),
    Support("S4", x= 0.8, y=-0.8, z=0.0, Fz=245.25),
]
