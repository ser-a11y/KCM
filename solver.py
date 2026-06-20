# solver.py
from __future__ import annotations
import numpy as np
from typing import List, Tuple
from config import G_DEFAULT, MAX_FRICTION_ITER, FRICTION_TOL

def R_from_angles(alpha, beta):
    """Матрица поворота из БСК в ГСК."""
    Ry = np.array([
        [ np.cos(alpha), 0, np.sin(alpha)],
        [ 0,             1, 0             ],
        [-np.sin(alpha), 0, np.cos(alpha)]
    ])
    Rx = np.array([
        [1, 0,              0             ],
        [0, np.cos(beta),  -np.sin(beta) ],
        [0, np.sin(beta),   np.cos(beta) ]
    ])
    return Rx @ Ry


def build_system(supports: List[Support], m: float, 
                 known_coord: Tuple[str, float], R_T: np.ndarray, g_val: float):
    """
    Строит систему уравнений равновесия в ГСК.
    
    known_coord: ('x'/'y'/'z', значение) — известная координата ЦМ в БСК.
    R_T: матрица перехода ГСК -> БСК (R.T).
    """
    unknowns = []
    
    for i, s in enumerate(supports):
        for comp_name in ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']:
            if getattr(s, comp_name) is None:
                unknowns.append((i, comp_name))
    
    n_unknowns = 3 + len(unknowns)  # x_c, y_c, z_c в ГСК + компоненты
    n_eq = 7
    A = np.zeros((n_eq, n_unknowns))
    b = np.zeros(n_eq)
    
    F_weight = np.array([0.0, 0.0, -m * g_val])
    Fwx, Fwy, Fwz = F_weight
    
    # === Уравнения сил (строки 0,1,2) ===
    for i, s in enumerate(supports):
        for comp_name, axis in [('Fx', 0), ('Fy', 1), ('Fz', 2)]:
            val = getattr(s, comp_name)
            if val is not None:
                b[axis] -= val
            else:
                idx = 3 + unknowns.index((i, comp_name))
                A[axis, idx] = 1.0
    
    b[0:3] -= F_weight
    
    # === Уравнения моментов (строки 3,4,5) ===
    for i, s in enumerate(supports):
        r = np.array([s.x, s.y, s.z])
        
        for comp_name, axis in [('Fx', 0), ('Fy', 1), ('Fz', 2)]:
            val = getattr(s, comp_name)
            if val is not None:
                F_vec = np.zeros(3)
                F_vec[axis] = val
                b[3:6] -= np.cross(r, F_vec)
            else:
                idx = 3 + unknowns.index((i, comp_name))
                e = np.zeros(3)
                e[axis] = 1.0
                A[3:6, idx] += np.cross(r, e)
        
        for comp_name, axis in [('Tx', 0), ('Ty', 1), ('Tz', 2)]:
            val = getattr(s, comp_name)
            if val is not None:
                b[3 + axis] -= val
            else:
                idx = 3 + unknowns.index((i, comp_name))
                A[3 + axis, idx] = 1.0
    
    # Момент от веса: r_c × F_weight
    A[3, 0] = 0
    A[3, 1] = Fwz
    A[3, 2] = -Fwy
    
    A[4, 0] = -Fwz
    A[4, 1] = 0
    A[4, 2] = Fwx
    
    A[5, 0] = Fwy
    A[5, 1] = -Fwx
    A[5, 2] = 0
    
    # === Уравнение связи: известная координата ЦМ в БСК ===
    coord_name, coord_value = known_coord
    coord_idx = {'x': 0, 'y': 1, 'z': 2}[coord_name]
    
    # R_T[coord_idx] @ r_c_gsk = coord_value
    A[6, 0] = R_T[coord_idx, 0]
    A[6, 1] = R_T[coord_idx, 1]
    A[6, 2] = R_T[coord_idx, 2]
    b[6] = coord_value
    
    return A, b, unknowns

def solve_with_friction(supports, alpha, beta, m, known_coord, g_val, 
                         max_iter=MAX_FRICTION_ITER, tol=FRICTION_TOL):
    """
    Решает с учётом трения. Итерационно ограничивает моменты.
    
    Дополнительные поля в Support (если есть):
    - fx, fy, fz: коэффициенты трения
    - Dx, Dy, Dz: диаметры (м)
    Если какое-то из полей None — ограничение не применяется.
    """    
    R = R_from_angles(alpha, beta)
    rank = 0
    n_unknowns = 0
    n_equations = 0
    
    for iteration in range(max_iter):
        r_c_bsk, solved, rank, n_unknowns, n_equations = solve(
            supports, alpha, beta, m, known_coord, g_val)
        
        limited = False
        for s in solved:
            F = np.array([s.Fx or 0, s.Fy or 0, s.Fz or 0])
            F_abs = np.linalg.norm(F)
            
            for f_attr, d_attr, t_attr in [('fx', 'Dx', 'Tx'), ('fy', 'Dy', 'Ty'), ('fz', 'Dz', 'Tz')]:
                f = getattr(s, f_attr, None)
                D = getattr(s, d_attr, None)
                T = getattr(s, t_attr)
                
                if f is not None and D is not None and T is not None and F_abs > 1e-10:
                    T_max = f * F_abs * D / 2.0
                    if abs(T) > T_max + tol:
                        setattr(s, t_attr, np.sign(T) * T_max)
                        limited = True
        
        if not limited:
            break
    
    return r_c_bsk, solved, rank, n_unknowns, n_equations


def solve(supports: List[Support], alpha: float, beta: float, 
          m: float, known_coord: Tuple[str, float], g_val: float):
    """
    Основная функция.
    
    Параметры:
    - supports: список опор (координаты в БСК, силы/моменты в ГСК).
    - alpha, beta: углы наклона БСК (радианы).
    - m: масса тела (кг).
    - known_coord: ('x'/'y'/'z', значение) — известная координата ЦМ в БСК (м).
    
    Возвращает:
    - r_c_bsk: (x_c, y_c, z_c) — ЦМ в БСК.
    - supports: дополненные опоры.
    """
    R = R_from_angles(alpha, beta)
    R_T = R.T
    
    # Переводим координаты опор в ГСК
    for s in supports:
        r_bsk = np.array([s.x, s.y, s.z])
        s.x, s.y, s.z = R @ r_bsk
    
    A, b, unknowns = build_system(supports, m, known_coord, R_T, g_val)
    x, residuals, rank, singular = np.linalg.lstsq(A, b, rcond=None)

    n_unknowns = len(x)
    n_equations = A.shape[0]
    
    r_c_glob = x[0:3]
    
    for k, (i, comp_name) in enumerate(unknowns):
        setattr(supports[i], comp_name, x[3 + k])
    
    # ЦМ в БСК
    r_c_bsk = R_T @ r_c_glob
    
    # Возвращаем координаты опор в БСК
    for s in supports:
        r_gsk = np.array([s.x, s.y, s.z])
        s.x, s.y, s.z = R_T @ r_gsk
    
    return r_c_bsk, supports, rank, n_unknowns, n_equations

