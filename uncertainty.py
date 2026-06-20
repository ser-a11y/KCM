# uncertainty.py

"""
Расчёт неисключённой систематической погрешности (НСП)
методом линеаризации (частные производные численно).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from config import UNCERTAINTY_K


def calculate_uncertainty(project) -> dict:
    """
    Вычисляет НСП для координат ЦМ (x_c, y_c, z_c в БСК).
    
    Возвращает словарь:
    {
        'theta_x': float,
        'theta_y': float,
        'theta_z': float,
        'theta_mass': float,
        'contributions': [...],
        'k': float,
    }
    """
    if project.results is None:
        raise ValueError("Сначала выполните детерминированный расчёт (calculate).")
    
    nominal_x = project.results['x_c']
    nominal_y = project.results['y_c']
    nominal_z = project.results['z_c']
    
    # Какая координата фиксирована
    fixed_coord = project.known_cm_coord[0]  # 'x', 'y' или 'z'
    fixed_idx = {'x': 0, 'y': 1, 'z': 2}[fixed_coord]
    
    # Собираем все параметры с ненулевыми допусками
    params = _collect_params(project)
    
    contributions = []
    sum_sq = [0.0, 0.0, 0.0]  # x, y, z
    
    for p in params:
        # Численная производная центральной разностью
        x_plus, y_plus, z_plus = _solve_with_delta(project, p['name'], +p['theta'])
        x_minus, y_minus, z_minus = _solve_with_delta(project, p['name'], -p['theta'])
        
        df_dx = (x_plus - x_minus) / (2.0 * p['theta'])
        df_dy = (y_plus - y_minus) / (2.0 * p['theta'])
        df_dz = (z_plus - z_minus) / (2.0 * p['theta'])
        
        contrib_x = abs(p['theta'] * df_dx)
        contrib_y = abs(p['theta'] * df_dy)
        contrib_z = abs(p['theta'] * df_dz)
        
        contributions.append({
            'param': p['display_name'],
            'theta_i': p['theta'],
            'df_dx': df_dx,
            'df_dy': df_dy,
            'df_dz': df_dz,
            'contrib_x': contrib_x,
            'contrib_y': contrib_y,
            'contrib_z': contrib_z,
        })
        
        sum_sq[0] += contrib_x ** 2
        sum_sq[1] += contrib_y ** 2
        sum_sq[2] += contrib_z ** 2
    
    theta_x = UNCERTAINTY_K * np.sqrt(sum_sq[0])
    theta_y = UNCERTAINTY_K * np.sqrt(sum_sq[1])
    theta_z = UNCERTAINTY_K * np.sqrt(sum_sq[2])
    
    # Для фиксированной координаты НСП = заданный допуск
    if fixed_idx == 0:
        theta_x = project.d_known_cm
    elif fixed_idx == 1:
        theta_y = project.d_known_cm
    else:
        theta_z = project.d_known_cm
    
    # Погрешность массы
    if project.mass_from_forces:
        sum_sq_Fz = 0.0
        for s in project.supports:
            if s.Fz is not None and s.dFz > 0:
                sum_sq_Fz += s.dFz ** 2
        theta_mass = np.sqrt(sum_sq_Fz) / project.g_val
    else:
        theta_mass = project.d_mass        
    
    return {
        'theta_x': theta_x,
        'theta_y': theta_y,
        'theta_z': theta_z,
        'theta_mass': theta_mass,
        'contributions': contributions,
        'k': UNCERTAINTY_K,
    }

def _collect_params(project) -> list:
    """Собирает список параметров с ненулевыми допусками."""
    params = []
    
    # Масса
    if project.d_mass > 0:
        params.append({
            'name': 'mass',
            'display_name': 'Масса',
            'theta': project.d_mass,
        })
    
    # Углы
    if project.d_alpha_deg > 0:
        params.append({
            'name': 'alpha',
            'display_name': 'α (тангаж)',
            'theta': project.d_alpha_deg,
        })
    if project.d_beta_deg > 0:
        params.append({
            'name': 'beta',
            'display_name': 'β (крен)',
            'theta': project.d_beta_deg,
        })
    
    # Известная координата ЦМ
    if project.d_known_cm > 0:
        params.append({
            'name': 'known_cm',
            'display_name': f'Изв. коорд. ЦМ ({project.known_cm_coord[0]})',
            'theta': project.d_known_cm,
        })
    
    # Опоры
    for s in project.supports:
        _add_if_nonzero(params, f'{s.name}.x', f'{s.name} X', s.dx)
        _add_if_nonzero(params, f'{s.name}.y', f'{s.name} Y', s.dy)
        _add_if_nonzero(params, f'{s.name}.z', f'{s.name} Z', s.dz)
        _add_if_nonzero(params, f'{s.name}.Fx', f'{s.name} Fx', s.dFx)
        _add_if_nonzero(params, f'{s.name}.Fy', f'{s.name} Fy', s.dFy)
        _add_if_nonzero(params, f'{s.name}.Fz', f'{s.name} Fz', s.dFz)
    
    if project.results and 'friction_info' in project.results:
        for fi in project.results['friction_info']:
            for axis in ['x', 'y', 'z']:
                info = fi[axis]
                tol = info['tolerance']
                if tol > 0:
                    param_name = f"{fi['name']}.{axis}_moment"
                    display_name = f"{fi['name']} M{axis} ({info['mode']})"
                    _add_if_nonzero(params, param_name, display_name, tol)        
    
    return params

def _add_if_nonzero(params, name, display_name, theta):
    """Добавляет параметр, если допуск > 0."""
    if theta > 0:
        params.append({'name': name, 'display_name': display_name, 'theta': theta})


def _solve_with_delta(project, param_name: str, delta: float) -> Tuple[float, float]:
    """
    Решает систему с изменённым параметром.
    param_name — например "mass", "S1.x", "S1.Fx"
    delta — добавка к номинальному значению
    Возвращает (x_c, y_c) в БСК.
    """
    import copy
    from solver import solve_with_friction
    from supports_data import Support
    
    # Копируем проект и модифицируем параметр
    p = copy.deepcopy(project)
    
    # Применяем delta
    _apply_delta(p, param_name, delta)
    
    # Формируем supports для солвера
    supports = []
    for s in p.supports:
        fx = None if s.dTx != 0.0 else s.fx
        Dx = None if s.dTx != 0.0 else s.Dx
        fy = None if s.dTy != 0.0 else s.fy
        Dy = None if s.dTy != 0.0 else s.Dy
        fz = None if s.dTz != 0.0 else s.fz
        Dz = None if s.dTz != 0.0 else s.Dz
        
        supports.append(Support(
            name=s.name,
            x=s.x, y=s.y, z=s.z,
            Fx=s.Fx, Fy=s.Fy, Fz=s.Fz,
            Tx=s.Tx, Ty=s.Ty, Tz=s.Tz,
            fx=fx, fy=fy, fz=fz,
            Dx=Dx, Dy=Dy, Dz=Dz,
        ))
    
    alpha = np.deg2rad(p.alpha_deg)
    beta = np.deg2rad(p.beta_deg)
    
    if p.mass_from_forces:
        sum_Fz = sum(s.Fz for s in p.supports if s.Fz is not None)
        m = abs(sum_Fz) / p.g_val
    else:
        m = p.mass
    
    r_c_bsk, _, _, _, _ = solve_with_friction(
        supports, alpha, beta, m, p.known_cm_coord, p.g_val)
    
    return r_c_bsk[0], r_c_bsk[1], r_c_bsk[2]


def _apply_delta(project, param_name: str, delta: float):
    """Применяет изменение delta к параметру проекта (in-place)."""
    # Простые параметры
    if param_name == 'mass':
        project.mass += delta
        return
    if param_name == 'alpha':
        project.alpha_deg += delta
        return
    if param_name == 'beta':
        project.beta_deg += delta
        return
    if param_name == 'known_cm':
        coord, val = project.known_cm_coord
        project.known_cm_coord = (coord, val + delta)
        return
    
    # Параметры опор: "S1.x", "S1.Fx", "S1.x_moment" и т.д.
    parts = param_name.split('.')
    if len(parts) == 2:
        sname, attr = parts
        for s in project.supports:
            if s.name == sname:
                if attr.endswith('_moment'):
                    # Это допуск момента через трение — варьируем через f
                    axis = attr[0]  # 'x', 'y', 'z'
                    f_attr = 'f' + axis
                    # Меняем f так, чтобы T_max изменился на delta
                    # T_max = f * |F| * D/2
                    # Чтобы ΔT_max = delta, нужно Δf = delta / (|F| * D/2)
                    # Но |F| и D меняются... Упростим: варьируем Tx непосредственно
                    t_attr = 'T' + axis
                    current = getattr(s, t_attr)
                    if current is None:
                        current = 0.0
                    setattr(s, t_attr, current + delta)
                    # Также сбрасываем dT в ненулевое значение, чтобы трение игнорировалось
                    dt_attr = 'dT' + axis
                    setattr(s, dt_attr, abs(delta))  # временно, для этого расчёта
                else:
                    current = getattr(s, attr)
                    setattr(s, attr, current + delta)
                return
