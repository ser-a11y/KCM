# project.py — обновлённый SupportData и Project

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import json
from config import G_DEFAULT

@dataclass
class SupportData:
    """Данные одной опоры."""
    name: str
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
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
    
    # Допуски
    dx: float = 0.0      # допуск координаты X
    dy: float = 0.0
    dz: float = 0.0
    dFx: float = 0.0
    dFy: float = 0.0
    dFz: float = 0.0
    dTx: float = 0.0
    dTy: float = 0.0
    dTz: float = 0.0
    dfx: float = 0.0
    dfy: float = 0.0
    dfz: float = 0.0
    dDx: float = 0.0
    dDy: float = 0.0
    dDz: float = 0.0
    
    def to_dict(self):
        return {
            'name': self.name,
            'x': self.x, 'y': self.y, 'z': self.z,
            'Fx': self.Fx, 'Fy': self.Fy, 'Fz': self.Fz,
            'Tx': self.Tx, 'Ty': self.Ty, 'Tz': self.Tz,
            'fx': self.fx, 'fy': self.fy, 'fz': self.fz,
            'Dx': self.Dx, 'Dy': self.Dy, 'Dz': self.Dz,
            'dx': self.dx, 'dy': self.dy, 'dz': self.dz,
            'dFx': self.dFx, 'dFy': self.dFy, 'dFz': self.dFz,
            'dTx': self.dTx, 'dTy': self.dTy, 'dTz': self.dTz,
            'dfx': self.dfx, 'dfy': self.dfy, 'dfz': self.dfz,
            'dDx': self.dDx, 'dDy': self.dDy, 'dDz': self.dDz,
        }
    
    @classmethod
    def from_dict(cls, d):
        defaults = {
            'fx': None, 'fy': None, 'fz': None,
            'Dx': None, 'Dy': None, 'Dz': None,
            'dfx': 0.0, 'dfy': 0.0, 'dfz': 0.0,
            'dDx': 0.0, 'dDy': 0.0, 'dDz': 0.0,
        }
        for k, v in defaults.items():
            d.setdefault(k, v)
        return cls(**d)
    
class Project:
    """Проект с данными и результатами расчёта."""
    
    def __init__(self, name: str = "Новый проект"):
        self.g_val: float = G_DEFAULT  # будет переопределено из config при создании        
        self.name = name
        self.mass: float = 0.0
        self.d_mass: float = 0.0          # допуск массы (±)
        self.alpha_deg: float = 0.0
        self.d_alpha_deg: float = 0.0     # допуск α (±)
        self.beta_deg: float = 0.0
        self.d_beta_deg: float = 0.0      # допуск β (±)
        self.known_cm_coord: Tuple[str, float] = ('z', 0.0)
        self.d_known_cm: float = 0.0       # допуск известной координаты (±)
        self.supports: List[SupportData] = []
        self.stl_path: Optional[str] = None
        self.stl_scale: float = 0.001       # по умолчанию мм -> м
        self.results: Optional[dict] = None
        self.mass_from_forces: bool = False  # False = масса задана, True = по сумме Fz        
    
    def to_dict(self) -> dict:
        return {
            'g_val': self.g_val,            
            'name': self.name,
            'mass': self.mass,
            'd_mass': self.d_mass,
            'alpha_deg': self.alpha_deg,
            'd_alpha_deg': self.d_alpha_deg,
            'beta_deg': self.beta_deg,
            'd_beta_deg': self.d_beta_deg,
            'known_cm_coord': list(self.known_cm_coord),
            'd_known_cm': self.d_known_cm,
            'supports': [s.to_dict() for s in self.supports],
            'stl_path': self.stl_path,
            'stl_scale': self.stl_scale,
            'results': self.results,
            'mass_from_forces': self.mass_from_forces,            
        }
    
    @classmethod
    def from_dict(cls, d: dict):
        p = cls(d['name'])
        p.g_val = d.get('g_val', G_DEFAULT)
        p.mass = d['mass']
        p.d_mass = d.get('d_mass', 0.0)
        p.alpha_deg = d['alpha_deg']
        p.d_alpha_deg = d.get('d_alpha_deg', 0.0)
        p.beta_deg = d['beta_deg']
        p.d_beta_deg = d.get('d_beta_deg', 0.0)
        p.known_cm_coord = tuple(d['known_cm_coord'])
        p.d_known_cm = d.get('d_known_cm', 0.0)
        p.supports = [SupportData.from_dict(s) for s in d['supports']]
        p.stl_path = d.get('stl_path')
        p.stl_scale = d.get('stl_scale', 1.0)
        p.results = d.get('results')
        p.mass_from_forces = d.get('mass_from_forces', False)        
        return p
    
    def save(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            d = json.load(f)
        return cls.from_dict(d)
    
    def calculate(self):
        import numpy as np
        from solver import solve_with_friction, R_from_angles        
        from supports_data import Support
        
        supports = []
        for s in self.supports:
            # Если допуск момента задан явно (≠0) — трение для этой оси игнорируем
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
        
        alpha = np.deg2rad(self.alpha_deg)
        beta = np.deg2rad(self.beta_deg)
        
        # Если масса по сумме Fz — пересчитываем
        if self.mass_from_forces:
            sum_Fz = sum(s.Fz for s in self.supports if s.Fz is not None)
            # В ГСК Fz — это глобально вертикальные силы, сумма = mg (для равновесия)
            self.mass = abs(sum_Fz) / self.g_val
        # Иначе используется self.mass как задано
        
        r_c_bsk, solved, rank, n_unknowns, n_equations = solve_with_friction(
            supports, alpha, beta, self.mass, self.known_cm_coord, self.g_val)        
        
        x_c, y_c, z_c = r_c_bsk
        self.results = {
            'x_c': x_c,
            'y_c': y_c,
            'z_c': z_c,
            'supports': [],
            'mass_used': self.mass,
        }
        for s in solved:
            self.results['supports'].append({
                'name': s.name,
                'Fx': s.Fx, 'Fy': s.Fy, 'Fz': s.Fz,
                'Tx': s.Tx, 'Ty': s.Ty, 'Tz': s.Tz,
            })
        
        # Суммы и невязки в ГСК
        R_mat = R_from_angles(alpha, beta)
        
        sum_F = np.zeros(3)
        sum_M = np.zeros(3)
        
        for s in solved:
            F_gsk = np.array([s.Fx or 0, s.Fy or 0, s.Fz or 0])
            T_gsk = np.array([s.Tx or 0, s.Ty or 0, s.Tz or 0])
            r_bsk = np.array([s.x, s.y, s.z])
            r_gsk = R_mat @ r_bsk
            sum_F += F_gsk
            sum_M += np.cross(r_gsk, F_gsk) + T_gsk
        
        mg = self.mass * self.g_val
        F_mg = np.array([0.0, 0.0, -mg])
        r_c_gsk = R_mat @ r_c_bsk
        M_mg = np.cross(r_c_gsk, F_mg)
        
        # Невязки (должны быть ~0)
        residual_F = sum_F + F_mg
        residual_M = sum_M + M_mg
        
        self.results.update({
            'sum_Fx': float(sum_F[0]), 'sum_Fy': float(sum_F[1]), 'sum_Fz': float(sum_F[2]),
            'sum_Mx': float(sum_M[0]), 'sum_My': float(sum_M[1]), 'sum_Mz': float(sum_M[2]),
            'mg': float(mg),
            'mg_Fx': float(F_mg[0]), 'mg_Fy': float(F_mg[1]), 'mg_Fz': float(F_mg[2]),
            'mg_Mx': float(M_mg[0]), 'mg_My': float(M_mg[1]), 'mg_Mz': float(M_mg[2]),
            'res_Fx': float(residual_F[0]), 'res_Fy': float(residual_F[1]), 'res_Fz': float(residual_F[2]),
            'res_Mx': float(residual_M[0]), 'res_My': float(residual_M[1]), 'res_Mz': float(residual_M[2]),
            'rank': int(rank),
            'n_unknowns': int(n_unknowns),
            'n_equations': int(n_equations),
            'underdetermined': bool(rank < n_unknowns),
        })

        # Информация о трении и допусках моментов
        friction_info = []
        for s, s_data in zip(solved, self.supports):
            F = np.array([s.Fx or 0, s.Fy or 0, s.Fz or 0])
            F_abs = np.linalg.norm(F)
            info = {'name': s.name}
            for axis, f_attr, d_attr, t_attr, dt_attr in [
                ('x', 'fx', 'Dx', 'Tx', 'dTx'),
                ('y', 'fy', 'Dy', 'Ty', 'dTy'),
                ('z', 'fz', 'Dz', 'Tz', 'dTz')
            ]:
                f = getattr(s_data, f_attr)
                D = getattr(s_data, d_attr)
                T_found = getattr(s, t_attr) or 0
                dt_val = getattr(s_data, dt_attr)
                
                # Вычисляем T_max если заданы f и D
                if f is not None and D is not None:
                    T_max = f * F_abs * D / 2.0
                else:
                    T_max = None
                
                # Если допуск задан явно (≠0) — трение игнорируется
                if dt_val != 0.0:
                    info[axis] = {
                        'mode': 'явный',
                        'T': T_found,
                        'T_max': None,
                        'tolerance': dt_val,
                    }
                elif T_max is not None:
                    info[axis] = {
                        'mode': 'трение',
                        'T': T_found,
                        'T_max': T_max,
                        'tolerance': T_max,
                    }
                else:
                    info[axis] = {
                        'mode': 'жёсткая',
                        'T': T_found,
                        'T_max': None,
                        'tolerance': 0.0,
                    }
            friction_info.append(info)

                    # Расчёт НСП
         # Сохраняем friction_info ДО расчёта НСП
        self.results['friction_info'] = friction_info

        # Расчёт НСП
        try:
            from uncertainty import calculate_uncertainty
            uncertainty = calculate_uncertainty(self)
            self.results['uncertainty'] = uncertainty
        except Exception as e:
            self.results['uncertainty_error'] = str(e)
