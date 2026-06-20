# visualize.py

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
from supports_data import Support
from stl import mesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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


def load_stl_mesh(filepath: str):
    """Загружает STL-файл."""
    return mesh.Mesh.from_file(filepath)


def plot_stl(ax, stl_mesh, R, color='cyan', alpha=0.3, scale=1.0):
    """
    Отрисовывает STL-модель, повёрнутую согласно R (БСК -> ГСК).
    scale — масштаб (0.001 если модель в мм).
    """
    # Вершины STL в БСК (после масштабирования)
    vertices_bsk = stl_mesh.vectors * scale  # (n, 3, 3)
    
    # Поворачиваем в ГСК
    n_tri = vertices_bsk.shape[0]
    vertices_gsk = np.zeros_like(vertices_bsk)
    for i in range(n_tri):
        for j in range(3):
            vertices_gsk[i, j] = R @ vertices_bsk[i, j]
    
    polys = Poly3DCollection(vertices_gsk, alpha=alpha, facecolor=color,
                              edgecolor='gray', linewidth=0.1)
    ax.add_collection3d(polys)
    return vertices_gsk  # возвращаем для расчёта лимитов


def plot_system(supports: List[Support], x_c_bsk: float, y_c_bsk: float, z_c_bsk: float,
                alpha: float, beta: float, m: float, G_VAL: float = 9.81,
                scale_force: float = 0.005,
                stl_path: Optional[str] = None, stl_scale: float = 1.0):
    """
    Визуализация в ГСК.
    
    Параметры:
    - supports: опоры с координатами в БСК, силами/моментами в ГСК.
    - x_c_bsk, y_c_bsk, z_c_bsk: ЦМ в БСК.
    - alpha, beta: углы наклона БСК.
    """
    R = R_from_angles(alpha, beta)
    
    # ЦМ в ГСК
    r_c_bsk = np.array([x_c_bsk, y_c_bsk, z_c_bsk])
    r_c_gsk = R @ r_c_bsk
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # --- STL-модель ---
    all_coords = []
    if stl_path:
        stl_mesh = load_stl_mesh(stl_path)
        verts_gsk = plot_stl(ax, stl_mesh, R, scale=stl_scale)
        all_coords.extend(verts_gsk.reshape(-1, 3))
    
    # --- Опоры и силы ---
    for s in supports:
        r_bsk = np.array([s.x, s.y, s.z])
        r_gsk = R @ r_bsk
        
        all_coords.append(r_gsk)
        
        # Точка опоры в ГСК
        ax.scatter(*r_gsk, color='red', s=50)
        ax.text(r_gsk[0], r_gsk[1], r_gsk[2], f'  {s.name}', color='red', fontsize=9)
        
        # Сила (уже в ГСК)
        F = np.array([s.Fx or 0, s.Fy or 0, s.Fz or 0])
        F_norm = np.linalg.norm(F)
        if F_norm > 1e-6:
            F_dir = F / F_norm * scale_force * F_norm
            ax.quiver(r_gsk[0], r_gsk[1], r_gsk[2],
                      F_dir[0], F_dir[1], F_dir[2],
                      color='orange', linewidth=2, arrow_length_ratio=0.15)
    
    # --- ЦМ ---
    all_coords.append(r_c_gsk)
    ax.scatter(*r_c_gsk, color='green', s=100)
    ax.text(r_c_gsk[0], r_c_gsk[1], r_c_gsk[2], '  ЦМ', color='green', fontsize=10)
    
    # Вес в ГСК (всегда вертикально вниз)
    F_weight = np.array([0.0, 0.0, -project.mass * g_val])
    Fw_dir = F_weight * scale_force
    ax.quiver(r_c_gsk[0], r_c_gsk[1], r_c_gsk[2],
              Fw_dir[0], Fw_dir[1], Fw_dir[2],
              color='blue', linewidth=2, arrow_length_ratio=0.15)
    
    # --- Настройка ---
    ax.set_xlabel('X (ГСК)')
    ax.set_ylabel('Y (ГСК)')
    ax.set_zlabel('Z (ГСК, вверх)')
    ax.set_title('Система опор и сил (ГСК)')
    
    # Автомасштаб
    all_coords = np.array(all_coords)
    max_range = np.ptp(all_coords, axis=0).max() / 2.0
    mid = np.mean(all_coords, axis=0)
    if max_range > 0:
        ax.set_xlim(mid[0] - max_range*1.2, mid[0] + max_range*1.2)
        ax.set_ylim(mid[1] - max_range*1.2, mid[1] + max_range*1.2)
        ax.set_zlim(mid[2] - max_range*1.2, mid[2] + max_range*1.2)
    
    plt.show()


def plot_on_axes(ax, project, stl_alpha=0.3):
    """
    Рисует систему на переданных осях ax.
    """
    alpha = np.deg2rad(project.alpha_deg)
    beta = np.deg2rad(project.beta_deg)
    R = R_from_angles(alpha, beta)
    
    g_val = project.g_val
    
    # === Авто-масштаб стрелок ===
    # Собираем все силы для определения максимальной
    all_forces = []
    for s_data in project.supports:
        if project.results:
            res_s = next((rs for rs in project.results['supports'] if rs['name'] == s_data.name), None)
        else:
            res_s = None
        if res_s:
            F_vec = np.array([res_s['Fx'] or 0, res_s['Fy'] or 0, res_s['Fz'] or 0])
        else:
            F_vec = np.array([s_data.Fx or 0, s_data.Fy or 0, s_data.Fz or 0])
        all_forces.append(np.linalg.norm(F_vec))
    
    # Добавляем вес
    all_forces.append(project.mass * g_val)
    max_force = max(all_forces) if all_forces else 1.0
    
    # Размер сцены (размах координат опор в БСК)
    all_pts = np.array([[s.x, s.y, s.z] for s in project.supports])
    scene_size = np.ptp(all_pts).max() if len(all_pts) > 0 else 1.0
    if scene_size < 0.01:
        scene_size = 1.0
    
    # Максимальная стрелка — 75% от размера сцены
    desired_max_arrow = scene_size * 0.75
    scale_force = desired_max_arrow / max_force if max_force > 1e-10 else 0.01
    
    all_coords = []    
    
    # --- STL-модель ---
    if project.stl_path:
        stl_mesh = load_stl_mesh(project.stl_path)
        vertices_bsk = stl_mesh.vectors * project.stl_scale
        n_tri = vertices_bsk.shape[0]
        vertices_gsk = np.zeros_like(vertices_bsk)
        for i in range(n_tri):
            for j in range(3):
                vertices_gsk[i, j] = R @ vertices_bsk[i, j]
        
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        polys = Poly3DCollection(vertices_gsk, alpha=stl_alpha, facecolor='cyan',
                                  edgecolor='gray', linewidth=0.1)
        ax.add_collection3d(polys)
        all_coords.extend(vertices_gsk.reshape(-1, 3))
    
    # --- Оси БСК (в ГСК) ---
    origin_bsk = np.array([0.0, 0.0, 0.0])
    origin_gsk = R @ origin_bsk
    axis_len = scene_size * 0.3  # 30% от размера сцены
    
    colors_bsk = ['red', 'green', 'blue']
    labels_bsk = ['Xбск', 'Yбск', 'Zбск']
    for axis in range(3):
        v_bsk = np.zeros(3)
        v_bsk[axis] = axis_len
        v_gsk = R @ v_bsk
        ax.quiver(origin_gsk[0], origin_gsk[1], origin_gsk[2],
                  v_gsk[0], v_gsk[1], v_gsk[2],
                  color=colors_bsk[axis], linewidth=2, arrow_length_ratio=0.15)
        ax.text(origin_gsk[0] + v_gsk[0], origin_gsk[1] + v_gsk[1], origin_gsk[2] + v_gsk[2],
                labels_bsk[axis], color=colors_bsk[axis], fontsize=10)
    
    # --- Оси ГСК ---
    axis_len_glob = scene_size * 0.35  # чуть длиннее
    colors_gsk = ['dimgray', 'dimgray', 'dimgray']
    labels_gsk = ['Xгск', 'Yгск', 'Zгск']
    for axis in range(3):
        v = np.zeros(3)
        v[axis] = axis_len_glob
        ax.quiver(0, 0, 0, v[0], v[1], v[2],
                  color=colors_gsk[axis], linewidth=1, arrow_length_ratio=0.15, linestyle='--')
        ax.text(v[0], v[1], v[2], labels_gsk[axis], color='gray', fontsize=8)        
    
    # --- Опоры и силы ---
    for s_data in project.supports:
        r_bsk = np.array([s_data.x, s_data.y, s_data.z])
        r_gsk = R @ r_bsk
        
        all_coords.append(r_gsk)
        
        ax.scatter(*r_gsk, color='red', s=50)
        ax.text(r_gsk[0], r_gsk[1], r_gsk[2], f'  {s_data.name}', color='red', fontsize=9)
        
        # Берём силы из результатов
        if project.results:
            res_s = next((rs for rs in project.results['supports'] if rs['name'] == s_data.name), None)
        else:
            res_s = None
        
        if res_s:
            F_vec = np.array([res_s['Fx'] or 0, res_s['Fy'] or 0, res_s['Fz'] or 0])
        else:
            F_vec = np.array([s_data.Fx or 0, s_data.Fy or 0, s_data.Fz or 0])
        
        # Раскладываем на составляющие
        comp_colors = ['#FF4444', '#44FF44', '#4444FF']  # красный, зелёный, синий для Fx, Fy, Fz
        comp_labels = ['Fx', 'Fy', 'Fz']
        for axis in range(3):
            comp = np.zeros(3)
            comp[axis] = F_vec[axis] * scale_force
            if np.abs(comp[axis]) > 1e-8:
                ax.quiver(r_gsk[0], r_gsk[1], r_gsk[2],
                          comp[0], comp[1], comp[2],
                          color=comp_colors[axis], linewidth=1.5, arrow_length_ratio=0.15)
    
    # --- ЦМ ---
    if project.results:
        x_c, y_c, z_c = project.results['x_c'], project.results['y_c'], project.results['z_c']
    else:
        x_c = y_c = z_c = 0.0
    
    r_c_bsk = np.array([x_c, y_c, z_c])
    r_c_gsk = R @ r_c_bsk
    
    all_coords.append(r_c_gsk)
    ax.scatter(*r_c_gsk, color='green', s=100)
    ax.text(r_c_gsk[0], r_c_gsk[1], r_c_gsk[2], '  ЦМ', color='green', fontsize=10)
    
    # Вес
    F_weight = np.array([0.0, 0.0, -project.mass * g_val])
    Fw_dir = F_weight * scale_force
    ax.quiver(r_c_gsk[0], r_c_gsk[1], r_c_gsk[2],
              Fw_dir[0], Fw_dir[1], Fw_dir[2],
              color='blue', linewidth=2, arrow_length_ratio=0.15)
    
    # --- Одинаковый масштаб ---
    ax.set_xlabel('X (ГСК)')
    ax.set_ylabel('Y (ГСК)')
    ax.set_zlabel('Z (ГСК)')
    
    if all_coords:
        all_coords = np.array(all_coords)
        max_range = np.ptp(all_coords, axis=0).max() / 2.0
        mid = np.mean(all_coords, axis=0)
        if max_range > 0:
            ax.set_xlim(mid[0] - max_range*1.2, mid[0] + max_range*1.2)
            ax.set_ylim(mid[1] - max_range*1.2, mid[1] + max_range*1.2)
            ax.set_zlim(mid[2] - max_range*1.2, mid[2] + max_range*1.2)
    
    # Одинаковый масштаб по всем осям
    ax.set_box_aspect([1.0, 1.0, 1.0])    
