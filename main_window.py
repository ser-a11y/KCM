# main_window.py

import sys
import os
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QMenuBar, QMenu, QTabWidget,
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QComboBox, QDoubleSpinBox, QFileDialog,
    QMessageBox, QHeaderView, QGroupBox, QFormLayout, QSplitter,
    QCheckBox, QTextEdit
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from config import G_CITIES, G_DEFAULT, G_DEFAULT_CITY

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import numpy as np
from project import Project, SupportData

import traceback

def resource_path(relative_path):
    """Получить абсолютный путь к ресурсу, работает и в exe, и при отладке."""
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(__file__)
    return os.path.join(base_path, relative_path)

legend_path = resource_path("legend.png")

def excepthook(type, value, tb):
    traceback.print_exception(type, value, tb)
    sys.exit(1)
sys.excepthook = excepthook


class MplCanvas(FigureCanvas):
    """Холст matplotlib для встраивания в Qt."""
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        super().__init__(self.fig)
        self.setParent(parent)


class ParamsPanel(QWidget):
    """Панель параметров проекта (таблица с допусками)."""
    def _on_g_city_changed(self, city):
        g = G_CITIES.get(city, G_DEFAULT)
        self.g_spin.blockSignals(True)
        self.g_spin.setValue(g)
        self.g_spin.blockSignals(False)
        self.project.g_val = g
    
    def _on_g_value_changed(self, value):
        self.g_city_combo.blockSignals(True)
        self.g_city_combo.setCurrentText("Другое")
        self.g_city_combo.blockSignals(False)
        self.project.g_val = value    
    
    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self.project = project
        self._setup_ui()
    
    def _on_mass_mode_changed(self, checked):
        self.project.mass_from_forces = checked
        self.mass_spin.setEnabled(not checked)
        self.d_mass_spin.setEnabled(not checked)        
        
    def _on_cell_changed(self, row, col):
        """Обновляет цвет ячейки сразу после редактирования."""
        item = self.table.item(row, col)
        if item is None:
            return
        
        is_tolerance_row = (row % 2 == 1)
        attrs_val = ['x','y','z','Fx','Fy','Fz','Tx','Ty','Tz','fx','fy','fz','Dx','Dy','Dz']
        attr_idx = col - 1
        
        if is_tolerance_row:
            text = item.text().strip().replace('±', '').strip()
            if text == "" or text == "0" or text == "0.0000":
                item.setText("±0")
                item.setForeground(Qt.gray)
            else:
                try:
                    val = float(text)
                    item.setText(f"±{val:.4f}")
                    item.setForeground(Qt.black)
                except ValueError:
                    item.setForeground(Qt.gray)
        elif attr_idx >= 0 and attr_idx < len(attrs_val):
            attr = attrs_val[attr_idx]
            text = item.text().strip()
            if text == "" or text == "жесткая":
                if attr in ('x', 'y', 'z'):
                    item.setForeground(Qt.black)
                else:
                    item.setForeground(Qt.gray)
            else:
                try:
                    float(text)
                    item.setForeground(Qt.black)
                except ValueError:
                    if attr in ('x', 'y', 'z'):
                        item.setForeground(Qt.black)
                    else:
                        item.setForeground(Qt.gray)        

    def _setup_ui(self):
        layout = QVBoxLayout(self)        
            # --- Основные параметры ---
        form = QFormLayout()
        
        # Масса
        mass_layout = QHBoxLayout()
        self.mass_spin = QDoubleSpinBox()
        self.mass_spin.setRange(0.0, 1e6)
        self.mass_spin.setDecimals(3)
        self.mass_spin.setSuffix(" кг")
        self.mass_spin.setValue(self.project.mass)
        self.mass_spin.setFixedWidth(200)  # ← фиксированная ширина           
        mass_layout.addWidget(self.mass_spin)
        mass_layout.addWidget(QLabel("±"))
        self.d_mass_spin = QDoubleSpinBox()
        self.d_mass_spin.setRange(0.0, 1e6)
        self.d_mass_spin.setDecimals(4)
        self.d_mass_spin.setValue(self.project.d_mass)
        self.d_mass_spin.setFixedWidth(100)  # ← фиксированная ширина        
        mass_layout.addWidget(self.d_mass_spin)
        mass_layout.addStretch()
        form.addRow("Масса:", mass_layout)

        # Переключатель: масса задана / по сумме Fz
        self.mass_mode_check = QCheckBox("Определять массу по сумме Fz")
        self.mass_mode_check.setChecked(self.project.mass_from_forces)
        self.mass_mode_check.toggled.connect(self._on_mass_mode_changed)
        form.addRow("", self.mass_mode_check)

        # Ускорение свободного падения
        g_layout = QHBoxLayout()
        g_layout.addWidget(QLabel("g:"))
        
        self.g_city_combo = QComboBox()
        self.g_city_combo.setFixedWidth(150)
        self.g_city_combo.addItems(G_CITIES.keys())
        self.g_city_combo.currentTextChanged.connect(self._on_g_city_changed)
        g_layout.addWidget(self.g_city_combo)
        
        self.g_spin = QDoubleSpinBox()
        self.g_spin.setRange(0.01, 1000.0)
        self.g_spin.setDecimals(5)
        self.g_spin.setSingleStep(0.0001)
        self.g_spin.setFixedWidth(120)
        self.g_spin.setSuffix(" м/с²")        
        self.g_spin.setValue(self.project.g_val)
        self.g_spin.valueChanged.connect(self._on_g_value_changed)
        g_layout.addWidget(self.g_spin)
        
        g_layout.addStretch()
        form.addRow("", g_layout)

        # Установим начальный город и значение из конфига
        idx = self.g_city_combo.findText(G_DEFAULT_CITY)
        if idx >= 0:
            self.g_city_combo.setCurrentIndex(idx)
        self.g_spin.setValue(G_DEFAULT)        
        
        # α (тангаж)
        alpha_layout = QHBoxLayout()
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(-90, 90)
        self.alpha_spin.setDecimals(4)
        self.alpha_spin.setSuffix("°")
        self.alpha_spin.setValue(self.project.alpha_deg)
        self.alpha_spin.setFixedWidth(200)
        alpha_layout.addWidget(self.alpha_spin)
        alpha_layout.addWidget(QLabel("±"))
        self.d_alpha_spin = QDoubleSpinBox()
        self.d_alpha_spin.setRange(0.0, 90)
        self.d_alpha_spin.setDecimals(4)
        self.d_alpha_spin.setValue(self.project.d_alpha_deg)
        self.d_alpha_spin.setFixedWidth(100)
        alpha_layout.addWidget(self.d_alpha_spin)
        alpha_layout.addStretch()
        form.addRow("α (тангаж):", alpha_layout)
        
        # β (крен)
        beta_layout = QHBoxLayout()
        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setRange(-90, 90)
        self.beta_spin.setDecimals(4)
        self.beta_spin.setSuffix("°")
        self.beta_spin.setValue(self.project.beta_deg)
        self.beta_spin.setFixedWidth(200)
        beta_layout.addWidget(self.beta_spin)
        beta_layout.addWidget(QLabel("±"))
        self.d_beta_spin = QDoubleSpinBox()
        self.d_beta_spin.setRange(0.0, 90)
        self.d_beta_spin.setDecimals(4)
        self.d_beta_spin.setValue(self.project.d_beta_deg)
        self.d_beta_spin.setFixedWidth(100)
        beta_layout.addWidget(self.d_beta_spin)
        beta_layout.addStretch()
        form.addRow("β (крен):", beta_layout)
        
        # Известная координата ЦМ
        cm_layout = QHBoxLayout()
        self.cm_coord_combo = QComboBox()
        self.cm_coord_combo.addItems(['x', 'y', 'z'])
        self.cm_coord_combo.setCurrentText(self.project.known_cm_coord[0])
        self.cm_coord_combo.setFixedWidth(80)
        cm_layout.addWidget(self.cm_coord_combo)
        cm_layout.addWidget(QLabel("="))
        self.cm_value_spin = QDoubleSpinBox()
        self.cm_value_spin.setRange(-100, 100)
        self.cm_value_spin.setDecimals(4)
        self.cm_value_spin.setSuffix(" м")
        self.cm_value_spin.setValue(self.project.known_cm_coord[1])
        self.cm_value_spin.setFixedWidth(99)
        cm_layout.addWidget(self.cm_value_spin)
        cm_layout.addWidget(QLabel("±"))
        self.d_cm_spin = QDoubleSpinBox()
        self.d_cm_spin.setRange(0.0, 100)
        self.d_cm_spin.setDecimals(4)
        self.d_cm_spin.setValue(self.project.d_known_cm)
        self.d_cm_spin.setFixedWidth(100)
        cm_layout.addWidget(self.d_cm_spin)
        cm_layout.addStretch()
        form.addRow("Изв. коорд. ЦМ:", cm_layout)
        
        layout.addLayout(form)
        
        # --- Таблица опор ---
        hints_label = QLabel(
            "Координаты опор X Y Z задаются в БСК (СК объекта)\n"
            "Величины сил и моментов — в ГСК (глобальная СК, связанная с горизонтом)\n"
            "Опоры: пустая ячейка = жёсткая связь, значение определяется расчётом; 0 = свободно (плавает/скользит); число = заданная нагрузка\n"
            "Моменты сил Tx Ty Tz можно задать напрямую, через коэффициенты трения fx fy fz и диаметры шарнира Dx Dy Dz, или комбинированно\n"
            "Если допуски dTx dTy dTz моментов сил заданы напрямую, трение игнорируется\n"
            "Для расчётов в кгс установите g = 1.0 или выберите техническую систему из списка"        
        )
        hints_label.setStyleSheet("color: #37474f; font-size: 11px;")
        layout.addWidget(hints_label)        
        layout.addWidget(QLabel("Опоры:"))
        self.table = QTableWidget()
        self.table.setColumnCount(17)
        self.table.setHorizontalHeaderLabels([
            "Имя", "X, м", "Y, м", "Z, м",
             "Fx, Н", "Fy, Н", "Fz, Н",
             "Tx, Н·м", "Ty, Н·м", "Tz, Н·м",
             "fx", "fy", "fz",
             "Dx, м", "Dy, м", "Dz, м"
        ] + [" "] * 0)  # placeholder — колонок пока 11 + 1 пустая
        # На самом деле колонок 12: имя + 10 параметров + не хватает?
        # Пересчитаем: имя, x,y,z, Fx,Fy,Fz, Tx,Ty,Tz, f = 11 колонок
        self.table.setColumnCount(16)
        self.table.setHorizontalHeaderLabels([
            "Имя", "X, м", "Y, м", "Z, м",
             "Fx, Н", "Fy, Н", "Fz, Н",
             "Tx, Н·м", "Ty, Н·м", "Tz, Н·м",
             "fx", "fy", "fz",
             "Dx, м", "Dy, м", "Dz, м"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table)
        self.table.cellChanged.connect(self._on_cell_changed)        
        
        # Кнопки
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Добавить опору")
        add_btn.clicked.connect(self._add_support)
        del_btn = QPushButton("Удалить опору")
        del_btn.clicked.connect(self._delete_support)
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(del_btn)
        layout.addLayout(btn_layout)
        
        self._refresh_table()
    
    def _refresh_table(self):
        """Обновляет таблицу, сохранив текущие данные."""
        self._save_table_to_supports()
        
        n = len(self.project.supports)
        self.table.setRowCount(n * 2)
        
        for i, s in enumerate(self.project.supports):
            row_val = i * 2
            row_tol = i * 2 + 1
            
            # Строка значений — имя опоры
            name_item = QTableWidgetItem(s.name)
            name_item.setForeground(Qt.black)
            self.table.setItem(row_val, 0, name_item)
            
            attrs_val = ['x','y','z','Fx','Fy','Fz','Tx','Ty','Tz','fx','fy','fz','Dx','Dy','Dz']
            attrs_tol = ['dx','dy','dz','dFx','dFy','dFz','dTx','dTy','dTz','dfx','dfy','dfz','dDx','dDy','dDz']
            
            for j, (attr_val, attr_tol) in enumerate(zip(attrs_val, attrs_tol)):
                val = getattr(s, attr_val)
                
                if attr_val in ('x', 'y', 'z'):
                    # Координаты всегда числа
                    text = f"{val:.4f}" if val is not None else "0.0000"
                    color = Qt.black
                elif val is None:
                    text = "жесткая"
                    color = Qt.gray
                else:
                    text = f"{val:.4f}"
                    color = Qt.black
                
                item = QTableWidgetItem(text)
                item.setForeground(color)
                self.table.setItem(row_val, j+1, item)
                
                # Строка допусков
                tol = getattr(s, attr_tol)       
                if tol and tol != 0.0:
                    tol_text = f"±{tol:.4f}"
                    tol_color = Qt.black
                else:
                    tol_text = "±0"
                    tol_color = Qt.gray
                tol_item = QTableWidgetItem(tol_text)
                tol_item.setForeground(tol_color)
                self.table.setItem(row_tol, j+1, tol_item)                
            
            self.table.setItem(row_tol, 0, QTableWidgetItem("± допуск"))
    
    def _save_table_to_supports(self):
        """Сохраняет текущее содержимое таблицы в project.supports (без удаления)."""
        if self.table.rowCount() == 0:
            return
        
        attrs_val = ['x','y','z','Fx','Fy','Fz','Tx','Ty','Tz','fx','fy','fz','Dx','Dy','Dz']
        attrs_tol = ['dx','dy','dz','dFx','dFy','dFz','dTx','dTy','dTz','dfx','dfy','dfz','dDx','dDy','dDz']
        
        for i in range(len(self.project.supports)):
            row_val = i * 2
            row_tol = i * 2 + 1
            
            if row_val >= self.table.rowCount():
                break
            
            s = self.project.supports[i]
            
            # Имя
            item_name = self.table.item(row_val, 0)
            if item_name:
                s.name = item_name.text()
            
            # Значения
            for j, attr in enumerate(attrs_val):
                item = self.table.item(row_val, j+1)
                if item:
                    text = item.text().strip()
                    if text == "" or text == "жесткая":
                        # Координаты не могут быть None, остальное — None (жёсткая связь)
                        if attr in ('x', 'y', 'z'):
                            setattr(s, attr, 0.0)
                        else:
                            setattr(s, attr, None)
                    else:
                        try:
                            setattr(s, attr, float(text))
                        except ValueError:
                            pass
                else:
                    if attr in ('x', 'y', 'z'):
                        setattr(s, attr, 0.0)
                    else:
                        setattr(s, attr, None)
            
            # Допуски
            for j, attr in enumerate(attrs_tol):
                item = self.table.item(row_tol, j+1)
                if item and item.text().strip():
                    text = item.text().replace('±', '').strip()
                    try:
                        setattr(s, attr, float(text))
                    except ValueError:
                        pass
                    
    def _add_support(self):
        n = len(self.project.supports) + 1
        s = SupportData(
            name=f"S{n}",
            x=0.0, y=0.0, z=0.0,
            Fx=None, Fy=None, Fz=None,
            Tx=None, Ty=None, Tz=None,
            fx=None, fy=None, fz=None,
            Dx=None, Dy=None, Dz=None,
        )
        self.project.supports.append(s)
        self._refresh_table()
    
    def _delete_support(self):
        row = self.table.currentRow()
        if row >= 0:
            idx = row // 2  # индекс опоры (две строки на опору)
            if idx < len(self.project.supports):
                del self.project.supports[idx]
                self._refresh_table()
    
    def save_to_project(self):
        """Сохраняет данные из виджетов в project."""
        self._save_table_to_supports()  # сначала таблицу        
        self.project.mass = self.mass_spin.value()
        self.project.d_mass = self.d_mass_spin.value()
        self.project.alpha_deg = self.alpha_spin.value()
        self.project.d_alpha_deg = self.d_alpha_spin.value()
        self.project.beta_deg = self.beta_spin.value()
        self.project.d_beta_deg = self.d_beta_spin.value()
        self.project.known_cm_coord = (
            self.cm_coord_combo.currentText(),
            self.cm_value_spin.value()
        )
        self.project.d_known_cm = self.d_cm_spin.value()
        self.project.mass_from_forces = self.mass_mode_check.isChecked()
        self.project.g_val = self.g_spin.value()    

    
class ResultsPanel(QWidget):
    """Панель результатов расчёта."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        # ЦМ с кнопкой копировать
        cm_layout = QHBoxLayout()
        self.cm_label = QLabel("ЦМ не рассчитан")
        self.cm_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        cm_layout.addWidget(self.cm_label)
        cm_layout.addStretch()
        
        self.copy_cm_btn = QPushButton("📋 Копировать ЦМ")
        self.copy_cm_btn.setFixedWidth(140)
        self.copy_cm_btn.clicked.connect(self._copy_cm)
        cm_layout.addWidget(self.copy_cm_btn)
        layout.addLayout(cm_layout)
        
        # Протокол
        layout.addWidget(QLabel("Протокол расчёта:"))
        self.protocol_text = QTextEdit()
        self.protocol_text.setReadOnly(True)
        self.protocol_text.setStyleSheet("font-family: monospace; font-size: 12px;")
        self.protocol_text.setMinimumHeight(300)
        layout.addWidget(self.protocol_text)
        
        # Кнопки копирования
        copy_layout = QHBoxLayout()
        
        self.copy_all_btn = QPushButton("📋 Копировать протокол (TSV)")
        self.copy_all_btn.clicked.connect(self._copy_all_tsv)
        copy_layout.addWidget(self.copy_all_btn)
        
        copy_layout.addStretch()
        layout.addLayout(copy_layout)
    
    def _copy_cm(self):
        """Копирует строку ЦМ в буфер обмена."""
        text = self.cm_label.text()
        if "X =" in text:
            parts = text.replace("ЦМ в БСК:", "").strip()
            QApplication.clipboard().setText(parts)
    
    def _copy_all_tsv(self):
        """Копирует протокол в TSV-формате."""
        if not hasattr(self, 'project') or not self.project.results:
            return
        r = self.project.results
        
        lines = []
        
        # ЦМ
        lines.append("ЦМ в БСК")
        lines.append(f"X, м\t{r['x_c']:.4f}")
        lines.append(f"Y, м\t{r['y_c']:.4f}")
        lines.append(f"Z, м\t{r['z_c']:.4f}")
        lines.append(f"Масса, кг\t{r['mass_used']:.4f}")
        if 'uncertainty' in r:
            u = r['uncertainty']
            lines.append(f"НСП X, м\t{u['theta_x']:.4e}")
            lines.append(f"НСП Y, м\t{u['theta_y']:.4e}")
            lines.append(f"НСП Z, м\t{u['theta_z']:.4e}")
            lines.append(f"НСП массы, кг\t{u['theta_mass']:.4e}")
        lines.append("")
        
        # Реакции
        lines.append("Реакции опор и T_max")
        lines.append("Опора\tFx, Н\tFy, Н\tFz, Н\tTx, Н·м\tTy, Н·м\tTz, Н·м\tTmax_x\tTmax_y\tTmax_z")
        for i, s in enumerate(r['supports']):
            row = s['name']
            for key in ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']:
                val = s[key]
                row += f"\t{val:.4f}" if val is not None else "\t—"
            if 'friction_info' in r and i < len(r['friction_info']):
                for axis in ['x', 'y', 'z']:
                    tmax = r['friction_info'][i][axis]['T_max']
                    row += f"\t{tmax:.4f}" if tmax is not None else "\t—"
            else:
                row += "\t—\t—\t—"
            lines.append(row)
        lines.append("")
        
        # Суммы
        lines.append("Суммы в ГСК")
        lines.append("\tX\tY\tZ")
        lines.append(f"ΣF, Н\t{r['sum_Fx']:+.4f}\t{r['sum_Fy']:+.4f}\t{r['sum_Fz']:+.4f}")
        lines.append(f"ΣM, Н·м\t{r['sum_Mx']:+.4f}\t{r['sum_My']:+.4f}\t{r['sum_Mz']:+.4f}")
        lines.append(f"Mg сила, Н\t{r['mg_Fx']:+.4f}\t{r['mg_Fy']:+.4f}\t{r['mg_Fz']:+.4f}")
        lines.append(f"Mg момент, Н·м\t{r['mg_Mx']:+.4f}\t{r['mg_My']:+.4f}\t{r['mg_Mz']:+.4f}")
        lines.append(f"ΔF, Н\t{r['res_Fx']:+.4e}\t{r['res_Fy']:+.4e}\t{r['res_Fz']:+.4e}")
        lines.append(f"ΔM, Н·м\t{r['res_Mx']:+.4e}\t{r['res_My']:+.4e}\t{r['res_Mz']:+.4e}")
        lines.append("")
        
        # НСП
        if 'uncertainty' in r:
            u = r['uncertainty']
            lines.append("НСП")
            lines.append("Параметр\tДопуск (±)\t∂Xc/∂p\tΔXc\t∂Yc/∂p\tΔYc\t∂Zc/∂p\tΔZc")
            for c in u['contributions']:
                lines.append(
                    f"{c['param']}\t{c['theta_i']:.4e}\t"
                    f"{c['df_dx']:.4e}\t{c['contrib_x']:.4e}\t"
                    f"{c['df_dy']:.4e}\t{c['contrib_y']:.4e}\t"
                    f"{c['df_dz']:.4e}\t{c['contrib_z']:.4e}"
                )
            lines.append(f"\nk = {u['k']}")
            lines.append(f"θ_x = {u['theta_x']:.4e} м")
            lines.append(f"θ_y = {u['theta_y']:.4e} м")
            lines.append(f"θ_z = {u['theta_z']:.4e} м")
        
        QApplication.clipboard().setText("\n".join(lines))
    
    def show_results(self, project: Project):
        """Отображает результаты из project.results."""
        if project.results is None:
            self.cm_label.setText("Нет результатов")
            self.protocol_text.clear()
            return
        
        self.project = project
        r = project.results
        
        self.cm_label.setText(
            f"ЦМ в БСК: X = {r['x_c']:.4f} м, Y = {r['y_c']:.4f} м, Z = {r['z_c']:.4f} м"
        )
        
        html_parts = ['<html><body style="font-family: monospace; font-size: 12px;">']
        
        # === Блок 0: предупреждение ===
        if r.get('underdetermined'):
            html_parts.append('<p><b>⚠ Система недоопределена.</b><br>')
            html_parts.append(f'Неизвестных: {r["n_unknowns"]}, уравнений: {r["n_equations"]}.<br>')
            html_parts.append('Найдено решение с минимальной нормой.</p>')
        
        # === Блок 1: ЦМ ===
        html_parts.append('<h3>Центр масс в БСК</h3>')
        html_parts.append('<table border="1" cellpadding="3" cellspacing="0">')
        html_parts.append('<tr><th></th><th>X, м</th><th>Y, м</th><th>Z, м</th><th>Масса, кг</th></tr>')
        html_parts.append(f'<tr><td><b>Значение</b></td><td>{r["x_c"]:.4f}</td><td>{r["y_c"]:.4f}</td><td>{r["z_c"]:.4f}</td><td>{r["mass_used"]:.4f}</td></tr>')
        
        if 'uncertainty' in r:
            u = r['uncertainty']
            html_parts.append(f'<tr><td><b>НСП (±)</b></td><td>{u["theta_x"]:.4e}</td><td>{u["theta_y"]:.4e}</td><td>{u["theta_z"]:.4e}</td><td>{u["theta_mass"]:.4e}</td></tr>')
        
        html_parts.append('</table>')
        
        # === Блок 2: предупреждения о явных допусках ===
        explicit_warnings = []
        for s_data in self.project.supports:
            for axis, dt_attr, f_attr in [('x', 'dTx', 'fx'), ('y', 'dTy', 'fy'), ('z', 'dTz', 'fz')]:
                dt = getattr(s_data, dt_attr)
                f = getattr(s_data, f_attr)
                if dt != 0.0 and f is not None:
                    explicit_warnings.append(
                        f'{s_data.name}: допуск момента M{axis} задан явно (±{dt:.4f} Н·м), '
                        f'трение (f{axis}={f}) не учитывается'
                    )
        if explicit_warnings:
            html_parts.append('<p><b>⚠ Допуски моментов заданы явно — трение проигнорировано:</b><br>')
            for w in explicit_warnings:
                html_parts.append(f'{w}<br>')
            html_parts.append('</p>')
        
        # === Блок 3: Реакции ===
        html_parts.append('<h3>Реакции опор и предельные моменты трения</h3>')
        html_parts.append('<table border="1" cellpadding="3" cellspacing="0">')
        html_parts.append('<tr><th>Опора</th><th>Fx, Н</th><th>Fy, Н</th><th>Fz, Н</th>'
                          '<th>Tx, Н·м</th><th>Ty, Н·м</th><th>Tz, Н·м</th>'
                          '<th>Tmax_x, Н·м</th><th>Tmax_y, Н·м</th><th>Tmax_z, Н·м</th></tr>')
        
        supports_res = r['supports']
        for i, s in enumerate(supports_res):
            html_parts.append('<tr>')
            html_parts.append(f'<td>{s["name"]}</td>')
            for key in ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']:
                val = s[key]
                html_parts.append(f'<td>{val:.4f}</td>' if val is not None else '<td>—</td>')
            
            if 'friction_info' in r and i < len(r['friction_info']):
                fi = r['friction_info'][i]
                for axis in ['x', 'y', 'z']:
                    tmax = fi[axis]['T_max']
                    html_parts.append(f'<td>{tmax:.4f}</td>' if tmax is not None else '<td>—</td>')
            else:
                html_parts.append('<td>—</td><td>—</td><td>—</td>')
            html_parts.append('</tr>')
        
        html_parts.append('</table>')
        
        # === Блок 4: Суммы, Mg, невязки ===
        html_parts.append('<h3>Суммы и невязки в ГСК</h3>')
        html_parts.append('<table border="1" cellpadding="3" cellspacing="0">')
        html_parts.append('<tr><th></th><th>X</th><th>Y</th><th>Z</th></tr>')
        html_parts.append(f'<tr><td>ΣF (реакции), Н</td><td>{r["sum_Fx"]:+.4f}</td><td>{r["sum_Fy"]:+.4f}</td><td>{r["sum_Fz"]:+.4f}</td></tr>')
        html_parts.append(f'<tr><td>ΣM (реакции), Н·м</td><td>{r["sum_Mx"]:+.4f}</td><td>{r["sum_My"]:+.4f}</td><td>{r["sum_Mz"]:+.4f}</td></tr>')
        html_parts.append(f'<tr><td>Mg (сила), Н</td><td>{r["mg_Fx"]:+.4f}</td><td>{r["mg_Fy"]:+.4f}</td><td>{r["mg_Fz"]:+.4f}</td></tr>')
        html_parts.append(f'<tr><td>Mg (момент), Н·м</td><td>{r["mg_Mx"]:+.4f}</td><td>{r["mg_My"]:+.4f}</td><td>{r["mg_Mz"]:+.4f}</td></tr>')
        
        mg = abs(r['mg'])
        threshold = mg * 0.00001
        
        def fmt_res(val):
            if abs(val) > threshold:
                return f'<span style="color:red">{val:+.4e}</span>'
            return f'{val:+.4e}'
        
        res_F_big = any(abs(r[k]) > threshold for k in ['res_Fx', 'res_Fy', 'res_Fz'])
        res_M_big = any(abs(r[k]) > threshold for k in ['res_Mx', 'res_My', 'res_Mz'])
        
        if res_F_big or res_M_big:
            html_parts.append('<tr><td colspan="4" style="color:red"><b>⚠ Значительные невязки!</b></td></tr>')
        
        html_parts.append(f'<tr><td>ΔF (невязка сил), Н</td>'
                          f'<td>{fmt_res(r["res_Fx"])}</td>'
                          f'<td>{fmt_res(r["res_Fy"])}</td>'
                          f'<td>{fmt_res(r["res_Fz"])}</td></tr>')
        html_parts.append(f'<tr><td>ΔM (невязка моментов), Н·м</td>'
                          f'<td>{fmt_res(r["res_Mx"])}</td>'
                          f'<td>{fmt_res(r["res_My"])}</td>'
                          f'<td>{fmt_res(r["res_Mz"])}</td></tr>')
        html_parts.append('</table>')
        
        html_parts.append(f'<p>Mg = {r["mg"]:.4f} Н (порог невязки: {threshold:.4e} Н)</p>')
        
        # === Блок 5: НСП ===
        if 'uncertainty' in r:
            u = r['uncertainty']
            html_parts.append('<h3>Неисключённая систематическая погрешность (НСП)</h3>')
            html_parts.append('<table border="1" cellpadding="3" cellspacing="0">')
            html_parts.append('<tr><th>Параметр</th><th>Допуск (±)</th>'
                              '<th>∂Xc/∂p</th><th>ΔXc</th>'
                              '<th>∂Yc/∂p</th><th>ΔYc</th>'
                              '<th>∂Zc/∂p</th><th>ΔZc</th></tr>')
            
            for c in u['contributions']:
                html_parts.append(
                    f'<tr>'
                    f'<td>{c["param"]}</td>'
                    f'<td>{c["theta_i"]:.4e}</td>'
                    f'<td>{c["df_dx"]:.4e}</td>'
                    f'<td>{c["contrib_x"]:.4e}</td>'
                    f'<td>{c["df_dy"]:.4e}</td>'
                    f'<td>{c["contrib_y"]:.4e}</td>'
                    f'<td>{c["df_dz"]:.4e}</td>'
                    f'<td>{c["contrib_z"]:.4e}</td>'
                    f'</tr>'
                )
            html_parts.append('</table>')
            html_parts.append(f'<p>Коэффициент k = {u["k"]}<br>')
            html_parts.append(f'θ_x (НСП Xc) = {u["theta_x"]:.4e} м<br>')
            html_parts.append(f'θ_y (НСП Yc) = {u["theta_y"]:.4e} м<br>')
            html_parts.append(f'θ_z (НСП Zc) = {u["theta_z"]:.4e} м</p>')
        
        if 'uncertainty_error' in r:
            html_parts.append(f'<p style="color:red">⚠ Ошибка расчёта НСП: {r["uncertainty_error"]}</p>')
        
        html_parts.append('</body></html>')
        self.protocol_text.setHtml("".join(html_parts))

class MainWindow(QMainWindow):
    """Главное окно приложения."""
    
    def __init__(self):
        super().__init__()
        self.project = Project()
        self._setup_ui()
        self._setup_menu()
        self.setWindowTitle("KCM — Расчёт центра масс")
        self.resize(1200, 800)
    
    def _setup_ui(self):
        """Создаёт вкладки."""
        self.tabs = QTabWidget()
        
        # --- Вкладка "Данные" ---
        self.params_panel = ParamsPanel(self.project)
        self.tabs.addTab(self.params_panel, "Данные")
        
        # --- Вкладка "3D-вид" ---
        self.canvas = MplCanvas()
        
        self.info_panel = QWidget()
        info_layout = QVBoxLayout(self.info_panel)
        
        # Правило знаков
        self.sign_label = QLabel(
            "Правило знаков:\n"
            "α (+): нос вниз (тангаж)\n"
            "β (+): правый борт вниз (крен)\n\n"
            "БСК: X — продольная, Y — поперечная, Z — вверх\n"
            "ГСК: X — восток, Y — север, Z — зенит"
        )
        self.sign_label.setStyleSheet("font-size: 11px; padding: 5px;")
        info_layout.addWidget(self.sign_label)
        
        # Разделитель
        line = QLabel("—" * 30)
        line.setStyleSheet("color: gray;")
        info_layout.addWidget(line)
        
        # Масштаб модели
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Масштаб модели:"))
        self.stl_scale_spin = QDoubleSpinBox()
        self.stl_scale_spin.setRange(0.0001, 100.0)
        self.stl_scale_spin.setDecimals(4)
        self.stl_scale_spin.setSingleStep(0.001)
        self.stl_scale_spin.setValue(self.project.stl_scale)
        self.stl_scale_spin.setFixedWidth(100)
        self.stl_scale_spin.valueChanged.connect(self._on_stl_scale_changed)
        scale_layout.addWidget(self.stl_scale_spin)
        info_layout.addLayout(scale_layout)
        
        # Прозрачность модели
        transp_layout = QHBoxLayout()
        transp_layout.addWidget(QLabel("Прозрачность:"))
        self.stl_alpha_spin = QDoubleSpinBox()
        self.stl_alpha_spin.setRange(0.0, 1.0)
        self.stl_alpha_spin.setDecimals(2)
        self.stl_alpha_spin.setSingleStep(0.1)
        self.stl_alpha_spin.setValue(0.3)
        self.stl_alpha_spin.setFixedWidth(100)
        self.stl_alpha_spin.valueChanged.connect(self._on_stl_param_changed)
        transp_layout.addWidget(self.stl_alpha_spin)
        info_layout.addLayout(transp_layout)

        # Легенда (картинка)
        self.legend_label = QLabel()
        legend_path = resource_path("legend.png")
        if os.path.exists(legend_path):
            from PySide6.QtGui import QPixmap
            pixmap = QPixmap(legend_path)
            self.legend_label.setPixmap(pixmap.scaledToWidth(420, Qt.SmoothTransformation))
        else:
            # Текстовая легенда, если картинки нет
            self.legend_label.setText(
                "Цвета сил:\n"
                "▬ X — красный\n"
                "▬ Y — зелёный\n"
                "▬ Z — синий\n"
                "▬ mg — голубой\n\n"
                "Оси:\n"
                "▬ БСК — цветные\n"
                "▬ ГСК — серые"
            )
            self.legend_label.setStyleSheet("font-size: 11px; padding: 5px;")
        info_layout.addWidget(self.legend_label)
        
        info_layout.addStretch()
        
        # Результаты ЦМ
        self.cm_short_label = QLabel("ЦМ: не рассчитан")
        self.cm_short_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px;")
        info_layout.addWidget(self.cm_short_label)
        
        splitter_3d = QSplitter(Qt.Horizontal)
        splitter_3d.addWidget(self.canvas)
        splitter_3d.addWidget(self.info_panel)
        splitter_3d.setSizes([700, 300])
        
        self.tabs.addTab(splitter_3d, "3D-вид")
        
        # --- Вкладка "Результаты" ---
        self.results_panel = ResultsPanel()
        self.tabs.addTab(self.results_panel, "Результаты")
        
        # --- Кнопка "Рассчитать" внизу ---
        calc_btn = QPushButton("⟳ Рассчитать")
        calc_btn.setStyleSheet("font-size: 16px; padding: 10px;")
        calc_btn.clicked.connect(self._on_calculate)
        
        # --- Компоновка ---
        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        layout.addWidget(calc_btn)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
    
    def _setup_menu(self):
        """Создаёт меню."""
        menubar = self.menuBar()
        
        # Файл
        file_menu = menubar.addMenu("Файл")
        
        new_action = QAction("Создать", self)
        new_action.triggered.connect(self._on_new)
        file_menu.addAction(new_action)
        
        open_action = QAction("Открыть...", self)
        open_action.triggered.connect(self._on_open)
        file_menu.addAction(open_action)
        
        save_action = QAction("Сохранить", self)
        save_action.triggered.connect(self._on_save)
        file_menu.addAction(save_action)

        save_as_action = QAction("Сохранить как...", self)
        save_as_action.triggered.connect(self._on_save_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        load_stl_action = QAction("Загрузить модель...", self)
        load_stl_action.triggered.connect(self._on_load_stl)
        file_menu.addAction(load_stl_action)
        
        remove_stl_action = QAction("Удалить модель", self)
        remove_stl_action.triggered.connect(self._on_remove_stl)
        file_menu.addAction(remove_stl_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Выход", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Справка
        help_menu = menubar.addMenu("Справка")
        about_action = QAction("О программе", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)
    
    def _on_new(self):
        self.project = Project()
        self.params_panel.project = self.project
        self.params_panel._refresh_table()
        self.results_panel.show_results(self.project)
        self.canvas.ax.clear()
        self.canvas.draw()
    
    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Открыть проект", "", "JSON (*.json)"
        )
        if path:
            try:
                self.project = Project.load(path)
                self.params_panel.project = self.project
                self.params_panel._refresh_table()
                
                # Обновляем поля g, углов, ЦМ
                self.params_panel.g_spin.setValue(self.project.g_val)
                self.params_panel.alpha_spin.setValue(self.project.alpha_deg)
                self.params_panel.beta_spin.setValue(self.project.beta_deg)
                self.params_panel.mass_spin.setValue(self.project.mass)
                self.params_panel.d_mass_spin.setValue(self.project.d_mass)
                self.params_panel.d_alpha_spin.setValue(self.project.d_alpha_deg)
                self.params_panel.d_beta_spin.setValue(self.project.d_beta_deg)
                self.params_panel.cm_value_spin.setValue(self.project.known_cm_coord[1])
                self.params_panel.d_cm_spin.setValue(self.project.d_known_cm)
                self.params_panel.mass_mode_check.setChecked(self.project.mass_from_forces)
                
                # Город для g
                from config import G_CITIES
                for city, val in G_CITIES.items():
                    if abs(val - self.project.g_val) < 1e-6:
                        self.params_panel.g_city_combo.setCurrentText(city)
                        break
                else:
                    self.params_panel.g_city_combo.setCurrentText("Другое")

                self.stl_scale_spin.setValue(self.project.stl_scale)
                
                self.results_panel.show_results(self.project)
                self._update_3d()
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось открыть файл:\n{e}")        
    
    def _on_save(self):
        """Сохраняет проект. Если файл не выбран — запрашивает."""
        if hasattr(self, '_current_file') and self._current_file:
            self._save_to_file(self._current_file)
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, "Сохранить проект", "project.json", "JSON (*.json)"
            )
            if path:
                self._save_to_file(path)

    def _on_save_as(self):
        """Сохраняет проект в новый файл."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить проект как", "project.json", "JSON (*.json)"
        )
        if path:
            self._save_to_file(path)                
    
    def _save_to_file(self, path):
        try:
            self.params_panel.save_to_project()
            self.project.save(path)
            self._current_file = path
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить:\n{e}")
    
    def _on_load_stl(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Загрузить STL-модель", "", "STL (*.stl)"
        )
        if path:
            self.project.stl_path = path
            self._update_3d()
    
    def _on_remove_stl(self):
        self.project.stl_path = None
        self._update_3d()
    
    def _on_calculate(self):
        """Запускает расчёт."""
        try:
            self.params_panel.save_to_project()
            self.project.calculate()
            self.results_panel.show_results(self.project)
            self._update_3d()
            self.tabs.setCurrentWidget(self.results_panel)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка расчёта", str(e))
    
    def _update_3d(self):
        self.canvas.ax.clear()
        self.canvas.ax.set_xlabel('')
        self.canvas.ax.set_ylabel('')
        self.canvas.ax.set_zlabel('')
        
        if self.project.results is not None or self.project.stl_path:
            from visualize import plot_on_axes
            plot_on_axes(self.canvas.ax, self.project, 
                        stl_alpha=self.stl_alpha_spin.value())
            
            if self.project.results:
                r = self.project.results
                self.cm_short_label.setText(
                    f"ЦМ в БСК:\nX = {r['x_c']:.4f} м\nY = {r['y_c']:.4f} м\nZ = {r['z_c']:.4f} м"
                )
        else:
            self.cm_short_label.setText("ЦМ: не рассчитан")
        
        self.canvas.draw()
    
    def _on_about(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("О программе")
        msg.setTextFormat(Qt.RichText)
        msg.setText(
            "KCM — Расчёт центра масс твёрдого тела<br>"
            "Версия 1.0<br><br>"
            "По вопросам и предложениям:<br>"
            "<a href='https://github.com/ser-a11y/KCM/issues'>GitHub Issues</a>"
        )
        msg.setIcon(QMessageBox.Information)
        msg.exec()
        
    def _on_stl_scale_changed(self):
        self.project.stl_scale = self.stl_scale_spin.value()
        self._update_3d()
    
    def _on_stl_param_changed(self):
        self._update_3d()        

def run():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
