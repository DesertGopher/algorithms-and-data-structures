from pathlib import Path
from PyQt5.QtGui import QIcon, QDoubleValidator
from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QLineEdit, QPushButton, QFrame, QSpacerItem,
    QSizePolicy, QTextEdit
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import time

from .matrix_calculations import classic_multiplication, strassen_multiplication


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout(self.main_widget)
        self.tabs = QTabWidget()
        self.double_validator = QDoubleValidator()
        self.initUI()
        self.show()

    def init_stylesheet(self):
        styles_path = str(Path(__file__).resolve().parent / "styles.css")
        with open(styles_path) as f:
            file_styles = f.read()
        self.setStyleSheet(file_styles)

    def init_window(self):
        self.setWindowTitle("Программирование структур данных")
        icon_path = str(Path(__file__).resolve().parent / "icon.png")
        self.setWindowIcon(QIcon(icon_path))
        self.setFixedSize(1200, 650)

    def init_tabs(self):
        tab1 = self.create_matrix_input_tab()
        self.tabs.addTab(tab1, "1. Исследование производительности алгоритмов")
        self.tabs.addTab(QWidget(), "Лаб 2")
        self.tabs.addTab(QWidget(), "Лаб 3")
        self.tabs.addTab(QWidget(), "Лаб 4")
        self.tabs.setTabsClosable(False)

    def create_matrix_input_tab(self):
        main_tab_layout = QHBoxLayout()
        main_widget = QWidget()
        main_widget.setLayout(main_tab_layout)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self.matrix_size_fields = QVBoxLayout()
        self.add_matrix_size_input()

        add_matrix_button = QPushButton("Добавить пару матриц")
        add_matrix_button.clicked.connect(self.add_matrix_size_input)

        calculate_button = QPushButton("Рассчитать")

        left_layout.addLayout(self.matrix_size_fields)
        left_layout.addWidget(add_matrix_button)
        left_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        left_layout.addWidget(calculate_button)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        self.graph_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        right_layout.addWidget(self.graph_canvas)

        self.explanation_text_edit = QTextEdit()
        self.explanation_text_edit.setReadOnly(True)
        explanation_frame = QFrame()
        explanation_frame.setFrameShape(QFrame.StyledPanel)
        explanation_frame.setFixedHeight(200)
        explanation_layout = QVBoxLayout(explanation_frame)
        explanation_layout.addWidget(self.explanation_text_edit)

        right_layout.addWidget(self.graph_canvas)
        right_layout.addWidget(explanation_frame)

        main_tab_layout.addWidget(left_widget, 30)
        main_tab_layout.addWidget(right_widget, 70)

        calculate_button.clicked.connect(self.calculate_matrices)

        return main_widget

    def remove_matrix_size_input(self, layout):
        for i in reversed(range(layout.count())):
            widget = layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        self.matrix_size_fields.removeItem(layout)

    def add_matrix_size_input(self):
        matrix_input_layout = QHBoxLayout()
        rows_input = QLineEdit()
        rows_input.setValidator(self.double_validator)
        rows_input.setPlaceholderText("Строк")
        cols_input = QLineEdit()
        cols_input.setValidator(self.double_validator)
        cols_input.setPlaceholderText("Столбцов")

        remove_button = QPushButton("⮾")
        remove_button.setStyleSheet("background-color: red; color: white; font-size: 20px; padding: 0 5px 0 5px;")
        remove_button.clicked.connect(lambda: self.remove_matrix_size_input(matrix_input_layout))

        matrix_input_layout.addWidget(QLabel("Матрица:"))
        matrix_input_layout.addWidget(rows_input)
        matrix_input_layout.addWidget(QLabel("x"))
        matrix_input_layout.addWidget(cols_input)
        matrix_input_layout.addWidget(remove_button)

        self.matrix_size_fields.addLayout(matrix_input_layout)

    def calculate_matrices(self):
        sizes = []
        classic_times = []
        strassen_times = []

        results_text = ""

        for layout in self.matrix_size_fields.children():
            if isinstance(layout, QHBoxLayout):
                rows_input = layout.itemAt(1).widget()
                cols_input = layout.itemAt(3).widget()
                rows = int(rows_input.text() or "0")
                cols = int(cols_input.text() or "0")

                if rows == cols and rows > 0:
                    matrix = np.random.randint(0, 10, (rows, cols))

                    start_time = time.time()
                    classic_result = classic_multiplication(matrix, matrix)
                    classic_time = time.time() - start_time
                    classic_times.append(classic_time)

                    start_time = time.time()
                    strassen_result = strassen_multiplication(matrix, matrix)
                    strassen_time = time.time() - start_time
                    strassen_times.append(strassen_time)

                    sizes.append(rows)

                    results_text += (
                        f"Матрица {rows}x{cols}:\n"
                        f"Классическое умножение: {classic_time:.4f} сек\n"
                        f"Умножение Штрассена: {strassen_time:.4f} сек\n\n"
                    )
                else:
                    results_text += f"Пропуск неквадратной матрицы {rows}x{cols}\n\n"

        self.plot_graph(sizes, classic_times, strassen_times)
        self.explanation_text_edit.setText(results_text)

    def plot_graph(self, sizes, classic_times, strassen_times):
        self.graph_canvas.figure.clear()
        ax = self.graph_canvas.figure.add_subplot(111)

        ax.plot(sizes, classic_times, label="Классическое умножение", marker='o')
        ax.plot(sizes, strassen_times, label="Умножение Штрассена", marker='o')

        ax.set_xlabel("Размер матрицы")
        ax.set_ylabel("Время выполнения (сек)")
        ax.set_title("Сравнение времени выполнения алгоритмов умножения матриц")
        ax.legend()

        self.graph_canvas.draw()

    def initUI(self):
        self.init_window()
        self.init_tabs()
        self.main_layout.addWidget(self.tabs, 1)
        self.setCentralWidget(self.main_widget)

        self.init_stylesheet()
