import random
import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator, QIcon, QIntValidator
from PyQt5.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .large_miltiplication import karatsuba, classic_large_multiplication
from .matrix_calculations import (
    classic_multiplication,
    custom_strassen_multiplication,
    numpy_multiplication,
    scipy_multiplication,
    strassen_multiplication,
    sympy_multiplication,
    tensorflow_multiplication,
)
from .queue_stack import QueueArray, QueueLinkedList, StackArray, StackLinkedList
from .tree_widget import TreesTabWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout(self.main_widget)
        self.tabs = QTabWidget()
        self.current_lm_length = 10
        self.double_validator = QDoubleValidator()
        self.int_validator = QIntValidator()
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
        tab2 = self.create_data_structures_tab()
        tab3 = self.create_large_multiplication_tab()
        tab4 = TreesTabWidget()
        self.tabs.addTab(tab1, "1. Исследование производительности алгоритмов")
        self.tabs.addTab(tab2, "2. Исследование программной реализации структур данных")
        self.tabs.addTab(tab3, "3. Алгоритмы длинной арифметики")
        self.tabs.addTab(tab4, "4. Реализация и исследование деревьев")
        self.tabs.setTabsClosable(False)

    def create_large_multiplication_tab(self):
        layout = QVBoxLayout()

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedHeight(150)
        self.scroll_area_content = QWidget()
        self.scroll_area_layout = QVBoxLayout(self.scroll_area_content)

        for _ in range(3):
            self.add_length_input()

        self.scroll_area.setWidget(self.scroll_area_content)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.scroll_area)

        button_container_layout = QVBoxLayout()

        add_button = QPushButton("Добавить поле для длины числа")
        add_button.clicked.connect(self.add_length_input)
        button_container_layout.addWidget(add_button)

        calculate_button = QPushButton("Запустить расчеты")
        calculate_button.clicked.connect(self.run_calculations)
        button_container_layout.addWidget(calculate_button)

        button_layout.addLayout(button_container_layout)
        layout.addLayout(button_layout)

        self.graph_widget = QWidget()
        self.graph_layout = QVBoxLayout()
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.graph_layout.addWidget(self.canvas)
        self.graph_widget.setLayout(self.graph_layout)

        layout.addWidget(self.graph_widget)

        tab = QWidget()
        tab.setLayout(layout)
        return tab

    def add_length_input(self):
        length_input = QLineEdit()
        length_input.setValidator(self.int_validator)
        length_input.setPlaceholderText("Введите длину числа")
        length_input.setText(str(self.current_lm_length))
        self.scroll_area_layout.addWidget(length_input)
        self.current_lm_length += 40

    def run_calculations(self):
        lengths = []

        for i in range(self.scroll_area_layout.count()):
            length_input = self.scroll_area_layout.itemAt(i).widget()
            if isinstance(length_input, QLineEdit):
                text = length_input.text().strip()
                if text.isdigit() and int(text) > 0:
                    lengths.append(int(text))

        times_classic = []
        times_karatsuba = []

        for length in lengths:
            num1 = self.generate_large_number(length)
            num2 = self.generate_large_number(length)

            start_time = time.time()
            classic_large_multiplication(num1, num2)
            classic_time = time.time() - start_time
            times_classic.append(classic_time)

            start_time = time.time()
            karatsuba(num1, num2)
            karatsuba_time = time.time() - start_time
            times_karatsuba.append(karatsuba_time)

        self.plot_lm_graph(lengths, times_classic, times_karatsuba)

    @staticmethod
    def generate_large_number(length):
        return random.randint(10 ** (length - 1), 10 ** length - 1)

    def plot_lm_graph(self, lengths, times_classic, times_karatsuba):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(lengths, times_classic, label='Классический алгоритм', marker='o')
        ax.plot(lengths, times_karatsuba, label='Алгоритм Карацубы', marker='o')
        ax.set_xlabel('Длина числа')
        ax.set_ylabel('Время (сек.)')
        ax.set_title('Сравнение времени выполнения алгоритмов')
        ax.legend()
        self.canvas.draw()

    def create_data_structures_tab(self):
        tab2 = QWidget()

        layout = QHBoxLayout()

        array_layout = QVBoxLayout()
        array_form = QFormLayout()
        self.array_input = QLineEdit()
        self.array_input.setText("10000")
        array_form.addRow(QLabel("Количество элементов (массив):"), self.array_input)
        array_layout.addLayout(array_form)

        self.array_display = QLabel()
        array_layout.addWidget(self.array_display)

        linked_list_layout = QVBoxLayout()
        linked_list_form = QFormLayout()
        self.linked_list_input = QLineEdit()
        self.linked_list_input.setText("10000")
        linked_list_form.addRow(
            QLabel("Количество элементов (связный список):"), self.linked_list_input
        )
        linked_list_layout.addLayout(linked_list_form)

        self.linked_list_display = QLabel()
        linked_list_layout.addWidget(self.linked_list_display)

        layout.addLayout(array_layout)
        layout.addLayout(linked_list_layout)

        self.test_button = QPushButton("Протестировать")
        self.test_button.clicked.connect(self.run_test)

        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        self.output_area.setStyleSheet("font-size: 22px;")

        bottom_layout = QVBoxLayout()
        bottom_layout.addWidget(self.test_button)
        bottom_layout.addWidget(self.output_area)

        main_layout = QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addLayout(bottom_layout)

        tab2.setLayout(main_layout)
        return tab2

    @staticmethod
    def format_elements(elements):
        n = len(elements)
        if n <= 6:
            return " -> ".join(map(str, elements))
        else:

            return f"{elements[0]} -> {elements[1]} -> ... -> {elements[-2]} -> {elements[-1]} (Всего: {n} элементов)"

    def run_test(self):

        array_size = (
            int(self.array_input.text()) if self.array_input.text().isdigit() else 0
        )
        linked_list_size = (
            int(self.linked_list_input.text())
            if self.linked_list_input.text().isdigit()
            else 0
        )

        stack_array = StackArray()
        queue_array = QueueArray(array_size)
        array_result = self.test_stack_and_queue(stack_array, queue_array, array_size)

        stack_array_elements = [i for i in range(array_size)]
        queue_array_elements = [i for i in range(array_size)]
        array_graphic_text = (
            f"Стек (массив): {self.format_elements(stack_array_elements)}\n"
            f"Очередь (массив): {self.format_elements(queue_array_elements)}"
        )
        self.array_display.setText(array_graphic_text)

        stack_linked = StackLinkedList()
        queue_linked = QueueLinkedList()
        linked_result = self.test_stack_and_queue(
            stack_linked, queue_linked, linked_list_size
        )

        stack_linked_elements = [i for i in range(linked_list_size)]
        queue_linked_elements = [i for i in range(linked_list_size)]
        linked_list_graphic_text = (
            f"Стек (связный список): {self.format_elements(stack_linked_elements)}\n"
            f"Очередь (связный список): {self.format_elements(queue_linked_elements)}"
        )
        self.linked_list_display.setText(linked_list_graphic_text)

        result = f"Тестирование производительности для массива:\n{array_result}\n\n"
        result += (
            f"Тестирование производительности для связного списка:\n{linked_result}\n"
        )

        self.output_area.setText(result)

    @staticmethod
    def test_stack_and_queue(stack, queue, n):
        result = ""

        start_time = time.time()
        for i in range(n):
            stack.push(i)
        push_time = time.time() - start_time

        start_time = time.time()
        for i in range(n):
            stack.pop()
        pop_time = time.time() - start_time

        result += f"  Стек (push): {push_time:.6f} с\n"
        result += f"  Стек (pop): {pop_time:.6f} с\n"

        start_time = time.time()
        for i in range(n):
            queue.enqueue(i)
        enqueue_time = time.time() - start_time

        start_time = time.time()
        for i in range(n):
            queue.dequeue()
        dequeue_time = time.time() - start_time

        result += f"  Очередь (enqueue): {enqueue_time:.6f} с\n"
        result += f"  Очередь (dequeue): {dequeue_time:.6f} с\n"

        return result

    def create_matrix_input_tab(self):
        main_tab_layout = QHBoxLayout()
        main_widget = QWidget()
        main_widget.setLayout(main_tab_layout)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        label = QLabel("Ввод матриц для умножения")
        left_layout.addWidget(label)

        checkbox_group = QGroupBox("Выберите алгоритмы")
        checkbox_layout = QVBoxLayout(checkbox_group)

        self.classic_checkbox = QCheckBox("Классическое умножение")
        self.strassen_checkbox = QCheckBox("Умножение Штрассена")
        self.custom_strassen_checkbox = QCheckBox("Свой Штрассен")
        self.numpy_strassen_checkbox = QCheckBox("Умножение Numpy")
        self.scipy_checkbox = QCheckBox("Умножение Scipy")
        self.sumpy_checkbox = QCheckBox("Умножение Sumpy")
        self.tensorflow_checkbox = QCheckBox("Умножение TensorFlow")

        checkbox_layout.addWidget(self.classic_checkbox)
        checkbox_layout.addWidget(self.strassen_checkbox)
        checkbox_layout.addWidget(self.custom_strassen_checkbox)
        checkbox_layout.addWidget(self.numpy_strassen_checkbox)
        checkbox_layout.addWidget(self.scipy_checkbox)
        checkbox_layout.addWidget(self.sumpy_checkbox)
        checkbox_layout.addWidget(self.tensorflow_checkbox)

        left_layout.addWidget(checkbox_group)

        matrix_input_widget = QWidget()
        matrix_input_layout = QVBoxLayout(matrix_input_widget)
        matrix_input_layout.setAlignment(Qt.AlignTop)

        self.matrix_size_fields = QVBoxLayout()
        self.add_matrix_size_input()

        matrix_input_layout.addLayout(self.matrix_size_fields)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll_area.setWidget(matrix_input_widget)

        left_layout.addWidget(scroll_area)

        add_matrix_button = QPushButton("Добавить пару матриц")
        add_matrix_button.clicked.connect(self.add_matrix_size_input)
        left_layout.addWidget(add_matrix_button)

        calculate_button = QPushButton("Рассчитать")
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
        remove_button.setStyleSheet(
            "background-color: #DC4242; color: white; font-size: 20px; padding: 0 5px 0 5px;"
        )
        remove_button.clicked.connect(
            lambda: self.remove_matrix_size_input(matrix_input_layout)
        )

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
        numpy_times = []
        custom_strassen_times = []
        scipy_times = []
        sumpy_times = []
        tensorflow_times = []

        self.graph_canvas.figure.clear()

        results_text = ""

        for layout in self.matrix_size_fields.children():
            if isinstance(layout, QHBoxLayout):
                rows_input = layout.itemAt(1).widget()
                cols_input = layout.itemAt(3).widget()
                rows = int(rows_input.text() or "0")
                cols = int(cols_input.text() or "0")

                if rows == cols and rows > 0:
                    matrix = np.random.randint(0, 10, (rows, cols))
                    results_text += f"Матрица {rows}x{cols}:\n"

                    if self.classic_checkbox.isChecked():
                        start_time = time.time()
                        classic_multiplication(matrix, matrix)
                        classic_time = time.time() - start_time
                        classic_times.append(classic_time)
                        results_text += (
                            f"Классическое умножение: {classic_time:.4f} сек\n"
                        )

                    if self.strassen_checkbox.isChecked():
                        start_time = time.time()
                        strassen_multiplication(list(matrix), list(matrix))
                        strassen_time = time.time() - start_time
                        strassen_times.append(strassen_time)
                        results_text += (
                            f"Умножение Штрассена: {strassen_time:.4f} сек\n"
                        )

                    if self.custom_strassen_checkbox.isChecked():
                        start_time = time.time()
                        custom_strassen_multiplication(matrix, matrix)
                        custom_strassen_time = time.time() - start_time
                        custom_strassen_times.append(custom_strassen_time)
                        results_text += (
                            f"Написанный Штрассен: {custom_strassen_time:.4f} сек\n"
                        )

                    if self.numpy_strassen_checkbox.isChecked():
                        start_time = time.time()
                        numpy_multiplication(matrix, matrix)
                        numpy_time = time.time() - start_time
                        numpy_times.append(numpy_time)
                        results_text += f"Умножение Numpy: {numpy_time:.4f} сек\n"

                    if self.scipy_checkbox.isChecked():
                        start_time = time.time()
                        scipy_multiplication(matrix, matrix)
                        scipy_time = time.time() - start_time
                        scipy_times.append(scipy_time)
                        results_text += f"Умножение Scipy: {scipy_time:.4f} сек\n"

                    if self.sumpy_checkbox.isChecked():
                        start_time = time.time()
                        sympy_multiplication(matrix, matrix)
                        sumpy_time = time.time() - start_time
                        sumpy_times.append(sumpy_time)
                        results_text += f"Умножение Sumpy: {sumpy_time:.4f} сек\n"

                    if self.tensorflow_checkbox.isChecked():
                        start_time = time.time()
                        tensorflow_multiplication(matrix, matrix)
                        tensorflow_time = time.time() - start_time
                        tensorflow_times.append(tensorflow_time)
                        results_text += (
                            f"Умножение TensorFlow: {tensorflow_time:.4f} сек\n"
                        )

                    sizes.append(rows)
                    results_text += "\n"

                else:
                    results_text += f"Пропуск неквадратной матрицы {rows}x{cols}\n\n"

        self.plot_graph(
            sizes,
            classic_times,
            strassen_times,
            custom_strassen_times,
            scipy_times,
            sumpy_times,
            tensorflow_times,
            numpy_times,
        )
        self.explanation_text_edit.setText(results_text)

    def plot_graph(
            self,
            sizes,
            classic_times,
            strassen_times,
            custom_strassen_times,
            scipy_times,
            sumpy_times,
            tensorflow_times,
            numpy_times,
    ):
        ax = self.graph_canvas.figure.add_subplot(111)

        if sizes and classic_times:
            ax.plot(sizes, classic_times, label="Классическое умножение", marker="o")
        if sizes and strassen_times:
            ax.plot(sizes, strassen_times, label="Умножение Штрассена", marker="o")
        if sizes and custom_strassen_times:
            ax.plot(sizes, custom_strassen_times, label="Свой Штрассен", marker="o")
        if sizes and numpy_times:
            ax.plot(sizes, numpy_times, label="Умножение Numpy", marker="o")
        if sizes and scipy_times:
            ax.plot(sizes, scipy_times, label="Умножение Scipy", marker="o")
        if sizes and sumpy_times:
            ax.plot(sizes, sumpy_times, label="Умножение Sumpy", marker="o")
        if sizes and tensorflow_times:
            ax.plot(sizes, tensorflow_times, label="Умножение TensorFlow", marker="o")

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
