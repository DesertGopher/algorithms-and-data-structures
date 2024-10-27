from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel,
                             QPushButton, QGraphicsView,
                             QGraphicsScene, QLineEdit, QMessageBox)
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPointF
import random


class AVLNode:
    def __init__(self, key):
        self.key = key
        self.height = 1
        self.left = None
        self.right = None


class AVLTree:
    def __init__(self):
        self.root = None
        self.rotation_count = 0

    def get_height(self, node):
        return node.height if node else 0

    def update_height(self, node):
        if node:
            node.height = max(self.get_height(node.left), self.get_height(node.right)) + 1

    def get_balance(self, node):
        return self.get_height(node.left) - self.get_height(node.right) if node else 0

    def left_rotate(self, z):
        y = z.right
        T2 = y.left
        y.left = z
        z.right = T2
        self.update_height(z)
        self.update_height(y)
        self.rotation_count += 1
        return y

    def right_rotate(self, z):
        y = z.left
        T3 = y.right
        y.right = z
        z.left = T3
        self.update_height(z)
        self.update_height(y)
        self.rotation_count += 1
        return y

    def balance(self, node):
        self.update_height(node)
        balance = self.get_balance(node)

        if balance > 1 and self.get_balance(node.left) >= 0:
            return self.right_rotate(node)

        if balance > 1 and self.get_balance(node.left) < 0:
            node.left = self.left_rotate(node.left)
            return self.right_rotate(node)

        if balance < -1 and self.get_balance(node.right) <= 0:
            return self.left_rotate(node)

        if balance < -1 and self.get_balance(node.right) > 0:
            node.right = self.right_rotate(node.right)
            return self.left_rotate(node)

        return node

    def insert(self, node, key):
        if not node:
            return AVLNode(key)

        if key < node.key:
            node.left = self.insert(node.left, key)
        elif key > node.key:
            node.right = self.insert(node.right, key)
        else:
            return node

        return self.balance(node)

    def insert_key(self, key):
        self.root = self.insert(self.root, key)

    def find_min(self, node):
        current = node
        while current.left is not None:
            current = current.left
        return current

    def delete(self, node, key):
        if not node:
            return node

        if key < node.key:
            node.left = self.delete(node.left, key)
        elif key > node.key:
            node.right = self.delete(node.right, key)
        else:
            if not node.left:
                return node.right
            elif not node.right:
                return node.left

            temp = self.find_min(node.right)
            node.key = temp.key
            node.right = self.delete(node.right, temp.key)

        return self.balance(node)

    def delete_key(self, key):
        self.root = self.delete(self.root, key)


class TreesTabWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.avl_tree = AVLTree()
        for _ in range(20):
            self.avl_tree.insert_key(random.randint(1, 100))

        self.init_ui()
        self.update_tree_view()

    def init_ui(self):
        layout = QVBoxLayout()
        self.label = QLabel("Реализация AVL-дерева с балансировкой.")
        layout.addWidget(self.label)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Введите элемент для добавления или удаления")
        layout.addWidget(self.input_field)

        self.insert_button = QPushButton("Добавить элемент")
        self.insert_button.clicked.connect(self.insert_element)
        layout.addWidget(self.insert_button)

        self.delete_button = QPushButton("Удалить элемент")
        self.delete_button.clicked.connect(self.delete_element)
        layout.addWidget(self.delete_button)

        self.scene = QGraphicsScene()
        self.graphics_view = QGraphicsView(self.scene)
        layout.addWidget(self.graphics_view)

        self.setLayout(layout)

    def insert_element(self):
        try:
            key = int(self.input_field.text())
            self.avl_tree.insert_key(key)
            self.input_field.clear()
            self.update_tree_view()
        except ValueError:
            QMessageBox.warning(self, "Ошибка", "Введите целое число")

    def delete_element(self):
        try:
            key = int(self.input_field.text())
            self.avl_tree.delete_key(key)
            self.input_field.clear()
            self.update_tree_view()
        except ValueError:
            QMessageBox.warning(self, "Ошибка", "Введите целое число")

    def update_tree_view(self):
        self.scene.clear()
        if self.avl_tree.root:
            self.draw_node(self.avl_tree.root, self.graphics_view.width() // 2, 30, self.graphics_view.width() // 4)

    def draw_node(self, node, x, y, offset):
        if node:
            
            if node.left:
                self.scene.addLine(x, y, x - offset, y + 50, QPen(Qt.black))
                self.draw_node(node.left, x - offset, y + 50, offset // 2)

            if node.right:
                self.scene.addLine(x, y, x + offset, y + 50, QPen(Qt.black))
                self.draw_node(node.right, x + offset, y + 50, offset // 2)

            ellipse = self.scene.addEllipse(x - 15, y - 15, 30, 30, QPen(Qt.black), QColor(102, 178, 255))
            text = self.scene.addText(str(node.key))
            text.setDefaultTextColor(Qt.black)
            text.setPos(x - 10, y - 10)
