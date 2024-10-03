class StackArray:
    def __init__(self):
        self.stack = []

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        if not self.is_empty():
            return self.stack.pop()
        return None

    def is_empty(self):
        return len(self.stack) == 0

    def size(self):
        return len(self.stack)


class QueueArray:
    def __init__(self, size_limit):
        self.size_limit = size_limit
        self.queue = [None] * size_limit
        self.enqueue_index = 0
        self.dequeue_index = 0
        self.current_size = 0

    def enqueue(self, item):
        if self.current_size < self.size_limit:
            self.queue[self.enqueue_index] = item
            self.enqueue_index = (self.enqueue_index + 1) % self.size_limit
            self.current_size += 1

    def dequeue(self):
        if self.current_size > 0:
            self.queue[self.dequeue_index] = None
            self.dequeue_index = (self.dequeue_index + 1) % self.size_limit
            self.current_size -= 1

    def is_empty(self):
        return self.current_size == 0

    def size(self):
        return self.current_size


class Node:
    def __init__(self, value):
        self.value = value
        self.next = None


class StackLinkedList:
    def __init__(self):
        self.head = None
        self.size = 0

    def push(self, item):
        new_node = Node(item)
        new_node.next = self.head
        self.head = new_node
        self.size += 1

    def pop(self):
        if not self.is_empty():
            popped = self.head.value
            self.head = self.head.next
            self.size -= 1
            return popped
        return None

    def is_empty(self):
        return self.head is None

    def get_size(self):
        return self.size


class QueueLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def enqueue(self, item):
        new_node = Node(item)
        if self.tail:
            self.tail.next = new_node
        self.tail = new_node
        if not self.head:
            self.head = new_node
        self.size += 1

    def dequeue(self):
        if not self.is_empty():
            dequeued = self.head.value
            self.head = self.head.next
            if self.head is None:
                self.tail = None
            self.size -= 1
            return dequeued
        return None

    def is_empty(self):
        return self.head is None

    def get_size(self):
        return self.size
