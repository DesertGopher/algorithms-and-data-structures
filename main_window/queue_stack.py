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
    def __init__(self):
        self.queue = []

    def enqueue(self, item):
        self.queue.append(item)

    def dequeue(self):
        if not self.is_empty():
            return self.queue.pop(0)
        return None

    def is_empty(self):
        return len(self.queue) == 0

    def size(self):
        return len(self.queue)


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
