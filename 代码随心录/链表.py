"""
链表 通过指针串联在一起的线性结构 链表的入口节点称为链表的头结点也就是head
单链表 双链表 循环链表
"""
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# 初始化一个单链表
class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, val):
        new_node = ListNode(val)
        if self.head is None:
            self.head = new_node
            return
        last = self.head
        while last.next is not None:
            last = last.next
        last.next = new_node

    def display(self):
        cur = self.head
        while cur is not None:
            print(cur.val, end=" -> ")
            cur = cur.next
        print("None")


linked_list = LinkedList()
for value in [1, 2, 3, 4]:
    linked_list.append(value)

linked_list.display()


# 删除链表中等于给定值 val 的所有节点  统一的逻辑来移除链表的节点的解决方法： 设置一个虚拟头结点
def removeElements(head: Optional[ListNode], val: int) -> Optional[ListNode]:
    # 创建虚拟头部节点以简化删除过程
    dummy_head = ListNode(next=head)

    # 遍历列表并删除值为val的节点
    current = dummy_head
    while current.next:
        if current.next.val == val:
            current.next = current.next.next
        else:
            current = current.next

    return dummy_head.next



# 单链表  实现增删改查事件
class MyLinkedList:
    """
    单链表法
    """

    def __init__(self):
        self.dummy_head = ListNode()
        self.size = 0

    def get(self, index: int) -> int:
        if index < 0 or index >= self.size:
            return -1

        current = self.dummy_head.next
        for i in range(index):
            current = current.next

        return current.val

    def addAtHead(self, val: int) -> None:
        self.dummy_head.next = ListNode(val, self.dummy_head.next)
        self.size += 1

    def addAtTail(self, val: int) -> None:
        current = self.dummy_head
        while current.next:
            current = current.next
        current.next = ListNode(val)
        self.size += 1

    def addAtIndex(self, index: int, val: int) -> None:
        if index < 0 or index > self.size:
            return

        current = self.dummy_head
        for i in range(index):
            current = current.next
        current.next = ListNode(val, current.next)
        self.size += 1

    def deleteAtIndex(self, index: int) -> None:
        if index < 0 or index >= self.size:
            return

        current = self.dummy_head
        for i in range(index):
            current = current.next
        current.next = current.next.next
        self.size -= 1

    def reverseList(self, head: ListNode) -> ListNode:
        """
        反转一个单链表 1->2->3->4->5->NULL 输出: 5->4->3->2->1->NULL
        """
        cur = head
        pre = None
        while cur:
            temp = cur.next  # 保存一下 cur的下一个节点，因为接下来要改变cur->next
            cur.next = pre  # 反转
            # 更新pre、cur指针
            pre = cur
            cur = temp
        return pre

    def swapPairs(self, head: ListNode) -> ListNode:
        """
        给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。 [1,2,3,4] -> [2,1,4,3]
        思路：指定一个虚拟头结点0，然后取当前的下一个 以及下下一个节点
        """
        dummy_head = ListNode(next=head)
        current = dummy_head

        # 必须有cur的下一个和下下个才能交换，否则说明已经交换结束了
        while current.next and current.next.next:
            temp = current.next  # 防止节点修改 1
            temp1 = current.next.next.next  # 3

            current.next = current.next.next    # current.next  2
            current.next.next = temp            # current.next.next 1
            temp.next = temp1                   # 1 的 next 指向 3
            current = current.next.next         # 1
        return dummy_head.next


# todo 双链表 实现增删改查 反转
class ListNode2:
    def __init__(self, val=0, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next


class MyLinkedList2:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def get(self, index: int) -> int:
        if index < 0 or index >= self.size:
            return -1

        if index < self.size // 2:
            current = self.head
            for i in range(index):
                current = current.next
        else:
            current = self.tail
            for i in range(self.size - index - 1):
                current = current.prev

        return current.val

    def addAtHead(self, val: int) -> None:
        new_node = ListNode(val, None, self.head)
        if self.head:
            self.head.prev = new_node
        else:
            self.tail = new_node
        self.head = new_node
        self.size += 1

    def addAtTail(self, val: int) -> None:
        new_node = ListNode(val, self.tail, None)
        if self.tail:
            self.tail.next = new_node
        else:
            self.head = new_node
        self.tail = new_node
        self.size += 1

    def addAtIndex(self, index: int, val: int) -> None:
        if index < 0 or index > self.size:
            return

        if index == 0:
            self.addAtHead(val)
        elif index == self.size:
            self.addAtTail(val)
        else:
            if index < self.size // 2:
                current = self.head
                for i in range(index - 1):
                    current = current.next
            else:
                current = self.tail
                for i in range(self.size - index):
                    current = current.prev
            new_node = ListNode(val, current, current.next)
            current.next.prev = new_node
            current.next = new_node
            self.size += 1

    def deleteAtIndex(self, index: int) -> None:
        if index < 0 or index >= self.size:
            return

        if index == 0:
            self.head = self.head.next
            if self.head:
                self.head.prev = None
            else:
                self.tail = None
        elif index == self.size - 1:
            self.tail = self.tail.prev
            if self.tail:
                self.tail.next = None
            else:
                self.head = None
        else:
            if index < self.size // 2:
                current = self.head
                for i in range(index):
                    current = current.next
            else:
                current = self.tail
                for i in range(self.size - index - 1):
                    current = current.prev
            current.prev.next = current.next
            current.next.prev = current.prev
        self.size -= 1

    def reverseList(self):
        cur = self.head
        prev_node = None
        while cur:
            # 交换前驱节点 后继节点
            prev_node = cur.prev
            cur.prev = cur.next
            cur.next = prev_node
            cur = cur.prev  # # 移动到下一个节点（实际上是前一个节点）
        if prev_node:
            self.head = prev_node.prev

def removeNthFromEnd(head: ListNode, n: int) -> ListNode:
    """删除倒数第n个节点"""
    # 创建一个虚拟节点，并将其下一个指针设置为链表的头部
    dummy_head = ListNode(0, head)

    # 创建两个指针，慢指针和快指针，并将它们初始化为虚拟节点
    slow = fast = dummy_head

    # 快指针比慢指针快 n+1 步
    for i in range(n + 1):
        fast = fast.next

    # 移动两个指针，直到快速指针到达链表的末尾
    while fast:
        slow = slow.next
        fast = fast.next

    # 通过更新第 (n-1) 个节点的 next 指针删除第 n 个节点
    slow.next = slow.next.next

    return dummy_head.next

# A: a1->a2->c1->c2->c3  B: b1 -> b2 -> b3 -> c1 -> c2   A和B相交于c1点
def getIntersectionNode(headA: ListNode, headB: ListNode) -> ListNode:
    """
    核心逻辑： 等比例
    :param headA:
    :param headB:
    :return:
    """
    # 处理边缘情况
    if not headA or not headB:
        return None

    # 在每个链表的头部初始化两个指针
    pointerA = headA
    pointerB = headB

    # 遍历两个链表直到指针相交
    while pointerA != pointerB:
        # 将指针向前移动一个节点
        pointerA = pointerA.next if pointerA else headB
        pointerB = pointerB.next if pointerB else headA

    # 如果相交，指针将位于交点节点，如果没有交点，值为None
    return pointerA

def getIntersectionNode(headA: ListNode, headB: ListNode) -> ListNode:
    """
    题解： 两个链表同时出发，将最长的链表头与较短的那个对齐，并遍历较短的链表，相等则返回
    :param headA:
    :param headB:
    :return:
    """
    lenA, lenB = 0, 0
    cur = headA
    while cur:         # 求链表A的长度
        cur = cur.next
        lenA += 1
    cur = headB
    while cur:         # 求链表B的长度
        cur = cur.next
        lenB += 1
    curA, curB = headA, headB
    if lenA > lenB:     # 让curB为最长链表的头，lenB为其长度
        curA, curB = curB, curA
        lenA, lenB = lenB, lenA
    for _ in range(lenB - lenA):  # 让curA和curB在同一起点上（末尾位置对齐）
        curB = curB.next
    while curA:         #  遍历curA 和 curB，遇到相同则直接返回
        if curA == curB:
            return curA
        else:
            curA = curA.next
            curB = curB.next
    return None

def detectCycle(head: ListNode) -> ListNode:
    """ 环形链表II
    给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null
    :param head:
    :return:
    """
    slow = head
    fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

        # 如果有环，则slow和fast会在此相遇
        if slow == fast:
            # 将一个节点重置为头结点， 另一个是当前相遇节点， 然后步步向前，直到再次相遇就是当前的入口点
            slow = head
            while slow != fast:
                slow = slow.next
                fast = fast.next
            return slow
    # If there is no cycle, return None
    return None

