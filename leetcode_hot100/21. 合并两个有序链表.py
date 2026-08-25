# https://leetcode.cn/problems/merge-two-sorted-lists/description/?envType=problem-list-v2&envId=2cktkvj
# 将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 
# 示例 1：
# 输入：l1 = [1,2,4], l2 = [1,3,4]
# 输出：[1,1,2,3,4,4]
# 示例 2：
# 输入：l1 = [], l2 = []
# 输出：[]
# 示例 3：
# 输入：l1 = [], l2 = [0]
# 输出：[0]

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

from typing import Optional

# 思路：使用迭代法来解决
# 1. 创建一个新链表，用于存储合并后的链表
# 2. 使用两个指针，分别指向两个链表的头节点
# 3. 比较两个指针所指向的节点的值，将较小的节点插入到新链表中
# 4. 移动指针，继续比较两个指针所指向的节点的值
# 5. 如果其中一个链表为空，则将另一个链表插入到新链表中
# 6. 返回新链表的头节点

class Solution:
    # 迭代法
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        new_listnode = ListNode(-1)
        
        pre_node=new_listnode
        while list1 and list2:
            if list1.val <= list2.val:
                pre_node.next = list1
                list1= list1.next
            else:
                pre_node.next = list2
                list2= list2.next
            pre_node=pre_node.next
        
        if list1 is not None:
            pre_node.next=list1
        else:
            pre_node.next=list2
        return new_listnode.next
                       