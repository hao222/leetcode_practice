from __future__ import annotations

from typing import List
import sys

"""
数组： 存放在连续内存空间上的相同类型数据的集合 通过下标索引的方式获取到下标对应的数据

"""

def search(nums, target: int) -> int:
    """二分查找"""
    left, right = 0, len(nums) - 1  # 定义target在左闭右闭的区间里，[left, right]

    while left <= right:
        middle = left + (right - left) // 2

        if nums[middle] > target:
            right = middle - 1  # target在左区间，所以[left, middle - 1]
        elif nums[middle] < target:
            left = middle + 1  # target在右区间，所以[middle + 1, right]
        else:
            return middle  # 数组中找到目标值，直接返回下标
    return -1


def removeElement(nums, val: int) -> int:
    """给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，并返回移除后数组的新长度。"""
    # 快慢指针  移除目标元素并返回新数组长度
    fast = 0  # 快指针
    slow = 0  # 慢指针
    size = len(nums)
    while fast < size:  # 不加等于是因为，a = size 时，nums[a] 会越界
        # slow 用来收集不等于 val 的值，如果 fast 对应值不等于 val，则把它与 slow 替换
        if nums[fast] != val:
            nums[slow] = nums[fast]
            slow += 1
        fast += 1
    return slow


def sortedSquares(nums: List[int]) -> List[int]:
    """给你一个按 非递减顺序 排序的整数数组 nums，返回 每个数字的平方 组成的新数组，要求也按 非递减顺序 排序"""
    # 1 暴力排序
    # return sorted(x * x for x in nums)
    # 2 双指针法
    l, r, i = 0, len(nums) - 1, len(nums) - 1
    res = [float('inf')] * len(nums)  # 需要提前定义列表，存放结果
    while l <= r:
        if nums[l] ** 2 < nums[r] ** 2:  # 左右边界进行对比，找出最大值
            res[i] = nums[r] ** 2
            r -= 1  # 右指针往左移动
        else:
            res[i] = nums[l] ** 2
            l += 1  # 左指针往右移动

        i -= 1  # 存放结果的指针需要往前平移一位
    return res


def minSubArrayLen(s: int, nums: List[int]) -> int:
    """给定一个含有 n 个正整数的数组和一个正整数 s ，找出该数组中满足其和 ≥ s 的长度最小的 连续 子数组，并返回其长度。如果不存在符合条件的子数组，返回 0。"""
    # 滑动窗口 7 [2, 5,7,3,4]
    l = len(nums)
    left = 0
    right = 0
    min_len = float('inf')
    cur_sum = 0  # 当前的累加值

    while right < l:
        cur_sum += nums[right]

        # 窗口起始位置如何移动： 如果当前窗口的值>=s， 窗口就要向前移动了（缩小）
        while cur_sum >= s:  # 当前累加值大于目标值
            min_len = min(min_len, right - left + 1)
            cur_sum -= nums[left]
            left += 1
        # 窗口结束位置： 结束位置就是遍历数组的指针
        right += 1

    return min_len if min_len != float('inf') else 0


def generateMatrix(n):
    """螺旋矩阵II"""
    if n <= 0:
        return []

    # 初始化 n x n 矩阵
    matrix = [[0] * n for _ in range(n)]

    # 初始化边界和起始值
    top, bottom, left, right = 0, n - 1, 0, n - 1
    num = 1

    while top <= bottom and left <= right:
        # 从左到右填充上边界
        for i in range(left, right + 1):
            matrix[top][i] = num
            num += 1
        top += 1

        # 从上到下填充右边界
        for i in range(top, bottom + 1):
            matrix[i][right] = num
            num += 1
        right -= 1

        # 从右到左填充下边界
        for i in range(right, left - 1, -1):
            matrix[bottom][i] = num
            num += 1
        bottom -= 1

        # 从下到上填充左边界
        for i in range(bottom, top - 1, -1):
            matrix[i][left] = num
            num += 1
        left += 1

    return matrix



def compare_sum():
    """
    # 每个指定区间内元素的总和  前缀和
    # arr [1, 2, 3, 4, 5]   p: [1, 3, 6, 10, 15]
    # 求指定区间内元素的总和
    :return:
    """
    input = sys.stdin.read()  # ctrl + d 终止输入

    data = input.split()
    index = 0
    n = int(data[index]) # 5
    index += 1
    vec = []      # [1, 2, 3, 4, 5]
    for i in range(n):
        vec.append(int(data[index + i]))
    index += n   # 6

    p = [0] * n   # 前缀和   [1, 3, 6, 10, 15]
    presum = 0   # 总和   15
    for i in range(n):
        presum += vec[i]
        p[i] = presum
    results = []

    while index < len(data):
        a = int(data[index])
        b = int(data[index + 1])
        index += 2

        if a == 0:
            sum_value = p[b]
        else:
            sum_value = p[b] - p[a - 1]

        results.append(sum_value)

    for result in results:
        print(result)


# A B 开发商购买土地
def main():
    import sys
    input = sys.stdin.read
    data = input().split()

    idx = 0
    n = int(data[idx])    # n 3
    idx += 1
    m = int(data[idx])     # m 3
    idx += 1
    sum = 0     # 总和
    vec = []  # 组成 n*m的二维数组 比如 3*3 [[1,2,3],[2,1,3],[1,2,3]]
    for i in range(n):
        row = []
        for j in range(m):
            num = int(data[idx])
            idx += 1
            row.append(num)
            sum += num
        vec.append(row)

    # 统计横向 行总计
    horizontal = [0] * n
    for i in range(n):
        for j in range(m):
            horizontal[i] += vec[i][j]

    # 统计纵向  列总计
    vertical = [0] * m
    for j in range(m):
        for i in range(n):
            vertical[j] += vec[i][j]

    result = float('inf')
    horizontalCut = 0
    for i in range(n):
        horizontalCut += horizontal[i]
        result = min(result, abs(sum - horizontalCut - horizontalCut))

    verticalCut = 0
    for j in range(m):
        verticalCut += vertical[j]
        result = min(result, abs(sum - verticalCut - verticalCut))

    print(result)





if __name__ == '__main__':
    compare_sum()
