############################# --------- 3.7 ---------###########################################

from __future__ import annotations  # 3.7

from functools import lru_cache, cached_property


class C:
    @classmethod
    def from_string(cls, source: str) -> C:
        ...

    def validate_b(self, obj: B) -> bool:
        ...


class B:
    ...


import asyncio
import contextvars

# 上下文切换 多用于多线程 协程中 ContextVar上下文变量
ctx = contextvars.ContextVar('trace')
ctx.set("begin")


async def fun():
    ctx.set(ctx.get() + "|fun")
    print("ctx:", ctx.get())


async def main():
    ctx.set(ctx.get() + "|main")
    print("befor call fun: ctx", ctx.get())
    await fun()
    print("after call fun: ctx", ctx.get())


# print("befor call main: ctx", ctx.get())
# asyncio.get_event_loop().run_until_complete(main())
# print("after call main: ctx", ctx.get())


from dataclasses import dataclass


# 数据类， 为该类自动添加一些特殊方法 init repr eq 等
@dataclass
class InventoryItem:
    """Class for keeping track of an item in inventory."""
    name: str
    unit_price: float
    quantity_on_hand: int = 0

    def total_cost(self) -> float:
        return self.unit_price * self.quantity_on_hand


# 以其他方式找不到某个模块属性时将会调用它
def __getattr__(name):
    return getattr(InventoryItem, name)


############################# --------- 3.8 ---------###########################################

# 海象运算符  表达式内部为变量赋值， 比如： 赋值表达式  正则表达式匹配， while循环计算 以及 列表推导式中
if (n := len([1, 2, 3])) >= 10:
    print("--------")


# 仅限位置形参 /   /之前的 a,b 形参为仅限位置形参， *之后的参数 e，g为关键字形参
def f(a, b, /, c, d, *, e, g):
    print(a, b, c, d, e, g)


# f'{expr=}', 输出的字符串将包含变量名称和其对应的值
p = 3.14
print(f"{p=}")
# dict 按照插入顺序反向迭代 reversed()
a_dict = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "g": 6}
for k, v in reversed(a_dict.items()):
    print(f"{k}: {v}")
# 读取第三方包数据中的源信息   distribution
import importlib.metadata

for distribution in importlib.metadata.distributions():
    print(f"{distribution.metadata['Name']} - {distribution.version}")

# 在finally中可以使用continue
for i in range(2):
    try:
        print(i)
    finally:
        print('A sentence.')
        continue
        print('This never shows.')  # noqa


@dataclass
class Q:
    name: str
    age: int

    # @property + @lru_cache 来实现cached_property类似的功能
    # @property
    # @lru_cache
    @cached_property  # noqa
    def get_name(self):
        return self.name


from typing import Literal  # Literal 只允许一个或者多个特定的值  # noqa


def get_status(port: int, status: Literal['success', 'error']) -> Literal['success', 'error']:  # noqa
    if status == 'success':
        return 'success'
    else:
        return 'error'  # noqa


############################# --------- 3.9 ---------###########################################

# 字典合并与更新运算符  合并| 更新|=
xx = {"key1": "value1 from x", "key2": "value2 from x"} | {"key2": "value2 from y", "key3": "value3 from y"}
# 移除前缀 removeprefix  移除后缀 removesuffix
ss = "www.cxxc.com".removeprefix('www')


# 标准多项集标注泛型， 可以使用内置的多项集类型， 比如 list dict queue 来代替 typing下的Dict List
def get_all(a: list[str], b: dict) -> list[str]:
    return list(set(a).intersection(b))


# 在3.10中 parser模块将被弃用

# zoneinfo iana时区数据库   基于datetime.tzinfo实现
from datetime import datetime
from zoneinfo import ZoneInfo

now_utc = datetime.now(tz=ZoneInfo("UTC"))
print(f"当前UTC时间: {now_utc}")
dt_ny = datetime.now(tz=ZoneInfo("America/New_York"))
print(f"纽约当前时间: {dt_ny}")

#  concurrent.futures.Executor.shutdown() 显式调用关闭执行， 新增cancel_futures参数，可以取消尚未开始运行的所有挂起的 Future，而不必等待它们完成运行再关闭执行器
# namedtuple 一个带字段名的元祖子类
from collections import namedtuple

Point = namedtuple("Point", ["x", "y"])
p1 = Point(1, 2)
print(p1[0], p1[1], p1.x, p1.y)

# 引入annotated诶行增加额外信息
from typing import Annotated


def process_data(data: Annotated[int, "This represents a count"]) -> None:
    print(f"Processing {data} items.")


############################# --------- 3.10 ---------###########################################

# 对类型注解增加精确，并可以对一个参数多个类型联合
def process_data(data: int | str, numbers: list[int]) -> None:
    ...


# 对错误的提示更加友好, 增加额外的错误明示

# 父类类型参数化  更加的面向对象编程
from typing import Generic, TypeVar

# 类型泛化
T = TypeVar('T')

# T可以是任意类型的变量
def get_first_element(elements: List[T]) -> T:
    if elements:
        return elements[0]
    raise ValueError("The list is empty")
class Parent(Generic[T]):  # 类可以接受任意变量T
    pass


class Child(Parent[int]):  # 明确指定继承了一个整数类型的 Parent
    pass

# 限制类型变量  只允许出现 int float类型
Number = TypeVar('Number', int, float)
def total(numbers: List[Number]) -> Number:
    return sum(numbers)

print(total([1,12,1.2]))



# 使用外层圆括号来使多个上下文管理器可以连续多行地书写

# 结构化模式匹配
def process_value(value):
    match value:
        case 0:
            print("Value is zero")
        case 1 | 2 | 3:  # 匹配到1 2 3
            print("Value is one, two, or three")
        case _ if value < 0:
            print("Value is negative")
        case _:
            print("Value is something else")


# 带有字面值和变量的模式   point is an (x, y) tuple
point = (1, 2)
match point:
    case (0, 0):
        print("Origin")
    case (0, y):
        print(f"Y={y}")
    case (x, 0):
        print(f"X={x}")
    case (x, y):
        print(f"X={x}, Y={y}")
    case _:
        raise ValueError("Not a point")


class Point:
    x: int
    y: int


def location(point):
    match point:
        case Point(x=0, y=0):
            print("Origin is the point's location.")
        case Point(x=0, y=y):
            print(f"Y={y} and the point is on the y-axis.")
        case Point(x=x, y=0):
            print(f"X={x} and the point is on the x-axis.")
        case Point():
            print("The point is located somewhere else on the plane.")
        case _:
            print("Not a point")


test_variable = ('error',)
match test_variable:
    case ('warning', code, 40):
        print("A warning has been received.")
    case ('error', code, _):
        print(f"An error {code} occurred.")

# 约束项 if 子句
match point:
    case Point(x, y) if x == y:
        print(f"The point is located on the diagonal Y=X at {x}.")
    case Point(x, y):
        print(f"Point is not on the diagonal.")

from enum import Enum


class Color(Enum):
    RED = 0
    GREEN = 1
    BLUE = 2


color = Color.GREEN
match color:
    case Color.RED:
        print("I see red!")
    case Color.GREEN:
        print("Grass is green")
    case Color.BLUE:
        print("I'm feeling the blues :(")

print(isinstance(1, int | str))

################################ --------- 3.11 -------- ######################################
# 错误提示 : 解释器现在不仅会指出错误所在行，还会进一步指出引发错误的表达式在哪里

# 类型提示可以使用Self
from typing import Self


class A:
    aa = None

    def __init__(self, a) -> Self:
        self.aa = a
        return self

    def __enter__(self) -> Self:
        self.aa += 1
        return self


# ExceptionGroup 抛出和处理多个异常 它可以同时抛出多个不同的异常

# TypeDict 可以创建具有特定键和值的字典  并且支持必填 非必填， total=False 或者
from typing import TypedDict, NotRequired, Required


class Person(TypedDict):
    name: str
    age: int
    car: str


class PersonA(Person, total=False):
    car: str


class PersonB(TypedDict):
    name: str
    age: int
    car: NotRequired[str]  # 非必填


################################ --------- 3.12 -------- ######################################

# f-string 中的表达式组件现在可以是任何有效的 Python 表达式
def hello():
    return "Hello World!"


f_string = f"hello() return {":" + hello()}"
# 更精确的错误消息提示
# 新类型语法 泛型
from typing import Iterable
def max[T](args: Iterable[T]) -> T:
    ...


class list[T]:
    def __getitem__(self, index: int, /) -> T:
        ...

    def append(self, element: T) -> None:
        ...


from typing import override

class Base:
    def get_color(self) -> str:
        return "blue"



class GoodChild(Base):
    @override  # ok: overrides Base.get_color
    def get_color(self) -> str:
        return "yellow"


class BadChild(Base):
    @override  # type checker error: does not override Base.get_color
    def get_colour(self) -> str:
        return "red"

print(BadChild().get_colour())