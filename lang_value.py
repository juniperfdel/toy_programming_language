from enum import Enum, auto
from typing import Self


class ValueTypes(Enum):
    Number = auto()
    String = auto()
    Boolean = auto()
    List = auto()
    ClassInstance = auto()

    Undefined = auto()


value_type_strs = {
    ValueTypes.Number: "Number",
    ValueTypes.String: "String",
    ValueTypes.Boolean: "Boolean",
    ValueTypes.List: "List",
    ValueTypes.ClassInstance: "ClassInstance",
    ValueTypes.Undefined: "Undefined",
}


class Value:
    type: ValueTypes = ValueTypes.Undefined
    value = None

    def add(self, other: Self):
        raise NotImplementedError(
            "Addition not implemented " + self.str_for_other_type(other)
        )

    def sub(self, other: Self):
        raise NotImplementedError(
            "Subtraction not implemented " + self.str_for_other_type(other)
        )

    def mul(self, other: Self):
        raise NotImplementedError(
            "Multiply not implemented " + self.str_for_other_type(other)
        )

    def div(self, other: Self):
        raise NotImplementedError(
            "Divide not implemented " + self.str_for_other_type(other)
        )

    def pow(self, other: Self):
        raise NotImplementedError(
            "Power not implemented " + self.str_for_other_type(other)
        )

    def lss_than(self, other: Self):
        raise NotImplementedError(
            "Less Than not implemented " + self.str_for_other_type(other)
        )

    def gtr_than(self, other: Self):
        raise NotImplementedError(
            "Greater Than not implemented " + self.str_for_other_type(other)
        )

    def and_op(self, other: Self):
        raise NotImplementedError(
            "And is not implemented " + self.str_for_other_type(other)
        )

    def or_op(self, other: Self):
        raise NotImplementedError(
            "Or is not implemented " + self.str_for_other_type(other)
        )

    def eq_op(self, other: Self):
        raise NotImplementedError(
            "Equality is not implemented " + self.str_for_other_type(other)
        )

    def neq_op(self, other: Self):
        raise NotImplementedError(
            "Not Equality is not implemented " + self.str_for_other_type(other)
        )

    def gtr_eq(self, other: Self):
        raise NotImplementedError(
            "Greater Than or Equal is not implemented " + self.str_for_other_type(other)
        )

    def lss_eq(self, other: Self):
        raise NotImplementedError(
            "Less Than or Equal is not implemented " + self.str_for_other_type(other)
        )

    def get_idx(self, other: Self):
        raise NotImplementedError(
            "Get Index is not implemented " + self.str_for_other_type(other)
        )

    def set_idx(self, key: Self, other: Self):
        raise NotImplementedError(
            "Set Index is not implemented " + self.str_for_other_type(other)
        )

    def str_for_other_type(self, other: Self):
        return f"for {self} and {other}"

    def __str__(self):
        if self.type == ValueTypes.Undefined:
            raise NotImplementedError("Unknown Type!")
        return f"<Type:{value_type_strs[self.type]}, Value:{self.value}>"

    __repr__ = __str__


class BooleanValue(Value):
    def __init__(self, value: bool) -> None:
        self.value = value
        self.type = ValueTypes.Boolean

    def and_op(self, other: Value):
        if other.type == ValueTypes.Boolean:
            return BooleanValue(self.value and other.value)
        super().and_op(other)

    def or_op(self, other: Value):
        if other.type == ValueTypes.Boolean:
            return BooleanValue(self.value or other.value)
        super().or_op(other)

    def eq_op(self, other: Value):
        if other.type == ValueTypes.Boolean:
            return BooleanValue(self.value == other.value)
        super().eq_op(other)

    def neq_op(self, other: Value):
        if other.type == ValueTypes.Boolean:
            return BooleanValue(self.value != other.value)
        super().neq_op(other)


class NumberValue(Value):
    def __init__(self, value):
        self.value = float(value)
        self.type = ValueTypes.Number

    def add(self, other: Value):
        if other.type == ValueTypes.Number:
            return NumberValue(self.value + other.value)
        super().add(other)

    def sub(self, other: Value):
        if other.type == ValueTypes.Number:
            return NumberValue(self.value - other.value)
        super().sub(other)

    def mul(self, other: Value):
        if other.type == ValueTypes.Number:
            return NumberValue(self.value * other.value)
        super().mul(other)

    def div(self, other: Value):
        if other.type == ValueTypes.Number:
            return NumberValue(self.value / other.value)
        super().div(other)

    def pow(self, other: Value):
        if other.type == ValueTypes.Number:
            return NumberValue(self.value**other.value)
        super().pow(other)

    def eq_op(self, other: Value):
        if other.type == ValueTypes.Number:
            return BooleanValue(self.value == other.value)
        super().eq_op(other)

    def neq_op(self, other: Value):
        if other.type == ValueTypes.Number:
            return BooleanValue(self.value != other.value)
        super().neq_op(other)

    def lss_than(self, other: Value):
        if other.type == ValueTypes.Number:
            return BooleanValue(self.value < other.value)
        super().lss_than(other)

    def gtr_than(self, other: Value):
        if other.type == ValueTypes.Number:
            return BooleanValue(self.value > other.value)
        super().gtr_than(other)

    def gtr_eq(self, other: Value):
        if other.type == ValueTypes.Number:
            return BooleanValue(self.value >= other.value)
        super().gtr_eq(other)

    def lss_eq(self, other: Value):
        if other.type == ValueTypes.Number:
            return BooleanValue(self.value <= other.value)
        super().lss_eq(other)


class StringValue(Value):
    def __init__(self, value):
        self.value = value
        self.type = ValueTypes.String

    def add(self, other: Value):
        if other.type == ValueTypes.String:
            return StringValue(self.value + other.value)
        super().add(other)

    def eq_op(self, other: Value):
        if other.type == ValueTypes.String:
            return BooleanValue(self.value == other.value)
        super().eq_op(other)

    def neq_op(self, other: Value):
        if other.type == ValueTypes.String:
            return BooleanValue(self.value != other.value)
        super().neq_op(other)

    def get_idx(self, other: Value):
        if other.type == ValueTypes.Number:
            return self.value[int(other.value)]
        return super().get_idx(other)


class ListValue(Value):
    def __init__(self, value: list) -> None:
        self.value = value
        self.type = ValueTypes.List

    def add(self, other: Value):
        if other.type == ValueTypes.List:
            return ListValue(self.value + other.value)
        super().add(other)

    def get_idx(self, other: Value):
        if other.type == ValueTypes.Number:
            return self.value[int(other.value)]
        return super().get_idx(other)

    def set_idx(self, key: Self, other: Self):
        if key.type == ValueTypes.Number:
            self.value[int(key.value)] = other
            return other
        return super().set_idx(key, other)