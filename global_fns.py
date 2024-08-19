from lang_value import ValueTypes, Value, StringValue, BooleanValue, NumberValue, ListValue

def make_string(fn_args: list[Value]):
    return StringValue(str(fn_args[0].value))

def make_bool(fn_args: list[Value]):
    return BooleanValue(bool(fn_args[0].value))

def make_number(fn_args: list[Value]):
    return NumberValue(float(fn_args[0].value))

def print_value(fn_args: list[Value]):
    print(fn_args[0])

def append_to_list(fn_args: list[Value]):
    first_arg = fn_args[0]
    if first_arg.type != ValueTypes.List:
        raise TypeError("Only lists can be appended to!")
    first_arg_l: list = fn_args[0].value
    second_arg = fn_args[1]
    first_arg_l.append(second_arg)
    return ListValue(first_arg_l)

global_functions = {
    "string": make_string,
    "boolean": make_bool,
    "number": make_number,
    "print": print_value,
    "append": append_to_list
}