import argparse
from pathlib import Path

from ast_parser import AstAccessors, ASTNode, ASTType, FuncDefStmt, Parser, VarDecl
from lang_value import BooleanValue, NumberValue, StringValue, Value
from tokenizer import Tokenizer, TokenType

MAX_FUNCTION_DEPTH = 100


class ClassDefinition:
    def __init__(self, name, parents, class_vars, class_methods):
        self.name: str = name
        self.parents: list[str] = parents
        self.class_vars: list[VarDecl] = class_vars
        self.class_methods: list[FuncDefStmt] = class_methods


class ClassInstanceFn:
    def __init__(self, fn_def: FuncDefStmt, my_inst: "ClassInstance" = None) -> None:
        self.fn_def: FuncDefStmt = fn_def
        self.my_inst: ClassInstance | None = my_inst


class ClassInstance:
    def __init__(self):
        self.class_name: str = ""
        self.class_table: dict[str, ClassDefinition] = {}
        self.function_table: dict[str, ClassInstanceFn] = {}
        self.symbol_table: dict = dict()


class FunctionFrame:
    def __init__(
        self,
        name: str = "main",
        body: list[ASTNode] | None = None,
    ) -> None:
        self.name: str = name
        self.body: list[ASTNode] = body if body is not None else list()
        self.symbol_table = dict()
        self.class_table: dict[str, ClassDefinition] = {}
        self.function_table: dict[str, ClassInstanceFn] = {}
        self.return_value = None
        self.halt = False
        self.break_loop = False


class Evaluator:
    def __init__(self) -> None:
        self.current_frame = None
        self.call_stack: list[FunctionFrame] = []

    def evaluate_frame(
        self, f_frame: FunctionFrame, inst: ClassInstance = None
    ) -> list:
        self.current_frame = f_frame
        for node in self.current_frame.body:
            self.evaluate(node, inst)
            if self.current_frame.halt:
                break

    def evaluate(self, node: ASTNode, inst: ClassInstance = None) -> Value:
        node_type: ASTType = node.type
        if node_type == ASTType.Number:
            return NumberValue(node.value)
        elif node_type == ASTType.String:
            return StringValue(node.value)
        elif node_type == ASTType.Boolean:
            return BooleanValue(node.value)
        elif node_type == ASTType.BinOp:
            left = self.evaluate(node.left, inst)
            right = self.evaluate(node.right, inst)
            if node.op == TokenType.PLUS:
                return left.add(right)
            elif node.op == TokenType.MINUS:
                return left.sub(right)
            elif node.op == TokenType.TIMES:
                return left.mul(right)
            elif node.op == TokenType.DIVIDE:
                return left.div(right)
            elif node.op == TokenType.POW:
                return left.pow(right)
            elif node.op == TokenType.LANGLE:
                return left.lss_than(right)
            elif node.op == TokenType.RANGLE:
                return left.gtr_than(right)
            elif node.op == TokenType.LO_AND:
                return left.and_op(right)
            elif node.op == TokenType.LO_OR:
                return left.or_op(right)
            elif node.op == TokenType.LO_EQU:
                return left.eq_op(right)
            elif node.op == TokenType.LO_GTE:
                return left.gtr_eq(right)
            elif node.op == TokenType.LO_LTE:
                return left.lss_eq(right)
        elif node_type == ASTType.VarDecl:
            value = self.evaluate(node.value, inst)
            self.evaluate_accessor(node.identifier, inst, value)
        elif node_type == ASTType.VarGet:
            return self.evaluate_accessor(node.value, inst)
        elif node_type == ASTType.FuncDefStmt:
            self.current_frame.function_table[node.name] = ClassInstanceFn(node, None)
        elif node_type == ASTType.FuncCallStmt:
            a_res = self.evaluate_accessor(node.name, inst)
            if isinstance(a_res, ClassDefinition):
                return self.make_instance(a_res, node.params, inst)
            if isinstance(a_res, ClassInstanceFn):
                return self.call_function(a_res, node.params, inst)
            raise Exception(
                f"Error: {node.name} is not defined in {self.current_frame.name}"
            )
        elif node_type == ASTType.FuncReturnStmt:
            self.current_frame.return_value = self.evaluate(node.value, inst)
            self.current_frame.halt = True
        elif node_type == ASTType.PrintStmt:
            value = self.evaluate(node.value, inst)
            print(value)
        elif node_type == ASTType.BreakStmt:
            self.current_frame.break_loop = True
        elif node_type == ASTType.IfStmt:
            if self.evaluate(node.cond, inst).value:
                for inode in node.body:
                    self.evaluate(inode, inst)
        elif node_type == ASTType.IfElseStmt:
            if self.evaluate(node.cond).value:
                for inode in node.true_body:
                    self.evaluate(inode, inst)
            else:
                for inode in node.false_body:
                    self.evaluate(inode, inst)
        elif node_type == ASTType.WhileStmt:
            cond_value: bool = self.evaluate(node.cond, inst).value
            while cond_value:
                for inode in node.body:
                    self.evaluate(inode, inst)
                    if self.current_frame.break_loop:
                        break
                cond_value: bool = self.evaluate(node.cond, inst).value
                if self.current_frame.break_loop:
                    cond_value = False
            if self.current_frame.break_loop:
                self.current_frame.break_loop = False
        elif node_type == ASTType.UntilStmt:
            cond_value = not self.evaluate(node.cond, inst).value
            while cond_value:
                for inode in node.body:
                    self.evaluate(inode, inst)
                    if self.current_frame.break_loop:
                        break
                cond_value: bool = not self.evaluate(node.cond, inst).value
                if self.current_frame.break_loop:
                    cond_value = False
            if self.current_frame.break_loop:
                self.current_frame.break_loop = False
        elif node_type == ASTType.ClassDefStmt:
            self.current_frame.class_table[node.name] = ClassDefinition(
                node.name, node.parents, node.class_vars, node.class_methods
            )
        elif node_type == ASTType.BareAccess or node_type == ASTType.MemberAccess:
            return self.evaluate_accessor(node, inst)
        return None

    def add_vars_and_fns_to_inst(self, c_inst: ClassInstance, c_def: ClassDefinition):
        for c_var in c_def.class_vars:
            self.evaluate_accessor(
                c_var.identifier, c_inst, self.evaluate(c_var.value), True
            )

        for c_fn in c_def.class_methods:
            c_inst.function_table[c_fn.name] = ClassInstanceFn(c_fn, c_inst)

    def make_instance(
        self,
        fut_class_def: ClassDefinition,
        class_params: list[ASTNode],
        inst: ClassInstance = None,
    ):

        fut_inst = ClassInstance()
        for par_class in fut_class_def.parents:
            if par_class == "#":
                continue
            try:
                self.add_vars_and_fns_to_inst(
                    fut_inst, self.current_frame.class_table[par_class]
                )
            except KeyError:
                raise Exception(
                    f"Error: attempting to instantiate parent class {par_class} of {fut_class_def.name}, but the class is not defined in function context"
                )

        self.add_vars_and_fns_to_inst(fut_inst, fut_class_def)

        if "__init__" in fut_inst.function_table:
            init_fn = fut_inst.function_table["__init__"]
            self.call_function(init_fn, class_params, fut_inst)

        fut_inst.class_name = fut_class_def.name
        return fut_inst

    def call_function(
        self,
        func_inst: ClassInstanceFn,
        func_call_params: list[ASTNode],
        c_inst: ClassInstance = None,
    ):
        func_def_node = func_inst.fn_def

        if (len(self.call_stack) + 1) >= MAX_FUNCTION_DEPTH:
            raise Exception(f"Error: function call stack too deep!")

        future_frame = FunctionFrame(func_def_node.name, func_def_node.body)

        if len(func_def_node.params) > 0:
            for p_name, p_node in zip(func_def_node.params, func_call_params):
                future_frame.symbol_table[p_name] = self.evaluate(p_node, c_inst)

        future_frame.function_table = self.current_frame.function_table

        self.call_stack.append(self.current_frame)
        self.evaluate_frame(future_frame, func_inst.my_inst)
        self.current_frame = self.call_stack.pop()

        return future_frame.return_value

    def evaluate_accessor(
        self,
        accessor: AstAccessors,
        instance: ClassInstance = None,
        assign_value=None,
        force_assign=False,
    ):
        if accessor.type == ASTType.BareAccess:
            a_value = accessor.value
            if assign_value:
                if instance and ((a_value in instance.symbol_table) or force_assign):
                    instance.symbol_table[a_value] = assign_value
                else:
                    self.current_frame.symbol_table[a_value] = assign_value
                return assign_value

            if instance and a_value in instance.symbol_table:
                return instance.symbol_table[a_value]

            if instance and a_value in instance.function_table:
                return instance.function_table[a_value]

            if instance and a_value in instance.class_table:
                return instance.symbol_table[a_value]

            if a_value in self.current_frame.symbol_table:
                return self.current_frame.symbol_table[a_value]

            if a_value in self.current_frame.class_table:
                return self.current_frame.class_table[a_value]

            if a_value in self.current_frame.function_table:
                return self.current_frame.function_table[a_value]

            raise Exception(f"Error: Undefined variable or property: {accessor}")

        if accessor.type == ASTType.MemberAccess:
            left_value = self.evaluate_accessor(accessor.left, instance)
            if isinstance(left_value, ClassInstance):
                return self.evaluate_accessor(
                    accessor.right, left_value, assign_value, force_assign
                )

            raise Exception(
                f"Left side of '.' must resolve to an object instance, got {left_value}."
            )
        return None


def main():
    parser = argparse.ArgumentParser("Juniper's Silly Little Language")
    parser.add_argument("ifile", help="The file with the code", type=Path)
    parser.add_argument(
        "--debug", help="Show debugging information", action="store_true"
    )
    args = parser.parse_args()

    DEBUG: bool = args.debug

    ifile: Path = args.ifile
    code = ifile.read_text()

    tokenizer = Tokenizer(code)
    tokens = tokenizer.tokenize()
    if DEBUG:
        print(tokens)

    parser = Parser(tokens)
    ast_lines = parser.parse()
    if DEBUG:
        for i, a in enumerate(ast_lines):
            print(i, ":", str(a))

    main_frame = FunctionFrame("main", ast_lines)
    evaluator = Evaluator()
    evaluator.evaluate_frame(main_frame)
    print(main_frame.symbol_table)


if __name__ == "__main__":
    main()
