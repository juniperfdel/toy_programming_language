import sys
from pathlib import Path

from ast_parser import (ASTNode, BinOp, Boolean, FuncCallStmt, FuncDefStmt,
                        FuncReturnStmt, IfElseStmt, IfStmt, Number, Parser,
                        PrintStmt, VarDecl, VarGet)
from tokenizer import Tokenizer, TokenType

MAX_FUNCTION_DEPTH = 100


class FunctionFrame:
    def __init__(
        self,
        name: str = "main",
        body: list[ASTNode] | None = None,
        symbol_table: dict | None = None,
    ) -> None:
        self.name: str = name
        self.body: list[ASTNode] = body if body is not None else list()
        self.symbol_table = symbol_table if symbol_table is not None else dict()
        self.return_value = None
        self.halt = False


class Evaluator:
    def __init__(self) -> None:
        self.current_frame = None
        self.call_stack: list[FunctionFrame] = []
        self.function_table: dict[str, FuncDefStmt] = {}

    def evaluate_frame(self, f_frame: FunctionFrame) -> list:
        self.current_frame = f_frame
        for node in self.current_frame.body:
            self.evaluate(node)
            if self.current_frame.halt:
                break

    def evaluate(self, node: ASTNode):
        if isinstance(node, Number):
            return node.value
        elif isinstance(node, Boolean):
            return node.value
        elif isinstance(node, BinOp):
            if node.op == TokenType.PLUS:
                return self.evaluate(node.left) + self.evaluate(node.right)
            elif node.op == TokenType.MINUS:
                return self.evaluate(node.left) - self.evaluate(node.right)
            elif node.op == TokenType.TIMES:
                return self.evaluate(node.left) * self.evaluate(node.right)
            elif node.op == TokenType.DIVIDE:
                return self.evaluate(node.left) / self.evaluate(node.right)
            elif node.op == TokenType.POW:
                return self.evaluate(node.left) ** self.evaluate(node.right)
            elif node.op == TokenType.LANGLE:
                return self.evaluate(node.left) < self.evaluate(node.right)
            elif node.op == TokenType.RANGLE:
                return self.evaluate(node.left) > self.evaluate(node.right)
            elif node.op == TokenType.LO_AND:
                return self.evaluate(node.left) and self.evaluate(node.right)
            elif node.op == TokenType.LO_OR:
                return self.evaluate(node.left) or self.evaluate(node.right)
            elif node.op == TokenType.LO_EQU:
                return self.evaluate(node.left) == self.evaluate(node.right)
            elif node.op == TokenType.LO_GTE:
                return self.evaluate(node.left) >= self.evaluate(node.right)
            elif node.op == TokenType.LO_LTE:
                return self.evaluate(node.left) <= self.evaluate(node.right)
        elif isinstance(node, VarDecl):
            value = self.evaluate(node.value)
            self.current_frame.symbol_table[node.identifier] = value
        elif isinstance(node, VarGet):
            var_name = node.value
            if var_name in self.current_frame.symbol_table:
                return self.current_frame.symbol_table[node.value]
            print(
                f"Error: {var_name} is not defined on line {node.get_line()} in {self.current_frame.name}"
            )
            sys.exit(1)
        elif isinstance(node, FuncDefStmt):
            self.function_table[node.name] = node
        elif isinstance(node, FuncCallStmt):
            return self.call_function(node)
        elif isinstance(node, FuncReturnStmt):
            self.current_frame.return_value = self.evaluate(node.value)
            self.current_frame.halt = True
        elif isinstance(node, PrintStmt):
            value = self.evaluate(node.value)
            print(value)
        elif isinstance(node, IfStmt):
            if self.evaluate(node.cond):
                for inode in node.body:
                    self.evaluate(inode)
        elif isinstance(node, IfElseStmt):
            if self.evaluate(node.cond):
                for inode in node.true_body:
                    self.evaluate(inode)
            else:
                for inode in node.false_body:
                    self.evaluate(inode)
        return None

    def call_function(self, func_call_node: FuncCallStmt):
        func_def_node = self.function_table[func_call_node.name]
        if len(func_def_node.params) > 0:
            param_symbols = {
                p_name: self.evaluate(p_node)
                for p_name, p_node in zip(func_def_node.params, func_call_node.params)
            }
        else:
            param_symbols = dict()
        future_frame = FunctionFrame(
            func_def_node.name, func_def_node.body, param_symbols
        )

        if (len(self.call_stack) + 1) >= MAX_FUNCTION_DEPTH:
            print(f"Error: function call stack too deep!")
            sys.exit(1)
        self.call_stack.append(self.current_frame)
        self.evaluate_frame(future_frame)
        self.current_frame = self.call_stack.pop()
        return future_frame.return_value


def main():
    code: str = Path("test_code.test").read_text()
    tokenizer = Tokenizer(code)
    tokens = tokenizer.tokenize()
    if False:
        print(tokens)

    parser = Parser(tokens)
    ast_lines = parser.parse()
    if False:
        for i, a in enumerate(ast_lines):
            print(i, ":", str(a))

    main_frame = FunctionFrame("main", ast_lines)
    evaluator = Evaluator()
    evaluator.evaluate_frame(main_frame)
    print(main_frame.symbol_table)


if __name__ == "__main__":
    main()
