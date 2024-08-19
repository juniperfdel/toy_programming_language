from enum import Enum, auto
from typing import Callable, TypeAlias, Union

from tokenizer import Token, TokenType, keyword_tokens


class ASTType(Enum):
    Number = auto()
    String = auto()
    Boolean = auto()
    ListType = auto()

    BinOp = auto()

    VarDecl = auto()
    VarGet = auto()

    FuncDefStmt = auto()
    FuncCallStmt = auto()
    FuncReturnStmt = auto()

    PrintStmt = auto()
    IfStmt = auto()
    IfElseStmt = auto()

    WhileStmt = auto()
    UntilStmt = auto()

    BreakStmt = auto()

    ClassDefStmt = auto()
    SelfStmt = auto()
    NewStmt = auto()

    # Types of Access
    BareAccess = auto()
    MemberAccess = auto()
    IndexAccess = auto()


class ASTNode:
    type = None
    line = -1


class BinOp(ASTNode):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right
        self.type = ASTType.BinOp

    def __str__(self):
        return f"({self.left} {self.op.name} {self.right})"


class Number(ASTNode):
    def __init__(self, value):
        self.value = value
        self.type = ASTType.Number

    def __str__(self):
        return str(self.value)


class String(ASTNode):
    def __init__(self, value):
        self.value = value
        self.type = ASTType.String

    def __str__(self):
        return str(self.value)


class BareAccess(ASTNode):
    def __init__(self, value) -> None:
        self.value: str = value
        self.type = ASTType.BareAccess

    def __str__(self) -> str:
        return f"{self.value}"


class MemberAccess(ASTNode):
    def __init__(self, left: "AstAccessors", right: "AstAccessors") -> None:
        self.left = left
        self.right = right
        self.type = ASTType.MemberAccess

    def __str__(self) -> str:
        return f"{self.left}.{self.right}"


class IndexAccess(ASTNode):
    def __init__(self, left: "AstAccessors", right: "AstAccessors") -> None:
        self.left = left
        self.right = right
        self.type = ASTType.IndexAccess

    def __str__(self) -> str:
        return f"{self.left}[{self.right}]"


AstAccessors: TypeAlias = Union[BareAccess, MemberAccess, IndexAccess]


class VarDecl(ASTNode):
    def __init__(self, identifier: AstAccessors, value) -> None:
        self.identifier: AstAccessors = identifier
        self.value = value
        self.type = ASTType.VarDecl

    def __str__(self) -> str:
        return f"(ASSIGN {self.identifier} {self.value})"


class VarGet(ASTNode):
    def __init__(self, value: AstAccessors) -> None:
        self.value: AstAccessors = value
        self.type = ASTType.VarGet

    def __str__(self) -> str:
        return f"(GET {self.value})"


class PrintStmt(ASTNode):
    def __init__(self, value) -> None:
        self.value = value
        self.type = ASTType.PrintStmt

    def __str__(self) -> str:
        return f"(PRINT {self.value})"


class FuncDefStmt(ASTNode):
    def __init__(self, name: str, params: list[str], body: list[ASTNode]) -> None:
        self.name = name
        self.params = params
        self.body = body
        self.type = ASTType.FuncDefStmt

    def __str__(self) -> str:
        return f'(FUNCDEF {self.name} {self.params} ({"; ".join(map(str,self.body))}))'


class FuncCallStmt(ASTNode):
    def __init__(self, name: AstAccessors, params: list[ASTNode]) -> None:
        self.name: AstAccessors = name
        self.params = params
        self.type = ASTType.FuncCallStmt

    def __str__(self) -> str:
        return f'(FUNCCALL {self.name} {"; ".join(map(str,self.params))})'


class FuncReturnStmt(ASTNode):
    def __init__(self, value: ASTNode) -> None:
        self.value = value
        self.type = ASTType.FuncReturnStmt

    def __str__(self) -> str:
        return f"(FUNCRETURN {self.value})"


class ClassDefStmt(ASTNode):
    def __init__(
        self,
        name: str,
        parents: list[str],
        class_vars: list[VarDecl],
        class_methods: list[FuncDefStmt],
    ) -> None:
        self.name: str = name
        self.parents: list[str] = parents
        self.class_vars: list[VarDecl] = class_vars
        self.class_methods: list[FuncDefStmt] = class_methods
        self.type = ASTType.ClassDefStmt

    def __str__(self) -> str:
        return f'(CLASSDEF {self.name} {self.parents} ({"; ".join(map(str,self.class_vars))}) ({"; ".join(map(str,self.class_methods))}))'


class IfStmt(ASTNode):
    def __init__(self, cond: ASTNode, body: list[ASTNode]) -> None:
        self.cond = cond
        self.body = body
        self.type = ASTType.IfStmt

    def __str__(self) -> str:
        return f'(IF {self.cond} ({"; ".join(map(str,self.body))})'


class IfElseStmt(ASTNode):
    def __init__(
        self, cond: ASTNode, true_body: list[ASTNode], false_body: list[ASTNode]
    ) -> None:
        self.cond = cond
        self.true_body = true_body
        self.false_body: list[ASTNode] = false_body
        self.type = ASTType.IfElseStmt

    def __str__(self) -> str:
        return f'(IFELSE {self.cond} ({"; ".join(map(str,self.true_body))}) ({"; ".join(map(str,self.false_body))})'


class WhileStmt(ASTNode):
    def __init__(self, cond: ASTNode, body: list[ASTNode]) -> None:
        self.cond = cond
        self.body = body
        self.type = ASTType.WhileStmt

    def __str__(self) -> str:
        return f'(WHILE {self.cond} ({"; ".join(map(str,self.body))})'


class UntilStmt(ASTNode):
    def __init__(self, cond: ASTNode, body: list[ASTNode]) -> None:
        self.cond = cond
        self.body = body
        self.type = ASTType.UntilStmt

    def __str__(self) -> str:
        return f'(UNTIL {self.cond} ({"; ".join(map(str,self.body))})'


class BreakStmt(ASTNode):
    def __init__(self) -> None:
        self.type = ASTType.BreakStmt

    def __str__(self) -> str:
        return "(BREAK)"


class Boolean(ASTNode):
    def __init__(self, value: bool) -> None:
        self.value = value
        self.type = ASTType.Boolean

    def __str__(self) -> str:
        return "TRUE" if self.value else "FALSE"


class ListType(ASTNode):
    def __init__(self, elements: list[ASTNode]) -> None:
        self.elements = elements
        self.type = ASTType.ListType

    def __str__(self) -> str:
        return f'(ListType ({"; ".join(map(str,self.elements))}))'


def check_if_token_keyword(itok: Token) -> None:
    if itok.value.lower() in keyword_tokens.keys():
        raise SyntaxError(
            f"ERROR @ [{itok.line}, {itok.column}]; Do not use a keyword for a class or variable name!"
        )


def consume_accessor(ip: "Parser") -> AstAccessors:
    id_tok = ip.consume(TokenType.IDENTIFIER)
    check_if_token_keyword(id_tok)
    if ip.pos < ip.tok_len:
        if ip.tokens[ip.pos].type == TokenType.DOT:
            ip.consume(TokenType.DOT)
            return MemberAccess(BareAccess(id_tok.value), consume_accessor(ip))
        if ip.tokens[ip.pos].type == TokenType.LSQBRA:
            return IndexAccess(
                BareAccess(id_tok.value),
                consume_block(
                    ip, left_bound=TokenType.LSQBRA, right_bound=TokenType.RSQBRA
                )[0],
            )
    return BareAccess(id_tok.value)


def consume_print(ip: "Parser"):
    ip.consume(TokenType.PRINT)
    ip.consume(TokenType.LPAREN)
    value: ASTNode = ip.logic_stg1()
    ip.consume(TokenType.RPAREN)
    return PrintStmt(value)


def consume_var_decl(ip: "Parser"):
    ip.consume(TokenType.ASSIGN)
    id_token: AstAccessors = consume_accessor(ip)
    ip.consume(TokenType.EQUALS)
    value: ASTNode = ip.logic_stg1()
    return VarDecl(id_token, value)


def consume_return(ip: "Parser"):
    ip.consume(TokenType.RETURN)
    value: ASTNode = ip.logic_stg1()
    return FuncReturnStmt(value)


def consume_break(ip: "Parser"):
    ip.consume(TokenType.BREAK)
    return BreakStmt()


def consume_block(
    ip: "Parser",
    token_ingest: "Parser" = None,
    left_bound: TokenType = TokenType.LBRACK,
    right_bound: TokenType = TokenType.RBRACK,
    include_comma: bool = False,
):
    tok_ingest = token_ingest if token_ingest is not None else Parser()
    ip.consume(left_bound)
    start_pos = ip.pos
    c_depth = 0
    while ip.pos < ip.tok_len:
        c_tok: Token = ip.tokens[ip.pos]
        if c_tok.type == right_bound:
            c_depth = c_depth - 1
            if c_depth < 0:
                break
        if c_tok.type == left_bound:
            c_depth = c_depth + 1
        ip.consume()
    end_pos = ip.pos
    ip.consume(right_bound)
    tok_ingest.reset(ip.tokens[start_pos:end_pos])
    return tok_ingest.parse(include_comma)


def consume_func_def(ip: "Parser"):
    ip.consume(TokenType.FUNCDEF)
    func_name_tok: Token = ip.consume(TokenType.IDENTIFIER)
    check_if_token_keyword(func_name_tok)

    ip.consume(TokenType.LPAREN)
    arg_names: list[str] = []
    while ip.pos < ip.tok_len:
        c_tok: Token = ip.tokens[ip.pos]
        if c_tok.type == TokenType.RPAREN:
            break
        arg_names.append(ip.consume(TokenType.IDENTIFIER).value)
        c_tok: Token = ip.tokens[ip.pos]
        if c_tok.type == TokenType.COMMA:
            ip.consume(TokenType.COMMA)
    ip.consume(TokenType.RPAREN)

    body_nodes = consume_block(ip)
    return FuncDefStmt(func_name_tok.value, arg_names, body_nodes)


def consume_class_def(ip: "Parser"):
    ip.consume(TokenType.CLASSDEF)
    class_name_tok = ip.consume(TokenType.IDENTIFIER)
    check_if_token_keyword(class_name_tok)

    # parents processing
    parents: list[str] = []
    if ip.tokens[ip.pos].type == TokenType.LPAREN:
        ip.consume(TokenType.LPAREN)
        proc_pars = True
        while proc_pars:
            parents.append(ip.consume(TokenType.IDENTIFIER).value)
            if ip.tokens[ip.pos].type == TokenType.COMMA:
                ip.consume(TokenType.COMMA)
                continue

            ip.consume(TokenType.RPAREN)
            proc_pars = False
    else:
        parents = ["#"]
    ip.consume(TokenType.LBRACK)

    class_methods = []
    class_vars = []

    proc_body = True
    while proc_body:
        c_tok = ip.tokens[ip.pos]
        if c_tok.type == TokenType.NEWLINE:
            ip.consume()
            continue
        elif c_tok.type == TokenType.FUNCDEF:
            class_methods.append(consume_func_def(ip))
            continue
        elif c_tok.type == TokenType.ASSIGN:
            class_vars.append(consume_var_decl(ip))
            continue
        ip.consume(TokenType.RBRACK)
        proc_body = False

    return ClassDefStmt(class_name_tok.value, parents, class_vars, class_methods)


def consume_if(ip: "Parser"):
    ip.consume(TokenType.LO_IF)
    if_parser = Parser()
    cond_node: ASTNode = consume_block(
        ip, if_parser, TokenType.LPAREN, TokenType.RPAREN
    )[0]
    body_nodes = consume_block(ip, if_parser)

    if ip.pos < ip.tok_len:
        c_tok: Token = ip.tokens[ip.pos]
        if c_tok.type == TokenType.LO_ELSE:
            ip.consume(TokenType.LO_ELSE)
            else_body_nodes: list[ASTNode] = consume_block(ip, if_parser)
            return IfElseStmt(cond_node, body_nodes, else_body_nodes)
    return IfStmt(cond_node, body_nodes)


def consume_while(ip: "Parser"):
    ip.consume(TokenType.WHILE)
    cond_parser = Parser()
    cond_node: ASTNode = consume_block(
        ip, cond_parser, TokenType.LPAREN, TokenType.RPAREN
    )[0]
    body_nodes = consume_block(ip, cond_parser)
    return WhileStmt(cond_node, body_nodes)


def consume_until(ip: "Parser"):
    ip.consume(TokenType.UNTIL)
    cond_parser = Parser()
    cond_node: ASTNode = consume_block(
        ip, cond_parser, TokenType.LPAREN, TokenType.RPAREN
    )[0]
    body_nodes = consume_block(ip, cond_parser)
    return UntilStmt(cond_node, body_nodes)


parse_keyword_ast: dict[TokenType, Callable[["Parser"], ASTNode]] = {
    TokenType.ASSIGN: consume_var_decl,
    TokenType.PRINT: consume_print,
    TokenType.RETURN: consume_return,
    TokenType.FUNCDEF: consume_func_def,
    TokenType.LO_IF: consume_if,
    TokenType.WHILE: consume_while,
    TokenType.UNTIL: consume_until,
    TokenType.CLASSDEF: consume_class_def,
    TokenType.BREAK: consume_break,
}


def consume_func_call(ip: "Parser", func_name_node: AstAccessors) -> FuncCallStmt:
    arg_nodes: list[ASTNode] = consume_block(
        ip,
        left_bound=TokenType.LPAREN,
        right_bound=TokenType.RPAREN,
        include_comma=True,
    )
    return FuncCallStmt(func_name_node, arg_nodes)


def consume_list_def(ip: "Parser") -> ListType:
    arg_nodes: list[ASTNode] = consume_block(
        ip,
        left_bound=TokenType.LSQBRA,
        right_bound=TokenType.RSQBRA,
        include_comma=True,
    )
    return ListType(arg_nodes)


def consume_func_set_id(ip: "Parser", assign_name: AstAccessors) -> VarDecl:
    ip.consume(TokenType.EQUALS)
    value: ASTNode = ip.logic_stg1()
    return VarDecl(assign_name, value)


def consume_identifier(ip: "Parser"):
    name_tok = consume_accessor(ip)
    if ip.pos < ip.tok_len:
        c_tok = ip.tokens[ip.pos]
        if c_tok.type == TokenType.LPAREN:
            return consume_func_call(ip, name_tok)
        elif c_tok.type == TokenType.EQUALS:
            return consume_func_set_id(ip, name_tok)
    rv = VarGet(name_tok)
    rv.line = name_tok.line
    return rv


# Parser implementation
class Parser:
    split_tokens = {
        False: [TokenType.NEWLINE],
        True: [TokenType.COMMA, TokenType.NEWLINE],
    }

    def __init__(self, tokens: list[Token] | None = None):
        if tokens is None:
            tokens = []
        self.tokens: list[Token] = tokens
        self.tok_len: int = len(self.tokens)
        self.pos: int = 0

    def reset(self, tokens: list[Token]):
        self.__init__(tokens)

    def parse(self, include_comma=False) -> list[ASTNode]:
        token_delim = self.split_tokens[include_comma]
        rv = []
        while self.pos < self.tok_len:
            if self.tokens[self.pos].type in token_delim:
                self.consume(self.tokens[self.pos].type)
                continue
            rv.append(self.statement())
            if self.pos < self.tok_len:
                if self.tokens[self.pos].type in token_delim:
                    self.consume(self.tokens[self.pos].type)
                else:
                    c_tok = self.tokens[self.pos]
                    raise SyntaxError(
                        f"Expected {'Newline or Comma' if include_comma else 'Newline'}, but got {c_tok.type} at {c_tok.line}:{c_tok.column}"
                    )
        return rv

    def consume(self, token_type: TokenType | None = None) -> Token:
        token: Token = self.tokens[self.pos]
        if (token_type is None) or (token.type == token_type):
            self.pos += 1
            return token
        raise SyntaxError(
            f"Expected {token_type} but got {token.type} at {token.line}:{token.column}"
        )

    def statement(self) -> ASTNode:
        if (self.pos < self.tok_len) and (
            self.tokens[self.pos].type in parse_keyword_ast
        ):
            return self.keyword_consume()
        return self.logic_stg1()

    def keyword_consume(self) -> ASTNode:
        token: Token = self.tokens[self.pos]
        try:
            return parse_keyword_ast[token.type](self)
        except KeyError:
            raise SyntaxError(f"Unknown token: {token}")

    def logic_stg1(self) -> ASTNode:
        node: ASTNode = self.logic_stg2()
        while self.pos < self.tok_len and self.tokens[self.pos].type in [
            TokenType.LO_AND,
            TokenType.LO_OR,
        ]:
            token = self.consume()
            node = BinOp(left=node, op=token.type, right=self.logic_stg2())
        return node

    def logic_stg2(self) -> ASTNode:
        node = self.exp()
        while self.pos < self.tok_len and self.tokens[self.pos].type in [
            TokenType.LO_LTE,
            TokenType.LO_GTE,
            TokenType.LO_EQU,
            TokenType.LO_NEQ,
            TokenType.LANGLE,
            TokenType.RANGLE,
        ]:
            token = self.consume()
            node = BinOp(left=node, op=token.type, right=self.exp())
        return node

    def exp(self) -> ASTNode:
        node = self.expr()
        while self.pos < self.tok_len and self.tokens[self.pos].type in [TokenType.POW]:
            token = self.consume()
            node = BinOp(left=node, op=token.type, right=self.expr())
        return node

    def expr(self) -> ASTNode:
        node = self.term()
        while self.pos < self.tok_len and self.tokens[self.pos].type in (
            TokenType.PLUS,
            TokenType.MINUS,
        ):
            token = self.consume()
            node = BinOp(left=node, op=token.type, right=self.term())
        return node

    def term(self) -> ASTNode:
        node = self.factor()
        while self.pos < self.tok_len and self.tokens[self.pos].type in (
            TokenType.TIMES,
            TokenType.DIVIDE,
        ):
            token = self.consume()
            node = BinOp(left=node, op=token.type, right=self.factor())
        return node

    def factor(self) -> ASTNode:
        token: Token = self.tokens[self.pos]
        if token.type == TokenType.NUMBER:
            self.pos += 1
            return Number(token.value)
        elif token.type == TokenType.STRING:
            self.pos += 1
            return String(token.value)
        elif token.type == TokenType.LO_TRUE:
            self.consume(TokenType.LO_TRUE)
            return Boolean(True)
        elif token.type == TokenType.LO_FALSE:
            self.consume(TokenType.LO_FALSE)
            return Boolean(False)
        elif token.type == TokenType.IDENTIFIER:
            return consume_identifier(self)
        elif token.type == TokenType.LSQBRA:
            return consume_list_def(self)
        elif token.type == TokenType.LPAREN:
            self.consume(TokenType.LPAREN)
            node = self.logic_stg1()
            self.consume(TokenType.RPAREN)
            return node
