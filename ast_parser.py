from typing import Callable

from tokenizer import Token, TokenType


class ASTNode:
    def set_line(self, line: int):
        self.line = line

    def get_line(self) -> int:
        if hasattr(self, "line"):
            return self.line
        return -1


class BinOp(ASTNode):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

    def __str__(self):
        return f"({self.left} {self.op.name} {self.right})"


class Number(ASTNode):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)


class Variable(ASTNode):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)


class VarDecl(ASTNode):
    def __init__(self, identifier, value) -> None:
        self.identifier = identifier
        self.value = value

    def __str__(self) -> str:
        return f"(ASSIGN {self.identifier} TO {self.value})"


class VarGet(ASTNode):
    def __init__(self, value) -> None:
        self.value = value

    def __str__(self) -> str:
        return f"(GET {self.value})"


class PrintStmt(ASTNode):
    def __init__(self, value) -> None:
        self.value = value

    def __str__(self) -> str:
        return f"(PRINT {self.value})"


class FuncDefStmt(ASTNode):
    def __init__(self, name: str, params: list[str], body: list[ASTNode]) -> None:
        self.name = name
        self.params = params
        self.body = body

    def __str__(self) -> str:
        return f'(FUNCDEF {self.name} {self.params} ({"; ".join(map(str,self.body))}))'


class FuncCallStmt(ASTNode):
    def __init__(self, name: str, params: list[ASTNode]) -> None:
        self.name = name
        self.params = params

    def __str__(self) -> str:
        return f'(FUNCCALL {self.name} {"; ".join(map(str,self.params))})'


class FuncReturnStmt(ASTNode):
    def __init__(self, value: ASTNode) -> None:
        self.value = value

    def __str__(self) -> str:
        return f"(FUNCRETURN {self.value})"


class IfStmt(ASTNode):
    def __init__(self, cond: ASTNode, body: list[ASTNode]) -> None:
        self.cond = cond
        self.body = body

    def __str__(self) -> str:
        return f'(IF {self.cond} ({"; ".join(map(str,self.body))})'


class IfElseStmt(ASTNode):
    def __init__(
        self, cond: ASTNode, true_body: list[ASTNode], false_body: list[ASTNode]
    ) -> None:
        self.cond = cond
        self.true_body = true_body
        self.false_body: list[ASTNode] = false_body

    def __str__(self) -> str:
        return f'(IFELSE {self.cond} ({"; ".join(map(str,self.true_body))}) ({"; ".join(map(str,self.false_body))})'


class Boolean(ASTNode):
    def __init__(self, value: bool) -> None:
        self.value = value

    def __str__(self) -> str:
        return "TRUE" if self.value else "FALSE"


def consume_print(ip: "Parser"):
    ip.consume(TokenType.PRINT)
    ip.consume(TokenType.LPAREN)
    value: ASTNode = ip.logic_stg1()
    ip.consume(TokenType.RPAREN)
    return PrintStmt(value)


def consume_var_decl(ip: "Parser"):
    ip.consume(TokenType.ASSIGN)
    id_token: Token = ip.consume(TokenType.IDENTIFIER)
    ip.consume(TokenType.EQUALS)
    value: ASTNode = ip.logic_stg1()
    return VarDecl(id_token.value, value)


def consume_return(ip: "Parser"):
    ip.consume(TokenType.RETURN)
    value: ASTNode = ip.logic_stg1()
    return FuncReturnStmt(value)


def consume_block(
    ip: "Parser",
    token_ingest: "Parser" = None,
    left_bound: TokenType = TokenType.LBRACK,
    right_bound: TokenType = TokenType.RBRACK,
):
    tok_ingest = token_ingest if token_ingest is not None else Parser()
    ip.consume(left_bound)
    body_tokens: list[Token] = []
    c_depth = 0
    while ip.pos < ip.tok_len:
        c_tok: Token = ip.tokens[ip.pos]
        if c_tok.type == right_bound:
            c_depth = c_depth - 1
            if c_depth < 0:
                break
        if c_tok.type == left_bound:
            c_depth = c_depth + 1
        body_tokens.append(ip.consume())
    ip.consume(right_bound)
    tok_ingest.reset(body_tokens)
    return tok_ingest.parse()


def consume_func_def(ip: "Parser"):
    ip.consume(TokenType.FUNCDEF)
    func_name: str = ip.consume(TokenType.IDENTIFIER).value

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
    return FuncDefStmt(func_name, arg_names, body_nodes)


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


parse_keyword_ast: dict[TokenType, Callable[["Parser"], ASTNode]] = {
    TokenType.ASSIGN: consume_var_decl,
    TokenType.PRINT: consume_print,
    TokenType.RETURN: consume_return,
    TokenType.FUNCDEF: consume_func_def,
    TokenType.LO_IF: consume_if,
}


def consume_func_call(ip: "Parser", func_name_node: str) -> FuncCallStmt:
    ip.consume(TokenType.LPAREN)
    arg_parser = Parser()
    arg_tokens: list[Token] = []
    arg_nodes: list[ASTNode] = []
    c_depth = 0
    while ip.pos < ip.tok_len:
        c_tok: Token = ip.tokens[ip.pos]
        if c_tok.type == TokenType.COMMA:
            arg_parser.reset(arg_tokens)
            arg_nodes.append(arg_parser.logic_stg1())
            arg_tokens = []
            ip.consume(TokenType.COMMA)
            continue
        if c_tok.type == TokenType.RPAREN:
            c_depth = c_depth - 1
            if c_depth < 0:
                arg_parser.reset(arg_tokens)
                arg_nodes.append(arg_parser.logic_stg1())
                arg_tokens = []
                ip.consume(TokenType.RPAREN)
                break
        if c_tok.type == TokenType.LPAREN:
            c_depth = c_depth + 1
        arg_tokens.append(ip.consume())

    if arg_tokens:
        arg_parser.reset(arg_tokens)
        arg_nodes.append(arg_parser.logic_stg1())

    return FuncCallStmt(func_name_node, arg_nodes)


def consume_func_set_id(ip: "Parser", assign_name: str) -> VarDecl:
    ip.consume(TokenType.EQUALS)
    value: ASTNode = ip.logic_stg1()
    return VarDecl(assign_name, value)


def consume_identifier(ip: "Parser"):
    name_tok = ip.consume(TokenType.IDENTIFIER)
    if ip.pos < ip.tok_len:
        c_tok = ip.tokens[ip.pos]
        if c_tok.type == TokenType.LPAREN:
            return consume_func_call(ip, name_tok.value)
        elif c_tok.type == TokenType.EQUALS:
            return consume_func_set_id(ip, name_tok.value)
    rv = VarGet(name_tok.value)
    rv.set_line(name_tok.line)
    return rv


# Parser implementation
class Parser:
    def __init__(self, tokens: list[Token] | None = None):
        if tokens is None:
            tokens = []
        self.tokens: list[Token] = tokens
        self.tok_len: int = len(self.tokens)
        self.pos: int = 0

    def reset(self, tokens: list[Token]):
        self.__init__(tokens)

    def parse(self) -> list[ASTNode]:
        rv = []
        while self.pos < self.tok_len:
            if self.tokens[self.pos].type == TokenType.NEWLINE:
                self.consume(TokenType.NEWLINE)
                continue
            rv.append(self.statement())
            if self.pos < self.tok_len:
                self.consume(TokenType.NEWLINE)
        return rv

    def consume(self, token_type: TokenType | None = None) -> Token:
        token: Token = self.tokens[self.pos]
        if (token_type is None) or (token.type == token_type):
            self.pos += 1
            return token
        raise SyntaxError(f"Expected {token_type} but got {token.type}")

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
        elif token.type == TokenType.LO_TRUE:
            self.consume(TokenType.LO_TRUE)
            return Boolean(True)
        elif token.type == TokenType.LO_FALSE:
            self.consume(TokenType.LO_FALSE)
            return Boolean(False)
        elif token.type == TokenType.IDENTIFIER:
            return consume_identifier(self)
        elif token.type == TokenType.LPAREN:
            self.consume(TokenType.LPAREN)
            node = self.logic_stg1()
            self.consume(TokenType.RPAREN)
            return node
