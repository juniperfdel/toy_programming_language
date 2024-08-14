from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    # Atomic Types
    NUMBER = auto()
    STRING = auto()

    # Math
    PLUS = auto()
    MINUS = auto()
    TIMES = auto()
    DIVIDE = auto()
    POW = auto()

    # Additional Symbols
    LANGLE = auto()
    RANGLE = auto()
    AMPER = auto()
    PIPE = auto()

    # Logic
    LO_AND = auto()
    LO_OR = auto()
    LO_LTE = auto()
    LO_GTE = auto()
    LO_EQU = auto()
    LO_NEQ = auto()

    LO_TRUE = auto()
    LO_FALSE = auto()

    # String
    QUOTE = auto()

    # Seperators
    LPAREN = auto()
    RPAREN = auto()

    LBRACK = auto()
    RBRACK = auto()

    LSQBRA = auto()
    RSQBRA = auto()

    # Code
    EQUALS = auto()
    COMMA = auto()
    IDENTIFIER = auto()
    NEWLINE = auto()

    # Keywords
    ASSIGN = auto()
    PRINT = auto()
    FUNCDEF = auto()
    RETURN = auto()
    LO_IF = auto()
    LO_ELSE = auto()

    # Loops
    WHILE = auto()
    UNTIL = auto()
    BREAK = auto()

    # Class
    CLASSDEF = auto()
    DOT = auto()


@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int


SPACE_CHAR = " 	"
FIRST_VCHAR = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
ID_VCHAR = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"


def make_token_factory(token_type: TokenType, token_str: None | str = None):
    if token_str is None:

        def token_factory(itokenizer: "Tokenizer", ikeyword: str):
            return Token(token_type, ikeyword, itokenizer.line, itokenizer.column)

    else:

        def token_factory(itokenizer: "Tokenizer"):
            return Token(token_type, token_str, itokenizer.line, itokenizer.column)

    return token_factory


D_LO_OP_TYPES = {
    "&&": TokenType.LO_AND,
    "||": TokenType.LO_OR,
    ">=": TokenType.LO_GTE,
    "<=": TokenType.LO_LTE,
    "=<": TokenType.LO_LTE,
    "=>": TokenType.LO_GTE,
    "==": TokenType.LO_EQU,
    "!=": TokenType.LO_NEQ,
}

d_lo_check = "&|><="

d_lo_op_tokens = {
    op_char: make_token_factory(token_type, op_char)
    for op_char, token_type in D_LO_OP_TYPES.items()
}

OP_TYPES = {
    TokenType.NEWLINE: "\n",
    TokenType.PLUS: "+",
    TokenType.MINUS: "-",
    TokenType.TIMES: "*",
    TokenType.DIVIDE: "/",
    TokenType.POW: "^",
    TokenType.LBRACK: "{",
    TokenType.RBRACK: "}",
    TokenType.LPAREN: "(",
    TokenType.RPAREN: ")",
    TokenType.LSQBRA: "[",
    TokenType.RSQBRA: "]",
    TokenType.EQUALS: "=",
    TokenType.COMMA: ",",
    TokenType.LANGLE: "<",
    TokenType.RANGLE: ">",
    TokenType.AMPER: "&",
    TokenType.PIPE: "|",
    TokenType.DOT: ".",
}

op_tokens = {
    op_char: make_token_factory(token_type, op_char)
    for token_type, op_char in OP_TYPES.items()
}

KEYWORD_TOKEN_TYPES = {
    TokenType.ASSIGN: "var",
    TokenType.PRINT: "print",
    TokenType.FUNCDEF: "func",
    TokenType.RETURN: "return",
    TokenType.LO_IF: "if",
    TokenType.LO_ELSE: "else",
    TokenType.LO_TRUE: "true",
    TokenType.LO_FALSE: "false",
    TokenType.WHILE: "while",
    TokenType.UNTIL: "until",
    TokenType.CLASSDEF: "class",
    TokenType.BREAK: "break"
}


keyword_tokens = {
    keyword: make_token_factory(token_type)
    for token_type, keyword in KEYWORD_TOKEN_TYPES.items()
}


class Tokenizer:
    def __init__(self, code: str):
        self.code: str = code
        self.pos: int = 0
        self.current_char: str = self.code[self.pos] if self.code else None
        self.line: int = 1
        self.column: int = 0

    def advance(self):
        self.pos += 1
        if self.pos < len(self.code):
            self.current_char = self.code[self.pos]
            if self.current_char == "\n":
                self.line += 1
                self.column = 0
            else:
                self.column += 1
        else:
            self.current_char = None

    def tokenize(self) -> list[Token]:
        tokens = []
        while self.current_char is not None:
            if self.current_char in SPACE_CHAR:
                self.skip_whitespace()
            if self.current_char == "#":
                self.skip_rest_of_line()
            if (self.current_char in d_lo_check) and (self.pos < len(self.code)):
                op_check: str = self.current_char + self.code[self.pos + 1]
                if op_check in d_lo_op_tokens:
                    tokens.append(d_lo_op_tokens[op_check](self))
                    self.advance()
                    self.advance()
                    continue
            token_func = op_tokens.get(self.current_char, None)
            if token_func:
                tokens.append(token_func(self))
                self.advance()
            elif self.current_char in FIRST_VCHAR:
                tokens.append(self.identifier())
            elif self.current_char.isdigit():
                tokens.append(self.number())
            elif self.current_char == '"' or self.current_char == "'":
                tokens.append(self.string(self.current_char))
            else:
                raise RuntimeError(
                    f"Unexpected character {self.current_char!r} at line {self.line} column {self.column}"
                )
        return tokens

    def skip_whitespace(self):
        while self.current_char is not None and (self.current_char in SPACE_CHAR):
            self.advance()

    def skip_rest_of_line(self):
        while self.current_char is not None and self.current_char != "\n":
            self.advance()

    def string(self, quote_type) -> Token:
        self.advance()
        start_pos = self.pos
        while self.current_char is not None and self.current_char != quote_type:
            self.advance()
        if self.current_char == quote_type:
            value = self.code[start_pos : self.pos]
            self.advance()
        else:
            raise RuntimeError(
                f"Did not find {quote_type} before line {self.line} column {self.column}"
            )
        return Token(TokenType.STRING, value, self.line, self.column)

    def number(self) -> Token:
        start_pos = self.pos
        while self.current_char is not None and self.current_char.isdigit():
            self.advance()
        if self.current_char == ".":
            self.advance()
            while self.current_char is not None and self.current_char.isdigit():
                self.advance()
            value = float(self.code[start_pos : self.pos])
        else:
            value = int(self.code[start_pos : self.pos])
        return Token(TokenType.NUMBER, value, self.line, self.column)

    def identifier(self) -> Token:
        start_pos = self.pos
        while (self.current_char is not None) and (self.current_char in ID_VCHAR):
            self.advance()
        value = self.code[start_pos : self.pos]
        token_func = keyword_tokens.get(value, None)
        if token_func is not None:
            return token_func(self, value)
        return Token(TokenType.IDENTIFIER, value, self.line, self.column)
