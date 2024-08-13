## Juniper's Silly Little Language (JSL)

I wanted to learn about programming languages, so I created a very simple toy language using python which tokenizes some text, turns the list of tokens into an AST, then evaluates it. The code is written to be a bit verbose in places for clarity’s sake. It also has been minimally tested.

Language features:
* Three whole types! Numbers, Strings, and Booleans!
* Functions!
* Set and Get Variables!
* If and Else!
* Math!
* True and False!
* Printing!
* While Loops!
* Classes!

Files:
* `tokenizer.py` - The tokenizer code
* `ast_parser.py` - The parser/AST code
* `lang_value.py` - The values/types code
* `lang.py` - The evaluator and code which runs the test code
* `test_code.jsl` - The test code