import py_compile
import sys

try:
    py_compile.compile('app.py', doraise=True)
    print('[OK] No syntax errors!')
except py_compile.PyCompileError as e:
    print(f'[ERROR] Syntax error found:')
    print(f'  File: {e.filename}')
    print(f'  Line: {e.lineno}')
    print(f'  Error: {e.msg}')
    print(f'  Text: {e.text}')
    sys.exit(1)
