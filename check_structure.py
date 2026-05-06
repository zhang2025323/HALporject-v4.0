import sys
import os

os.chdir(r'c:\HALporject\HALporject')
sys.path.insert(0, r'c:\HALporject\HALporject')

print("Checking Python syntax...")
import py_compile
try:
    py_compile.compile(r'c:\HALporject\HALporject\app.py', doraise=True)
    print("[OK] Syntax check passed")
except py_compile.PyCompileError as e:
    print(f"[ERROR] Syntax error: {e}")
    sys.exit(1)

print("\nChecking string and bracket balance...")
with open(r'c:\HALporject\HALporject\app.py', 'r', encoding='utf-8') as f:
    content = f.read()
    
triple_double = content.count('"""')
triple_single = content.count("'''")
print(f"Triple double quotes: {triple_double} (should be even)")
print(f"Triple single quotes: {triple_single} (should be even)")

if triple_double % 2 != 0:
    print("[ERROR] Triple double quotes not closed!")
elif triple_single % 2 != 0:
    print("[ERROR] Triple single quotes not closed!")
else:
    print("[OK] All strings properly closed")

print("\nChecking bracket balance...")
stack = []
brackets = {'(': ')', '[': ']', '{': '}'}
line_num = 1
for i, char in enumerate(content):
    if char == '\n':
        line_num += 1
    if char in brackets:
        stack.append((char, line_num))
    elif char in brackets.values():
        if stack:
            expected = stack[-1][0]
            if brackets[expected] != char:
                print(f"[ERROR] Bracket mismatch at line {line_num}: expected {brackets[expected]}, found {char}")
                break
            stack.pop()

if not stack:
    print("[OK] All brackets properly closed")
else:
    print(f"[ERROR] {len(stack)} unclosed brackets:")
    for char, pos in stack[-5:]:
        print(f"  - '{char}' at line {pos}")

print("\nBasic checks completed!")
