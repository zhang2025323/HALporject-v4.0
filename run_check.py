import subprocess
import sys

result = subprocess.run(
    [sys.executable, 'app.py'],
    capture_output=True,
    text=True,
    timeout=10
)

print("=== STDOUT ===")
print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)

print("\n=== STDERR ===")
print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)

print(f"\n=== Exit Code: {result.returncode} ===")
