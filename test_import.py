import sys
import traceback

sys.path.insert(0, r'c:\HALporject\HALporject')

print("=" * 60)
print("开始测试 app.py 导入...")
print("=" * 60)

try:
    print("\n[1] 正在导入模块...")
    import app
    print("[✓] 模块导入成功!")
    
except SyntaxError as e:
    print(f"\n[✗] 语法错误:")
    print(f"    文件: {e.filename}")
    print(f"    行号: {e.lineno}")
    print(f"    位置: {e.offset}")
    print(f"    内容: {e.text}")
    traceback.print_exc()
    
except ImportError as e:
    print(f"\n[✗] 导入错误: {e}")
    traceback.print_exc()
    
except Exception as e:
    print(f"\n[✗] 运行时错误: {type(e).__name__}: {e}")
    print("\n完整错误堆栈:")
    traceback.print_exc()

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
