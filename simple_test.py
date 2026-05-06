"""
超级简单测试 - 100% 原生YOLO调用，不经过任何封装
"""

import sys
from ultralytics import YOLO
import os

print("\n" + "█"*70)
print("█" + " "*15 + "🔬 最简测试 - 纯原生YOLO调用" + " "*18 + "█")
print("█"*70)

# 配置
if len(sys.argv) > 1:
    TEST_IMAGE = sys.argv[1]
else:
    TEST_IMAGE = "test.jpg"  # ← 改成你的图片

print(f"\n📂 测试图片: {TEST_IMAGE}")

if not os.path.exists(TEST_IMAGE):
    print(f"❌ 图片不存在！")
    print(f"\n用法: python simple_test.py <图片路径>")
    sys.exit(1)

if not os.path.exists("models/missing_screw_best.pt"):
    print(f"❌ 模型文件不存在！")
    sys.exit(1)

# 第一步：加载模型（纯原生方式）
print(f"\n[步骤1] 加载漏装螺丝模型...")
model = YOLO("models/missing_screw_best.pt")
print(f"   ✅ 模型加载成功")
print(f"   📦 类别: {model.names}")

# 第二步：直接预测（就像你单独测试时那样）
print(f"\n[步骤2] 直接预测 (model(图片路径, conf=0.25))")
print(f"   这是你单独测试时的代码方式")

try:
    results = model(TEST_IMAGE, conf=0.25)[0]

    boxes = results.boxes
    count = len(boxes) if boxes is not None else 0

    print(f"\n{'='*50}")
    print(f"🎯 检测结果: {count} 个目标")
    print(f"{'='*50}")

    if count > 0:
        print(f"\n✅ 成功检测到！详细信息:")
        for i, box in enumerate(boxes):
            xyxy = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = model.names.get(cls_id, f"class_{cls_id}")

            print(f"\n  目标 [{i+1}]:")
            print(f"    类别: {cls_name} (ID: {cls_id})")
            print(f"    置信度: {conf:.4f} ({conf*100:.2f}%)")
            print(f"    位置: ({xyxy[0]:.1f}, {xyxy[1]:.1f}) - ({xyxy[2]:.1f}, {xyxy[3]:.1f})")

        print(f"\n{'='*50}")
        print(f"✅ 结论: 模型本身能正常工作！")
        print(f"❓ 问题可能在: Detector类的封装或图像预处理")
        print(f"{'='*50}")

    else:
        print(f"\n⚠️ 未检测到任何目标！")
        print(f"\n可能原因:")
        print(f"  1. 图片中确实没有螺丝缺失（或太不明显）")
        print(f"  2. 训练数据和这张图片差异太大")
        print(f"  3. 模型需要重新训练")
        print(f"\n建议:")
        print(f"  - 用训练集中的图片测试看能否检出")
        print(f"  - 降低阈值到 0.1 或 0.05 再试")

except Exception as e:
    print(f"\n❌ 预测出错: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'█'*70}\n")
