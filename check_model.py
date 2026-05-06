"""
模型信息检查脚本 - 检查模型配置和类别
"""

from ultralytics import YOLO
import os

print("\n" + "="*70)
print("🔍 检查模型文件信息")
print("="*70)

# 检查两个模型
models = {
    "划痕模型": "models/scratch_best.pt",
    "漏装螺丝模型": "models/missing_screw_best.pt"
}

for name, path in models.items():
    print(f"\n{'─'*70}")
    print(f"📦 {name}: {path}")
    print(f"{'─'*70}")

    if not os.path.exists(path):
        print(f"❌ 文件不存在！")
        continue

    try:
        model = YOLO(path)

        print(f"✅ 模型加载成功")
        print(f"\n   基本信息:")
        print(f"   - 任务类型: {model.task}")
        print(f"   - 类别数量: {len(model.names)}")
        print(f"   - 类别名称: {model.names}")

        # 尝试获取更多信息
        if hasattr(model, 'model') and hasattr(model.model, 'names'):
            print(f"   - 内部类别: {model.model.names}")

        if hasattr(model, 'args'):
            print(f"   - 训练参数: {model.args}")

    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*70}\n")

# 现在测试预测（如果用户提供了图片）
import sys
if len(sys.argv) > 1:
    test_image = sys.argv[1]
    print("="*70)
    print(f"🧪 测试预测: {test_image}")
    print("="*70)

    if os.path.exists(test_image):
        for name, path in models.items():
            if os.path.exists(path):
                print(f"\n\n{'━'*50}")
                print(f"测试 {name}")
                print(f"{'━'*50}")

                try:
                    model = YOLO(path)
                    results = model(test_image, conf=0.25, verbose=True)[0]

                    count = len(results.boxes) if results.boxes is not None else 0
                    print(f"\n检测结果: {count} 个目标")

                    if count > 0 and results.boxes is not None:
                        for i, box in enumerate(results.boxes):
                            xyxy = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            cls_id = int(box.cls[0].cpu().numpy())
                            cls_name = model.names[cls_id] if cls_id in model.names else f"class_{cls_id}"
                            print(f"  [{i+1}] {cls_name}: 置信度={conf:.4f}, 位置=({xyxy[0]:.1f},{xyxy[1]:.1f})-({xyxy[2]:.1f},{xyxy[3]:.1f})")

                except Exception as e:
                    print(f"❌ 预测失败: {e}")
                    import traceback
                    traceback.print_exc()
    else:
        print(f"❌ 图片不存在: {test_image}")
else:
    print("💡 用法: python check_model.py <图片路径>")
    print("   示例: python check_model.py test.jpg")
