"""
精确模拟界面流程的诊断脚本 - 100% 复制 app.py 的处理步骤
"""

import sys
import os
import numpy as np
from PIL import Image

# ==================== 配置 ====================
TEST_IMAGE = "test.jpg"  # ← 改成你的图片路径


def simulate_app_flow(image_path, detector):
    """
    100% 模拟 app.py 的图像处理流程
    从 BatchProcessor.process_batch 中提取的逻辑
    """
    print("\n" + "▶"*35)
    print(" 模拟 app.py 的完整流程")
    print("▶"*35)

    # 步骤1: Image.open (app.py:211)
    print(f"\n[步骤1] Image.open + convert('RGB')")
    image = Image.open(image_path).convert("RGB")
    print(f"   ✅ PIL Image: size={image.size}, mode={image.mode}")

    # 步骤2: 缩放 (app.py:217-222) - 假设预处理关闭
    print(f"\n[步骤2] 尺寸缩放 (预处理关闭时)")
    max_size = 640
    if max(image.width, image.height) > max_size:
        ratio = max_size / max(image.width, image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        print(f"   ⚙️ 需要缩放: {image.size} → {new_size}")
        image = image.resize(new_size, Image.LANCZOS)
    else:
        print(f"   ✅ 无需缩放 (尺寸: {image.size})")

    # 步骤3: 转numpy (app.py:225)
    print(f"\n[步骤3] np.array(image)")
    img_np = np.array(image)
    print(f"   ✅ NumPy: shape={img_np.shape}, dtype={img_np.dtype}")
    print(f"   ✅ 像素范围: [{img_np.min()}, {img_np.max()}]")

    # 步骤4: 调用 detect_both (app.py:216)
    print(f"\n[步骤4] detector.detect_both(img_np)")
    combined_img, info = detector.detect_both(img_np)

    return info['missing_count'], img_np


def test_direct_yolo_call(image_path):
    """
    测试方式：直接调用 YOLO（你可能单独测试的方式）
    """
    from ultralytics import YOLO

    print("\n" + "◆"*35)
    print(" 直接调用 YOLO 模型（单独测试方式）")
    print("◆"*35)

    model = YOLO("models/missing_screw_best.pt")

    # 方式1: 直接传路径
    print(f"\n[方式A] model(图片路径)")
    results_a = model(image_path, conf=0.25)[0]
    count_a = len(results_a.boxes) if results_a.boxes is not None else 0
    print(f"   结果: {count_a} 个目标")

    # 方式2: cv2读取后传入
    import cv2
    img_cv = cv2.imread(image_path)
    print(f"\n[方式B] model(cv2.imread结果)")
    results_b = model(img_cv, conf=0.25)[0]
    count_b = len(results_b.boxes) if results_b.boxes is not None else 0
    print(f"   结果: {count_b} 个目标")

    # 方式3: PIL → numpy
    img_pil = Image.open(image_path).convert("RGB")
    img_np = np.array(img_pil)
    print(f"\n[方式C] model(PIL→NumPy)")
    results_c = model(img_np, conf=0.25)[0]
    count_c = len(results_c.boxes) if results_c.boxes is not None else 0
    print(f"   结果: {count_c} 个目标")

    # 输出详细信息
    best_count = max(count_a, count_b, count_c)
    if best_count > 0:
        print(f"\n🎯 最佳结果: {best_count} 个目标")
        # 找到有结果的那一个输出详情
        if count_c > 0 and results_c.boxes is not None:
            for i, box in enumerate(results_c.boxes):
                conf = box.conf[0].cpu().numpy()
                cls_id = int(box.cls[0].cpu().numpy())
                print(f"   [{i+1}] 置信度: {conf:.4f} ({conf*100:.2f}%), 类别ID: {cls_id}")

    return count_a, count_b, count_c


def check_model_info():
    """检查模型信息"""
    print("\n" + "●"*35)
    print(" 检查模型信息")
    print("●"*35)

    try:
        from ultralytics import YOLO

        model = YOLO("models/missing_screw_best.pt")

        print(f"\n📦 模型信息:")
        print(f"   类型: {type(model)}")
        print(f"   任务类型: {model.task}")
        print(f"   类别名称: {model.names}")
        print(f"   类别数量: {len(model.names)}")

        # 检查模型输入尺寸
        if hasattr(model, 'args'):
            print(f"   模型参数: {model.args}")

    except Exception as e:
        print(f"❌ 检查失败: {e}")


def main():
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  🔬 精确诊断工具 - 完全模拟界面流程".center(66) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)

    # 检查文件
    if not os.path.exists(TEST_IMAGE):
        print(f"\n❌ 错误: 找不到图片 '{TEST_IMAGE}'")
        print(f"\n用法: python precise_test.py <图片路径>")
        print(f"示例: python precise_test.py test.jpg")
        return

    if not os.path.exists("models/missing_screw_best.pt"):
        print(f"\n❌ 错误: 找不到模型文件 'models/missing_screw_best.pt'")
        return

    # 1. 检查模型信息
    check_model_info()

    # 2. 初始化检测器
    print("\n" + "■"*35)
    print(" 初始化 Detector")
    print("■"*35)

    from utils.model_loader import Detector
    detector = Detector(
        scratch_path="models/scratch_best.pt",
        missing_path="models/missing_screw_best.pt",
        scratch_conf=0.25,
        missing_conf=0.25
    )

    # 3. 直接调用YOLO测试
    count_a, count_b, count_c = test_direct_yolo_call(TEST_IMAGE)

    # 4. 模拟app流程
    count_app, img_used = simulate_app_flow(TEST_IMAGE, detector)

    # 5. 总结
    print("\n" + "="*70)
    print(" 📊 完整对比总结")
    print("="*70)

    print(f"\n┌────────────────────────────┬────────────┐")
    print(f"│ 测试方式                   │ 检测数量   │")
    print(f"├────────────────────────────┼────────────┤")
    print(f"│ A: YOLO(路径)              │    {count_a:<6}  │")
    print(f"│ B: YOLO(cv2读取)           │    {count_b:<6}  │")
    print(f"│ C: YOLO(PIL→NP)           │    {count_c:<6}  │")
    print(f"│ D: 界面流程(detect_both)   │    {count_app:<6}  │")
    print(f"└────────────────────────────┴────────────┘")

    all_counts = [count_a, count_b, count_c, count_app]
    max_count = max(all_counts)

    print(f"\n{'='*70}")
    if max_count == 0:
        print(" ❌ 所有方式都未检测到！")
        print("\n 💡 关键问题:")
        print("   1. 这个图片在训练集中存在吗？")
        print("   2. 训练时的数据增强设置？")
        print("   3. 模型的类别定义是否正确？")
        print("   4. 图片中的缺陷是否明显？")
        print("\n 📞 请将以下信息发给我:")
        print("   - 这张图片的样子（描述或截图）")
        print("   - 你训练时的命令和配置")
        print("   - 单独测试时的完整代码")
    else:
        print(f" ✅ 最佳检测: {max_count} 个目标")

        if count_c > 0 and count_app == 0:
            print(f"\n 🔍 关键发现!")
            print(f"   单独调用YOLO能检出，但通过Detector检不出")
            print(f"   问题可能在 Detector.detect_both() 内部")

        elif count_a > 0 and count_c == 0:
            print(f"\n 🔍 关键发现!")
            print(f"   直接传路径能检出，但转成NumPy后检不出")
            print(f"   问题在图像格式转换环节")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        TEST_IMAGE = sys.argv[1]

    main()
