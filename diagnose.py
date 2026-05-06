"""
诊断脚本：对比单独测试 vs 集成测试的差异
用途：定位为什么单独用 missing_screw 模型能检出，但集成到界面就检不出
"""

import cv2
import numpy as np
from PIL import Image
from utils.model_loader import Detector
import os

# ==================== 配置区域 ====================
# 请修改为你要测试的图片路径
TEST_IMAGE_PATH = "test_image.jpg"  # ← 改成你的图片路径

# 模型路径
SCRATCH_MODEL = "models/scratch_best.pt"
MISSING_MODEL = "models/missing_screw_best.pt"

# 测试置信度（先用较低的值）
TEST_CONF = 0.25


def load_image(path):
    """加载并转换图片为多种格式"""
    print(f"\n{'='*60}")
    print(f"📂 加载图片: {path}")
    print(f"{'='*60}")

    # 方式1: PIL Image (Streamlit上传的方式)
    pil_image = Image.open(path).convert("RGB")
    print(f"\n✅ PIL Image:")
    print(f"   - 尺寸: {pil_image.size}")
    print(f"   - 模式: {pil_image.mode}")

    # 方式2: NumPy数组 (直接cv2读取)
    cv_image = cv2.imread(path)
    cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    print(f"\n✅ CV2 Image (RGB):")
    print(f"   - 尺寸: {cv_image_rgb.shape}")
    print(f"   - 数据类型: {cv_image_rgb.dtype}")
    print(f"   - 像素值范围: [{cv_image_rgb.min()}, {cv_image_rgb.max()}]")

    # 方式3: PIL转NumPy (界面的方式)
    pil_to_np = np.array(pil_image)
    print(f"\n✅ PIL → NumPy:")
    print(f"   - 尺寸: {pil_to_np.shape}")
    print(f"   - 数据类型: {pil_to_np.dtype}")
    print(f"   - 像素值范围: [{pil_to_np.min()}, {pil_to_np.max()}]")

    return pil_image, cv_image_rgb, pil_to_np


def test_single_model(model_path, image_np, conf, model_name):
    """单独测试一个模型（模拟你单独测试的情况）"""
    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print(f"🔬 单独测试模型: {model_name}")
    print(f"{'='*60}")
    print(f"📦 模型路径: {model_path}")
    print(f"🎯 置信度阈值: {conf}")
    print(f"📐 输入图像尺寸: {image_np.shape}")
    print(f"📊 输入图像数据类型: {image_np.dtype}")

    model = YOLO(model_path)

    # 直接推理（你单独测试时的方式）
    results = model(image_np, conf=conf)[0]

    boxes = results.boxes
    count = len(boxes) if boxes is not None else 0

    print(f"\n📈检测结果:")
    print(f"   - 检测到目标数: {count}")

    if count > 0 and boxes is not None:
        print(f"\n   详细信息:")
        for i, box in enumerate(boxes):
            xyxy = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            print(f"   [{i+1}] 位置: ({xyxy[0]:.1f}, {xyxy[1]:.1f}) - ({xyxy[2]:.1f}, {xyxy[3]:.1f})")
            print(f"       置信度: {confidence:.4f} ({confidence*100:.2f}%)")
            print(f"       类别ID: {class_id}")
    else:
        print(f"   ⚠️ 未检测到任何目标！")

    return count, results


def test_integrated_detector(detector, image_pil, image_np, conf):
    """通过集成检测器测试（界面的方式）"""
    print(f"\n{'='*60}")
    print(f"🔗 集成测试 (Detector.detect_both)")
    print(f"{'='*60}")
    print(f"🎯 置信度阈值: {conf}")
    print(f"📐 输入图像 (PIL): {image_pil.size}")
    print(f"📐 输入图像 (NP): {image_np.shape}")

    # 设置置信度
    detector.set_scratch_conf(conf)
    detector.set_missing_conf(conf)

    # 调用detect_both（界面的方式）
    combined_img, info = detector.detect_both(image_np)

    print(f"\n📈 集成检测结果:")
    print(f"   - 划痕数量: {info['scratch_count']}")
    print(f"   - 漏装螺丝数量: {info['missing_count']}")

    return info


def compare_differences():
    """主函数：对比差异"""
    print("\n" + "="*70)
    print("🔍 开始诊断：单独测试 vs 集成测试")
    print("="*70)

    # 1. 检查文件是否存在
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"\n❌ 错误：找不到测试图片 '{TEST_IMAGE_PATH}'")
        print(f"请修改脚本中的 TEST_IMAGE_PATH 变量为正确的图片路径")
        return

    if not os.path.exists(MISSING_MODEL):
        print(f"\n❌ 错误：找不到模型文件 '{MISSING_MODEL}'")
        return

    # 2. 加载图片
    pil_img, cv_img_rgb, pil_to_np = load_image(TEST_IMAGE_PATH)

    # 3. 测试1：单独使用 missing_screw 模型（你能检出的方式）
    print("\n\n" + "🔷"*30)
    print("测试 1: 单独调用 missing_screw 模型（你的测试方式）")
    print("🔷"*30)

    missing_count_single, missing_results_single = test_single_model(
        MISSING_MODEL,
        pil_to_np,  # 用PIL转的数组
        TEST_CONF,
        "Missing Screw (单独)"
    )

    # 4. 测试2：通过集成检测器
    print("\n\n" + "🔶"*30)
    print("测试 2: 通过 Detector.detect_both（界面的方式）")
    print("🔶"*30)

    detector = Detector(SCRATCH_MODEL, MISSING_MODEL, scratch_conf=TEST_CONF, missing_conf=TEST_CONF)
    integrated_info = test_integrated_detector(detector, pil_img, pil_to_np, TEST_CONF)

    # 5. 对比结果
    print("\n\n" + "="*70)
    print("📊 结果对比分析")
    print("="*70)

    print(f"\n┌─────────────────────┬──────────────────┐")
    print(f"│ 测试方式            │ 检测到的螺丝缺失数 │")
    print(f"├─────────────────────┼──────────────────┤")
    print(f"│ 单独调用模型        │       {missing_count_single:<10} │")
    print(f"│ 集成 detect_both    │       {integrated_info['missing_count']:<10} │")
    print(f"└─────────────────────┴──────────────────┘")

    if missing_count_single > 0 and integrated_info['missing_count'] == 0:
        print("\n❌ 发现问题！单独能检出，但集成后检不出！")

        print("\n🔍 可能的原因分析:")

        # 检查1: 图像是否被修改
        print("\n   [检查1] detect_both 是否修改了输入图像？")
        print(f"         输入尺寸: {pil_to_np.shape}")

        # 检查2: 模型是否被正确加载
        print("\n   [检查2] 模型加载顺序是否有影响？")
        print(f"         scratch_model: {SCRATCH_MODEL}")
        print(f"         missing_model: {MISSING_MODEL}")

        # 建议
        print("\n💡 建议的解决方案:")
        print("   1. 检查 detect_both 函数中的图像预处理逻辑")
        print("   2. 确认两个模型没有共享状态或冲突")
        print("   3. 尝试在 detect_both 中只调用 missing_model")

    elif missing_count_single == integrated_info['missing_count']:
        print("\n✅ 两种方式结果一致，问题可能在其他地方")

    print("\n" + "="*70)


if __name__ == "__main__":
    # 使用说明
    print("""
╔════════════════════════════════════════════════════════════╗
║                    诊断工具使用说明                          ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  步骤 1: 修改 TEST_IMAGE_PATH 为你要测试的图片路径          ║
║                                                            ║
║  步骤 2: 运行此脚本                                        ║
║          python diagnose.py                                 ║
║                                                            ║
║  步骤 3: 查看输出，对比两种方式的差异                       ║
║                                                            ║
║  步骤 4: 将输出发给我，我会帮你分析具体原因                 ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
    """)

    # 运行诊断
    try:
        compare_differences()
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
