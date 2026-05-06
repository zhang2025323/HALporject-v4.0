"""
终极诊断脚本 - 完整模拟界面的每一步操作
用途：精确定位哪个环节导致检测失败
"""

import sys
import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2

# ==================== 配置 ====================
TEST_IMAGE = "test.jpg"  # ← 改成你的测试图片路径


def print_section(title):
    """打印分隔线"""
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")


def step1_load_original():
    """步骤1: 加载原始图片（模拟文件上传）"""
    print_section("步骤1: 加载原始图片 (模拟文件上传)")

    print(f"\n📂 图片路径: {TEST_IMAGE}")

    # 方式A: cv2读取（你单独测试时可能用的方式）
    img_cv = cv2.imread(TEST_IMAGE)
    if img_cv is None:
        print("❌ 无法用cv2读取")
        return None, None, None
    img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    # 方式B: PIL读取（Streamlit上传的方式）
    try:
        img_pil = Image.open(TEST_IMAGE).convert("RGB")
    except Exception as e:
        print(f"❌ PIL读取失败: {e}")
        return None, None, None

    print(f"\n✅ CV2方式 (RGB):")
    print(f"   - shape: {img_cv_rgb.shape}")
    print(f"   - dtype: {img_cv_rgb.dtype}")
    print(f"   - min/max: [{img_cv_rgb.min()}, {img_cv_rgb.max()}]")

    print(f"\n✅ PIL方式:")
    print(f"   - size: {img_pil.size}")
    print(f"   - mode: {img_pil.mode}")

    return img_cv_rgb, img_pil, np.array(img_pil)


def step2_convert_to_numpy(img_pil):
    """步骤2: PIL转NumPy（界面中的转换）"""
    print_section("步骤2: PIL → NumPy 转换")

    img_np = np.array(img_pil)

    print(f"\n📐 转换结果:")
    print(f"   - shape: {img_np.shape}")
    print(f"   - dtype: {img_np.dtype}")  # 关键！可能是 uint8
    print(f"   - min/max: [{img_np.min()}, {img_np.max()}]")

    return img_np


def step3_preprocess_disabled(img_np):
    """步骤3a: 无预处理（关闭增强时）"""
    print_section("步骤3a: 无预处理 (智能图像增强=关闭)")

    max_size = 640
    width, height = img_np.shape[1], img_np.shape[0]

    if max(width, height) > max_size:
        ratio = max_size / max(width, height)
        new_size = (int(width * ratio), int(height * ratio))
        print(f"\n⚙️ 需要缩放:")
        print(f"   原始尺寸: {width}x{height}")
        print(f"   缩放比例: {ratio:.3f}")
        print(f"   新尺寸: {new_size}")

        pil_img = Image.fromarray(img_np)
        pil_img_resized = pil_img.resize(new_size, Image.LANCZOS)
        img_processed = np.array(pil_img_resized)
    else:
        print(f"\n✅ 无需缩放 (尺寸已符合要求)")
        img_processed = img_np.copy()

    print(f"\n📐 处理后:")
    print(f"   - shape: {img_processed.shape}")
    print(f"   - dtype: {img_processed.dtype}")
    print(f"   - min/max: [{img_processed.min()}, {img_processed.max()}]")

    return img_processed


def step4_test_with_different_inputs(detector, img_original, img_processed):
    """步骤4: 用不同输入测试模型"""
    print_section("步骤4: 对比不同输入的检测结果")

    conf = 0.25  # 用低阈值测试

    print(f"\n🎯 测试置信度: {conf}")

    # 测试A: 原始numpy数组
    print(f"\n--- 测试A: 原始NumPy数组 ---")
    count_a, results_a = detector.test_single_missing(img_original.copy(), conf=conf)

    # 测试B: 处理后的数组
    print(f"\n--- 测试B: 处理后的数组 ---")
    count_b, results_b = detector.test_single_missing(img_processed.copy(), conf=conf)

    # 测试C: 通过detect_both
    print(f"\n--- 测试C: 通过 detect_both ---")
    detector.set_scratch_conf(0.25)
    detector.set_missing_conf(conf)
    _, info_c = detector.detect_both(img_processed.copy())
    count_c = info_c['missing_count']

    # 对比
    print_section("结果对比")
    print(f"\n┌───────────────────────┬────────────┐")
    print(f"│ 输入方式              │ 检测数量   │")
    print(f"├───────────────────────┼────────────┤")
    print(f"│ A: 原始NumPy          │    {count_a:<6}  │")
    print(f"│ B: 处理后NumPy        │    {count_b:<6}  │")
    print(f"│ C: detect_both集成    │    {count_c:<6}  │")
    print(f"└───────────────────────┴────────────┘")

    if count_a > 0 or count_b > 0 or count_c > 0:
        best = max(count_a, count_b, count_c)
        print(f"\n✅ 最佳结果: 检测到 {best} 个目标")
    else:
        print(f"\n❌ 所有方式都未检测到！")

    return count_a, count_b, count_c


def step5_check_image_details(img):
    """步骤5: 检查图像详细信息"""
    print_section("步骤5: 图像详细分析")

    print(f"\n📊 统计信息:")
    print(f"   形状: {img.shape}")
    print(f"   数据类型: {img.dtype}")

    for i, channel in enumerate(['R', 'G', 'B']):
        channel_data = img[:, :, i]
        print(f"\n   {channel}通道:")
        print(f"      最小值: {channel_data.min()}")
        print(f"      最大值: {channel_data.max()}")
        print(f"      平均值: {channel_data.mean():.2f}")
        print(f"      标准差: {channel_data.std():.2f}")

    print(f"\n💡 提示:")
    if img.dtype != np.uint8:
        print(f"   ⚠️ 数据类型不是uint8，可能导致问题！")
    if img.max() > 255 or img.min() < 0:
        print(f"   ⚠️ 像素值超出[0,255]范围！")


def main():
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  🔬 终极诊断工具 - 精确定位检测失败原因".center(66) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)

    # 检查文件
    if not os.path.exists(TEST_IMAGE):
        print(f"\n❌ 错误: 找不到图片 '{TEST_IMAGE}'")
        print(f"请修改脚本第12行的 TEST_IMAGE 变量")
        return

    # 步骤1: 加载图片
    img_cv, img_pil, img_np = step1_load_original()
    if img_np is None:
        return

    # 步骤2: 转换
    img_for_model = step2_convert_to_numpy(img_pil)

    # 步骤5: 详细分析
    step5_check_image_details(img_for_model)

    # 步骤3: 预处理
    img_processed = step3_preprocess_disabled(img_for_model)

    # 导入检测器
    print_section("初始化检测器")
    try:
        from utils.model_loader import Detector
        detector = Detector(
            scratch_path="models/scratch_best.pt",
            missing_path="models/missing_screw_best.pt"
        )
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 步骤4: 对比测试
    count_a, count_b, count_c = step4_test_with_different_inputs(
        detector, img_np, img_processed
    )

    # 最终建议
    print_section("诊断结论与建议")

    if count_a == 0 and count_b == 0 and count_c == 0:
        print("\n❌ 所有测试都失败了！可能的原因:")

        print("\n1️⃣ 模型文件问题:")
        print("   - 模型可能损坏或不兼容")
        print("   - 建议: 重新下载/训练模型")

        print("\n2️⃣ 图片格式问题:")
        print("   - 图片可能不是标准RGB格式")
        print("   - 建议: 将图片转换为PNG/JPG标准格式")

        print("\n3️⃣ 训练/测试不匹配:")
        print("   - 你训练时的预处理和现在不同")
        print("   - 建议: 检查训练代码中的预处理步骤")

        print("\n4️⃣ 类别ID/名称问题:")
        print("   - 模型输出的类别可能不对")
        print("   - 建议: 查看模型的yaml配置文件")

        print("\n💡 下一步操作:")
        print("   1. 运行: python -c \"from ultralytics import YOLO; m=YOLO('models/missing_screw_best.pt'); print(m.names)\"")
        print("   2. 查看模型能识别哪些类别")
        print("   3. 确认你的图片中确实包含这些类别")

    elif count_a > 0 and count_c == 0:
        print("\n🔍 发现问题! 单独调用能检出，但detect_both检不出")
        print("   这说明 detect_both 函数内部有问题")

    elif count_a > 0 and count_b == 0:
        print("\n🔍 发现问题! 原图能检出，但处理后检不出")
        print("   说明预处理步骤改变了关键特征")

    else:
        print(f"\n✅ 至少有一种方式能检出！")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        TEST_IMAGE = sys.argv[1]

    main()
