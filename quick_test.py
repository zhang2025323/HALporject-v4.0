"""
快速验证脚本 - 测试终极修复是否有效
"""

import sys
from PIL import Image
import numpy as np

TEST_IMAGE = "test.jpg"  # ← 改成你的图片路径


def main():
    print("\n" + "="*70)
    print("� 验证终极修复 - 文件路径模式检测")
    print("="*70)

    try:
        from utils.model_loader import Detector
        print("\n✅ 导入成功")
    except Exception as e:
        print(f"\n❌ 导入失败: {e}")
        return

    if len(sys.argv) > 1:
        global TEST_IMAGE
        TEST_IMAGE = sys.argv[1]

    import os
    if not os.path.exists(TEST_IMAGE):
        print(f"\n❌ 找不到图片: {TEST_IMAGE}")
        print(f"\n用法: python quick_test.py <图片路径>")
        return

    # 加载图片（模拟界面流程）
    print(f"\n📂 加载图片: {TEST_IMAGE}")
    image_pil = Image.open(TEST_IMAGE).convert("RGB")
    image_np = np.array(image_pil)
    print(f"   尺寸: {image_np.shape}")

    # 初始化检测器
    print(f"\n🤖 初始化检测器...")
    detector = Detector(
        scratch_path="models/scratch_best.pt",
        missing_path="models/missing_screw_best.pt",
        scratch_conf=0.25,
        missing_conf=0.25  # 低阈值
    )

    # 测试
    print(f"\n{'🔷'*30}")
    print(f"测试: 通过 detect_both (新版本)")
    print(f"{'�'*30}")

    _, info = detector.detect_both(image_np)

    count = info['missing_count']

    print(f"\n{'='*70}")
    if count > 0:
        print(f"✅ 成功！检测到 {count} 个螺丝缺失！")
        print(f"\n🎉 问题已解决！现在可以在界面中使用了！")
        print(f"\n下一步:")
        print(f"   1. 运行 streamlit run app.py")
        print(f"   2. 上传这张图片测试")
        print(f"   3. 确认能正常检测")
    else:
        print(f"❌ 还是未检测到")
        print(f"\n请运行详细诊断:")
        print(f"   python precise_test.py {TEST_IMAGE}")
        print(f"\n并将输出发给我分析")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
