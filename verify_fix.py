"""
验证修复效果 - 模拟界面的完整流程
"""

import sys
from PIL import Image
from utils.model_loader import Detector

# 配置
TEST_IMAGE = "C:\\Users\\TXL19\\Desktop\\工件图片材料\\ls_640\\333.jpg"


def main():
    print("\n" + "="*70)
    print("🚀 验证修复 - 模拟界面完整流程")
    print("="*70)

    # 步骤1: 模拟界面的图片加载方式
    print(f"\n[步骤1] 模拟 Image.open() (界面中的方式)")
    image = Image.open(TEST_IMAGE).convert("RGB")
    print(f"   ✅ 加载成功: {image.size}, mode={image.mode}")

    # 步骤2: 模拟缩放（如果需要）
    max_size = 640
    if max(image.width, image.height) > max_size:
        ratio = max_size / max(image.width, image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        print(f"\n[步骤2] 缩放到 {new_size}")
        image = image.resize(new_size, Image.LANCZOS)
    else:
        print(f"\n[步骤2] 无需缩放")

    # 步骤3: 初始化检测器
    print(f"\n[步骤3] 初始化检测器")
    detector = Detector(
        scratch_path="models/scratch_best.pt",
        missing_path="models/missing_screw_best.pt",
        scratch_conf=0.25,
        missing_conf=0.25
    )

    # 步骤4: 直接传PIL对象（关键！不转numpy）
    print(f"\n[步骤4] 调用 detect_both(image) - 直接传PIL对象")
    print(f"   ⚠️ 注意：这里不再使用 np.array(image)")
    _, info = detector.detect_both(image)

    # 结果
    count = info['missing_count']

    print(f"\n{'='*70}")
    if count > 0:
        print(f"✅✅✅ 成功！检测到 {count} 个螺丝缺失！")
        print(f"\n🎉 修复有效！问题已解决！")
        print(f"\n下一步:")
        print(f"   1. 运行 streamlit run app.py")
        print(f"   2. 上传这张图片测试")
        print(f"   3. 应该能正常检测了！")
    else:
        print(f"❌ 还是未检测到")
        print(f"\n请检查控制台输出的详细信息")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        TEST_IMAGE = sys.argv[1]

    main()
