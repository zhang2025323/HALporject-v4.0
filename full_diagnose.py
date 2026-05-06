"""
全面诊断工具 - 找出为什么训练集图片也检测不到
测试内容：
1. 不同置信度阈值 (0.05 - 0.9)
2. 不同输入方式 (文件路径/PIL/numpy)
3. 图像尺寸影响
4. 模型原始输出分析
"""

import sys
import os
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2


def test_with_different_confidences(model, image_path):
    """测试不同置信度阈值"""
    print("\n" + "▶"*50)
    print(" 测试1: 不同置信度阈值")
    print("▶"*50)

    confidences = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    print(f"\n{'阈值':<8} {'检测数量':<10} {'最高置信度':<12} {'详情'}")
    print("-" * 60)

    best_conf = 0
    best_count = 0
    best_details = []

    for conf in confidences:
        try:
            results = model(image_path, conf=conf, verbose=False)[0]
            boxes = results.boxes
            count = len(boxes) if boxes is not None else 0

            # 获取最高置信度
            max_conf = 0
            details = ""
            if count > 0 and boxes is not None:
                confs = boxes.conf.cpu().numpy()
                max_conf = np.max(confs)
                details = f"范围: {np.min(confs):.3f} - {max_conf:.3f}"

                if count == best_count and conf < best_conf:
                    best_conf = conf
                    best_details = [(box.conf[0].cpu().numpy(), int(box.cls[0].cpu().numpy()))
                                   for box in boxes]

                if count > best_count:
                    best_count = count
                    best_conf = conf
                    best_details = [(box.conf[0].cpu().numpy(), int(box.cls[0].cpu().numpy()))
                                   for box in boxes]

            marker = " ✅" if count > 0 else ""
            print(f"{conf:<8.2f} {count:<10} {max_conf:<12.4f} {details}{marker}")

        except Exception as e:
            print(f"{conf:<8.2f} {'ERROR':<10} {str(e)[:20]}")

    print(f"\n🎯 最佳结果: 阈值={best_conf}, 检测到={best_count}个")
    if best_details:
        print(f"   置信度列表: {[f'{c:.4f}' for c, _ in best_details]}")

    return best_count, best_conf


def test_with_different_inputs(model, image_path):
    """测试不同输入方式"""
    print("\n\n" + "◆"*50)
    print(" 测试2: 不同输入方式 (conf=0.25)")
    print("◆"*50)

    results_dict = {}

    # 方式A: 文件路径
    print(f"\n[A] 文件路径: model('{image_path}')")
    try:
        results_a = model(image_path, conf=0.25, verbose=False)[0]
        count_a = len(results_a.boxes) if results_a.boxes is not None else 0
        results_dict['文件路径'] = (count_a, results_a)
        print(f"   结果: {count_a} 个目标")
    except Exception as e:
        print(f"   ❌ 错误: {e}")
        results_dict['文件路径'] = (0, None)

    # 方式B: cv2读取
    print(f"\n[B] CV2读取: cv2.imread() → model()")
    try:
        img_cv = cv2.imread(image_path)
        results_b = model(img_cv, conf=0.25, verbose=False)[0]
        count_b = len(results_b.boxes) if results_b.boxes is not None else 0
        results_dict['CV2数组'] = (count_b, results_b)
        print(f"   结果: {count_b} 个目标")
    except Exception as e:
        print(f"   ❌ 错误: {e}")
        results_dict['CV2数组'] = (0, None)

    # 方式C: PIL → numpy
    print(f"\n[C] PIL→NumPy: Image.open() → numpy → model()")
    try:
        img_pil = Image.open(image_path).convert("RGB")
        img_np = np.array(img_pil)
        results_c = model(img_np, conf=0.25, verbose=False)[0]
        count_c = len(results_c.boxes) if results_c.boxes is not None else 0
        results_dict['PIL→NP'] = (count_c, results_c)
        print(f"   结果: {count_c} 个目标")
    except Exception as e:
        print(f"   ❌ 错误: {e}")
        results_dict['PIL→NP'] = (0, None)

    # 方式D: PIL直接传入
    print(f"\n[D] PIL对象: Image.open() → model()")
    try:
        img_pil = Image.open(image_path).convert("RGB")
        results_d = model(img_pil, conf=0.25, verbose=False)[0]
        count_d = len(results_d.boxes) if results_d.boxes is not None else 0
        results_dict['PIL对象'] = (count_d, results_d)
        print(f"   结果: {count_d} 个目标")
    except Exception as e:
        print(f"   ❌ 错误: {e}")
        results_dict['PIL对象'] = (0, None)

    # 对比
    print(f"\n{'='*50}")
    print(f" 对比总结:")
    print(f"{'='*50}")
    for method, (count, _) in results_dict.items():
        status = "✅" if count > 0 else "❌"
        print(f"   {status} {method:<12}: {count} 个目标")

    return results_dict


def test_image_sizes(model, image_path):
    """测试不同图像尺寸的影响"""
    print("\n\n" + "●"*50)
    print(" 测试3: 不同图像尺寸 (conf=0.25)")
    print("●"*50)

    sizes = [
        ("原始尺寸", None),
        ("320x320", 320),
        ("640x640", 640),
        ("1280x1280", 1280),
    ]

    results_size = {}

    for name, size in sizes:
        print(f"\n[{name}] ", end="")

        try:
            if size is None:
                results = model(image_path, conf=0.25, verbose=False)[0]
            else:
                img = Image.open(image_path).convert("RGB")
                img_resized = img.resize((size, size), Image.LANCZOS)
                results = model(img_resized, conf=0.25, verbose=False)[0]

            count = len(results.boxes) if results.boxes is not None else 0
            results_size[name] = count
            print(f"结果: {count} 个目标")

        except Exception as e:
            print(f"❌ 错误: {e}")
            results_size[name] = 0

    return results_size


def analyze_raw_output(model, image_path):
    """分析模型的原始输出（不过滤）"""
    print("\n\n" + "■"*50)
    print(" 测试4: 原始输出分析 (无过滤)")
    print("■"*50)

    try:
        # 使用非常低的阈值获取所有预测
        results = model(image_path, conf=0.01, verbose=True)[0]

        print(f"\n📊 原始预测结果 (conf>0.01):")

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes

            # 按置信度排序
            confs = boxes.conf.cpu().numpy()
            sorted_indices = np.argsort(-confs)

            print(f"\n 总共 {len(boxes)} 个预测框:")
            print(f"{'序号':<6}{'置信度':<10}{'类别ID':<8}{'是否>0.25':<10}")
            print("-" * 40)

            above_25 = 0
            above_10 = 0
            above_05 = 0

            for rank, idx in enumerate(sorted_indices[:20]):  # 只显示前20个
                conf = confs[idx]
                cls_id = int(boxes.cls[idx].cpu().numpy())

                above_25 += 1 if conf >= 0.25 else 0
                above_10 += 1 if conf >= 0.10 else 0
                above_05 += 1 if conf >= 0.05 else 0

                marker = "✅" if conf >= 0.25 else ("⚠️" if conf >= 0.10 else "❌")
                print(f"{rank+1:<6}{conf:<10.4f}{cls_id:<8}{marker:<10}")

            print(f"\n📈 统计:")
            print(f"   conf ≥ 0.25: {above_25} 个")
            print(f"   conf ≥ 0.10: {above_10} 个")
            print(f"   conf ≥ 0.05: {above_05} 个")
            print(f"   conf < 0.05: {len(boxes) - above_05} 个")

            return len(boxes), above_25, above_10
        else:
            print(f"\n⚠️ 无任何预测！（即使阈值降到0.01也没有）")
            return 0, 0, 0

    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0, 0


def main():
    print("\n" + "█"*70)
    print("█" + " "*10 + "🔬 全面诊断工具 - 为什么检测不到？".center(46) + " "*10 + "█")
    print("█"*70)

    # 获取图片路径
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "test.jpg"

    print(f"\n📂 测试图片: {image_path}")

    if not os.path.exists(image_path):
        print(f"\n❌ 图片不存在!")
        print(f"用法: python full_diagnose.py <图片路径>")
        return

    if not os.path.exists("models/missing_screw_best.pt"):
        print(f"\n❌ 模型不存在!")
        return

    # 加载模型
    print(f"\n🤖 加载模型...")
    model = YOLO("models/missing_screw_best.pt")
    print(f"✅ 模型加载成功")
    print(f"   类别: {model.names}")

    # 运行所有测试
    count1, best_conf = test_with_different_confidences(model, image_path)
    results_input = test_with_different_inputs(model, image_path)
    results_size = test_image_sizes(model, image_path)
    total_preds, above_25, above_10 = analyze_raw_output(model, image_path)

    # 最终结论
    print("\n\n" + "="*70)
    print(" 📋 最终诊断报告")
    print("="*70)

    print(f"\n1️⃣ 阈值测试:")
    print(f"   最佳阈值: {best_conf}, 最大检出数: {count1}")

    print(f"\n2️⃣ 输入方式:")
    best_method = max(results_input.items(), key=lambda x: x[1][0])
    print(f"   最佳方式: {best_method[0]} ({best_method[1][0]} 个)")

    print(f"\n3️⃣ 原始预测:")
    print(f"   总预测数: {total_preds}")
    print(f"   conf≥0.25: {above_25}")
    print(f"   conf≥0.10: {above_10}")

    # 给出建议
    print(f"\n\n{'='*70}")
    print(" 💡 诊断结论与建议")
    print("="*70)

    if total_preds == 0:
        print(f"""
   ❌ 严重问题: 模型对这张图片没有任何预测！

   可能原因:
   1. 图片与训练数据差异太大（光照/角度/背景）
   2. 模型训练不充分（epochs太少）
   3. 数据标注有问题
   4. 模型文件损坏或不匹配

   建议:
   - 用训练集中的一张原图测试（不是增强后的）
   - 检查训练时的数据增强设置
   - 重新训练并增加 epochs
""")

    elif above_25 == 0 and total_preds > 0:
        print(f"""
   ⚠️ 置信度偏低: 有{total_preds}个预测但都低于0.25

   这说明模型检测到了一些东西，但不确定
   原因可能是:
   1. 图片质量不够好（模糊/噪声）
   2. 目标特征不明显
   3. 模型泛化能力不足

   解决方案:
   - 降低阈值到 0.10 或 0.05 试试
   - 改善图片质量（更清晰的光线）
   - 收集更多类似图片重新训练
""")

    elif above_25 > 0 and above_25 < total_preds:
        print(f"""
   ✅ 正常情况: 能检测到{above_25}个高置信度目标
   
   但有{total_preds - above_25}个低置信度预测被过滤了
   这是正常的，说明模型工作正常
""")

    elif above_25 > 0:
        print(f"""
   ✅ 完美: 检测到{above_25}个目标，模型工作正常！
   
   如果界面中还是检测不到，问题在代码集成部分
""")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
