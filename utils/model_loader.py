import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
import cv2
import numpy as np
import tempfile
import os

# 关键：PyTorch 安全白名单（新版本依然需要）
torch.serialization.add_safe_globals([DetectionModel])

class Detector:
    def __init__(self, scratch_path, missing_path,
                 scratch_conf=0.25, missing_conf=0.25):
        print(f"📦 加载划痕模型: {scratch_path}")
        self.scratch_model = YOLO(scratch_path)

        print(f"📦 加载漏装螺丝模型: {missing_path}")
        self.missing_model = YOLO(missing_path)

        self.scratch_conf = scratch_conf
        self.missing_conf = missing_conf

        print(f"✅ 两个模型加载完成")

    def set_scratch_conf(self, conf):
        self.scratch_conf = conf

    def set_missing_conf(self, conf):
        self.missing_conf = conf

    def _prepare_input_for_yolo(self, image):
        """
        准备YOLO输入 - 根据类型选择最佳方式

        根据诊断结果：
        ✅ 文件路径 → 正常
        ✅ CV2数组 → 正常
        ✅ PIL对象 → 正常
        ❌ PIL→NumPy → 失败（不要用这种方式！）
        """
        # 如果是PIL Image，直接返回（YOLO能处理）
        if hasattr(image, 'save') and hasattr(image, 'size'):
            return image  # 直接传PIL对象

        # 如果是numpy数组，用临时文件方式
        if isinstance(image, np.ndarray):
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"temp_detect_{id(image)}.jpg")

            try:
                # 确保是RGB格式
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # 检查是否是BGR (cv2格式)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image

                cv2.imwrite(temp_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
                return temp_path  # 返回路径，让YOLO自己读取

            except Exception as e:
                print(f"   ⚠️ 创建临时文件失败: {e}")
                return image  # 回退：直接传数组

        # 其他情况直接返回
        return image

    def detect_both(self, image):
        """
        双模型检测（最终修复版）

        关键改进：
        1. 支持PIL对象、numpy数组、文件路径等多种输入
        2. 避免PIL→numpy转换问题
        3. 使用最适合的方式调用YOLO
        """
        print(f"\n{'='*50}")
        print(f"🔍 开始检测")
        print(f"   输入类型: {type(image)}")

        if isinstance(image, np.ndarray):
            print(f"   输入尺寸: {image.shape}")
        elif hasattr(image, 'size'):
            print(f"   输入尺寸: {image.size}")

        print(f"   划痕阈值: {self.scratch_conf}")
        print(f"   漏装阈值: {self.missing_conf}")
        print(f"{'='*50}")

        # 准备输入（关键步骤！）
        input_for_model = self._prepare_input_for_yolo(image)

        # 划痕检测
        print(f"\n[1/2] 执行划痕检测...")
        try:
            scratch_results = self.scratch_model(input_for_model, conf=self.scratch_conf)[0]
            scratch_boxes = scratch_results.boxes
            scratch_count = len(scratch_boxes) if scratch_boxes is not None else 0
            print(f"      ✅ 划痕检测结果: {scratch_count} 个")
        except Exception as e:
            print(f"      ❌ 划痕检测失败: {e}")
            scratch_results = None
            scratch_boxes = None
            scratch_count = 0

        # 漏装螺丝检测
        print(f"\n[2/2] 执行漏装螺丝检测...")
        try:
            missing_results = self.missing_model(input_for_model, conf=self.missing_conf)[0]
            missing_boxes = missing_results.boxes
            missing_count = len(missing_boxes) if missing_boxes is not None else 0
            print(f"      ✅ 漏装螺丝检测结果: {missing_count} 个")

            # 详细输出
            if missing_count > 0 and missing_boxes is not None:
                print(f"\n      📋 漏装螺丝详细信息:")
                for i, box in enumerate(missing_boxes):
                    xyxy = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    print(f"         [{i+1}] 位置: ({xyxy[0]:.1f}, {xyxy[1]:.1f})-({xyxy[2]:.1f}, {xyxy[3]:.1f})")
                    print(f"             置信度: {confidence:.4f} ({confidence*100:.2f}%)")
                    print(f"             类别ID: {class_id}")
        except Exception as e:
            print(f"      ❌ 漏装螺丝检测失败: {e}")
            missing_results = None
            missing_boxes = None
            missing_count = 0

        # 清理临时文件（如果使用了）
        if isinstance(input_for_model, str) and os.path.exists(input_for_model):
            try:
                os.remove(input_for_model)
            except:
                pass

        # 绘制结果
        if scratch_results is not None:
            combined_img = scratch_results.plot()
        else:
            # 从原图像创建画布
            if hasattr(image, 'size'):  # PIL
                combined_img = np.array(image)
            elif isinstance(image, np.ndarray):  # numpy
                combined_img = image.copy()
            else:
                combined_img = np.ones((640, 640, 3), dtype=np.uint8) * 255

        # 绘制漏装螺丝（红色框）
        if missing_boxes is not None:
            boxes = missing_boxes.xyxy.cpu().numpy().astype(int)
            for box in boxes:
                cv2.rectangle(combined_img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                cv2.putText(combined_img, "missing_screw", (box[0], box[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        print(f"\n{'='*50}")
        print(f"🎯 最终结果: 划痕={scratch_count}, 漏装={missing_count}")
        print(f"{'='*50}\n")

        return combined_img, {"scratch_count": scratch_count, "missing_count": missing_count}

    def test_single_missing(self, image, conf=None):
        """单独测试漏装螺丝模型"""
        if conf is None:
            conf = self.missing_conf

        print(f"\n{'='*50}")
        print(f"🔬 单独测试漏装螺丝模型")
        print(f"   输入类型: {type(image)}")
        print(f"   置信度阈值: {conf}")
        print(f"{'='*50}")

        input_for_model = self._prepare_input_for_yolo(image)
        results = self.missing_model(input_for_model, conf=conf)[0]
        boxes = results.boxes
        count = len(boxes) if boxes is not None else 0

        print(f"检测结果: {count} 个目标")

        if count > 0 and boxes is not None:
            for i, box in enumerate(boxes):
                xyxy = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                print(f"  [{i+1}] 置信度: {confidence:.4f} ({confidence*100:.2f}%)")

        # 清理
        if isinstance(input_for_model, str) and os.path.exists(input_for_model):
            try:
                os.remove(input_for_model)
            except:
                pass

        return count, results
