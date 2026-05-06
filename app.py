import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import time
import pandas as pd
from io import BytesIO
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import requests
from datetime import datetime, timedelta
from utils.model_loader import Detector

# MES 配置
MES_URL = "http://8.156.84.27:9091"
MES_USERNAME = "admin"
MES_PASSWORD = "admin123"

class MESConnector:
    def __init__(self):
        self.session = requests.Session()
        self.token = None
        self.token_expiry = None

    def login_and_get_token(self):
        """登录获取Token"""
        try:
            if self.token and self.token_expiry and datetime.now() < self.token_expiry:
                return self.token

            print(f"🔐 正在登录 MES: {MES_URL}/api/login")
            response = self.session.post(
                f"{MES_URL}/api/login",
                json={
                    "username": MES_USERNAME,
                    "password": MES_PASSWORD
                },
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                self.token = data.get("token")
                # Token 有效期设为23小时（比实际的24小时短一点）
                self.token_expiry = datetime.now() + timedelta(hours=23)
                print(f"✅ 登录成功，获取到Token")
                return self.token
            else:
                print(f"❌ 登录失败: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            print(f"❌ 登录异常: {e}")
            return None

    def send_detection_result(self, result):
        try:
            # 先尝试不带Token调用 /aiDetect（如果Security配置生效了）
            data = {
                "fileName": result.get("文件名"),
                "scratchCount": result.get("划痕数量", 0),
                "missingCount": result.get("漏装螺丝数量", 0),
                "detectionTime": result.get("检测耗时(ms)", 0),
                "productId": result.get("产品ID", ""),
                "imageUrl": ""
            }

            print(f"🚀 正在发送数据到 MES (尝试1: 无Token)")
            print(f"   URL: {MES_URL}/mes/api/aiDetect")

            response = self.session.post(
                f"{MES_URL}/mes/api/aiDetect",
                json=data,
                timeout=10
            )

            print(f"   响应状态码: {response.status_code}")

            if response.status_code == 200:
                print(f"✅ 检测结果已发送到MES (无需Token): {result.get('文件名')}")
                return True

            # 如果返回401或404，尝试带Token
            if response.status_code in [401, 404]:
                print(f"⚠️ 尝试失败，切换到带Token模式...")

                token = self.login_and_get_token()
                if not token:
                    print(f"❌ 无法获取Token，尝试备用方案...")
                    return self.send_via_addimage(result)

                headers = {"Authorization": f"Bearer {token}"}
                response = self.session.post(
                    f"{MES_URL}/mes/api/aiDetect",
                    json=data,
                    headers=headers,
                    timeout=10
                )

                print(f"   带Token响应状态码: {response.status_code}")

                if response.status_code == 200:
                    print(f"✅ 检测结果已发送到MES (带Token): {result.get('文件名')}")
                    return True

                # 如果还是404，使用备用方案
                if response.status_code == 404:
                    print(f"⚠️ 新端点不可用，使用备用方案...")
                    return self.send_via_addimage(result)

            print(f"❌ 发送失败: {response.status_code} - {response.text}")
            return False

        except Exception as e:
            print(f"❌ 发送异常: {e}")
            import traceback
            print(f"   详细错误信息: {traceback.format_exc()}")
            return False

    def send_via_addimage(self, result):
        """备用方案：通过 /addImage 发送"""
        try:
            import base64
            import json

            detection_data = {
                "fileName": result.get("文件名"),
                "scratchCount": result.get("划痕数量", 0),
                "missingCount": result.get("漏装螺丝数量", 0),
                "detectionTime": result.get("检测耗时(ms)", 0),
                "productId": result.get("产品ID", "")
            }

            data = {
                "img": "",  # 空图片
                "address": f"AI_DETECT:{json.dumps(detection_data)}"  # 将检测数据嵌入address字段
            }

            print(f"🔄 使用备用方案: {MES_URL}/mes/api/addImage")
            print(f"   数据嵌入address字段")

            response = self.session.post(
                f"{MES_URL}/mes/api/addImage",
                json=data,
                timeout=10
            )

            if response.status_code == 200:
                print(f"✅ 检测结果已通过备用方案发送: {result.get('文件名')}")
                return True
            else:
                print(f"❌ 备用方案也失败: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ 备用方案异常: {e}")
            return False

# 全局MES连接器实例
mes_connector = MESConnector()

# 获取项目根目录
BASE_DIR = Path(__file__).parent

# PDF 生成函数（支持中文）
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# 注册中文字体
try:
    pdfmetrics.registerFont(TTFont('MicrosoftYaHei', 'C:/Windows/Fonts/msyh.ttc'))
    FONT_NAME = 'MicrosoftYaHei'
except:
    try:
        pdfmetrics.registerFont(TTFont('SimSun', 'C:/Windows/Fonts/simsun.ttc'))
        FONT_NAME = 'SimSun'
    except:
        FONT_NAME = 'Helvetica'

def generate_pdf_report(records):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('CustomTitle', parent=styles['Title'], fontName=FONT_NAME, fontSize=16, alignment=1)
    normal_style = ParagraphStyle('CustomNormal', parent=styles['Normal'], fontName=FONT_NAME, fontSize=10)
    
    title = Paragraph("工件缺陷检测报告", title_style)
    elements.append(title)
    elements.append(Spacer(1, 12))
    
    data = [["文件名", "检测时间", "划痕数量", "漏装数量", "耗时(ms)"]]
    for r in records:
        data.append([r["文件名"], r["检测时间"], str(r["划痕数量"]), str(r["漏装螺丝数量"]), str(r["检测耗时(ms)"])])
    
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), FONT_NAME),
        ('FONTNAME', (0,1), (-1,-1), FONT_NAME),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
    ]))
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ==================== 图像预处理器（针对手机拍照优化）====================
class SmartImagePreprocessor:
    """智能图像预处理器 - 应对手机拍照的各种环境问题"""
    
    @staticmethod
    def preprocess_for_mobile(image: Image.Image) -> Image.Image:
        """
        对手机拍摄的图片进行智能预处理：
        1. 自动旋转（根据EXIF信息）
        2. 光线校正（自动亮度/对比度）
        3. 锐化增强（提升细节）
        4. 噪声抑制（低光环境）
        5. 尺寸标准化
        """
        # 1. 自动旋转（处理手机竖拍照片）
        image = SmartImagePreprocessor._auto_rotate(image)
        
        # 2. 光线分析与校正
        image = SmartImagePreprocessor._enhance_lighting(image)
        
        # 3. 质量增强
        image = SmartImagePreprocessor._enhance_quality(image)
        
        # 4. 尺寸标准化（保持宽高比）
        image = SmartImagePreprocessor._resize_smart(image, max_size=640)
        
        return image
    
    @staticmethod
    def _auto_rotate(image: Image.Image) -> Image.Image:
        """根据EXIF信息自动旋转图片"""
        try:
            if hasattr(image, '_getexif'):
                exif = image._getexif()
                if exif is not None:
                    orientation = exif.get(274)  # Orientation tag
                    rotation_map = {
                        3: Image.ROTATE_180,
                        6: Image.ROTATE_270,
                        8: Image.ROTATE_90,
                    }
                    if orientation in rotation_map:
                        image = image.transpose(rotation_map[orientation])
        except:
            pass
        return image
    
    @staticmethod
    def _enhance_lighting(image: Image.Image) -> Image.Image:
        """智能光线校正 - 适应过暗/过亮/逆光等场景"""
        img_array = np.array(image)
        
        # 计算当前亮度
        brightness = np.mean(img_array)
        
        # 动态调整参数
        if brightness < 100:  # 过暗（夜景/室内弱光）
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.5)  # 提亮50%
            
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)  # 增强对比度
            
            # 轻微降噪
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
        elif brightness > 180:  # 过亮（强光/逆光）
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(0.85)  # 降低亮度
            
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.15)
            
        else:  # 正常光线 - 轻微优化
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.08)
            
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
        
        return image
    
    @staticmethod
    def _enhance_quality(image: Image.Image) -> Image.Image:
        """通用质量增强"""
        # 锐化（提升边缘清晰度）
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.15)
        
        # 色彩饱和度微调（让颜色更鲜明）
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.05)
        
        return image
    
    @staticmethod
    def _resize_smart(image: Image.Image, max_size: int = 640) -> Image.Image:
        """智能缩放 - 保持宽高比，避免变形"""
        width, height = image.size
        
        if max(width, height) <= max_size:
            return image
        
        ratio = max_size / max(width, height)
        new_size = (int(width * ratio), int(height * ratio))
        
        # 使用高质量重采样
        image = image.resize(new_size, Image.LANCZOS)
        
        return image


# ==================== 批量处理器（支持30+张稳定运行）====================
class BatchProcessor:
    """高效批量处理器 - 内存管理 + 错误恢复 + 进度追踪"""
    
    def __init__(self, detector, max_batch_size=30, enable_preprocess=False):
        self.detector = detector
        self.max_batch_size = max_batch_size
        self.enable_preprocess = enable_preprocess
        self.preprocessor = SmartImagePreprocessor()
        
    def process_batch(self, files_list, progress_callback=None):
        """
        批量处理图片列表
        - 自动内存管理
        - 错误隔离（单张失败不影响其他）
        - 进度回调
        """
        results = []
        errors = []
        total = len(files_list)
        
        for idx, uploaded_file in enumerate(files_list[:self.max_batch_size]):
            file_key = uploaded_file.name
            current_progress = ((idx + 1) / total) * 100
            
            try:
                # 更新进度
                if progress_callback:
                    progress_callback(idx + 1, total, f"正在检测: {file_key}")
                
                # 读取图片
                image = Image.open(uploaded_file).convert("RGB")

                # 根据设置决定是否进行智能预处理
                if self.enable_preprocess:
                    image = self.preprocessor.preprocess_for_mobile(image)
                else:
                    # 仅做基本的尺寸缩放（不超过640）
                    max_size = 640
                    if max(image.width, image.height) > max_size:
                        ratio = max_size / max(image.width, image.height)
                        new_size = (int(image.width * ratio), int(image.height * ratio))
                        image = image.resize(new_size, Image.LANCZOS)

                # 执行检测 - 直接传PIL对象，不转numpy（避免转换问题）
                start_time = time.time()
                combined_img, info = self.detector.detect_both(image)
                inference_time = time.time() - start_time
                
                # 记录结果
                result = {
                    'file_key': file_key,
                    'image': image.copy(),  # 处理后的图片
                    'combined_img': combined_img,
                    'info': info,
                    'inference_time': inference_time,
                    'record': {
                        "文件名": file_key,
                        "检测时间": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "划痕数量": info['scratch_count'],
                        "漏装螺丝数量": info['missing_count'],
                        "检测耗时(ms)": round(inference_time * 1000, 1)
                    }
                }
                results.append(result)

                # 及时释放内存
                del image, combined_img
                
            except Exception as e:
                error_info = {
                    'file_key': file_key,
                    'error': str(e),
                    'index': idx + 1
                }
                errors.append(error_info)
                
                # 继续处理下一张
                continue
        
        return results, errors


# ==================== 页面配置 ====================
st.set_page_config(
    page_title="智能质检系统 - MES团队",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化 session_state
if "detection_records" not in st.session_state:
    st.session_state.detection_records = []
if "detection_cache" not in st.session_state:
    st.session_state.detection_cache = {}
if "all_uploaded_files_persistent" not in st.session_state:
    st.session_state.all_uploaded_files_persistent = []
if "deleted_files" not in st.session_state:
    st.session_state.deleted_files = set()
# 新增：用于标记是否刚执行了清除操作，防止rerun时重复添加
if "just_cleared" not in st.session_state:
    st.session_state.just_cleared = False
# 新增：上传器版本号（用于强制重置文件列表）
if "uploader_key_version" not in st.session_state:
    st.session_state.uploader_key_version = 0
# 新增：自动上传相关状态
if "auto_upload_enabled" not in st.session_state:
    st.session_state.auto_upload_enabled = True
if "auto_uploaded_files" not in st.session_state:
    st.session_state.auto_uploaded_files = set()

# ==================== 移动端优化的CSS ====================
mobile_optimized_css = """
.stApp {
    background: linear-gradient(135deg, #f0f4fc 0%, #d9e2ef 100%);
    background-attachment: fixed;
}

/* 移动端触摸优化 */
@media (max-width: 768px) {
    .stButton button {
        min-height: 48px !important;  /* Apple推荐的最小触控区域 */
        font-size: 16px !important;   /* 防止iOS自动缩放 */
    }
    
    [data-testid="stFileUploadWrapper"] {
        padding: 1.5rem !important;
    }
    
    /* 结果卡片移动端优化 */
    .card {
        margin-bottom: 1.5rem !important;
        padding: 1rem !important;
    }
    
    /* 统计面板移动端 */
    .stats-grid {
        grid-template-columns: repeat(2, 1fr) !important;
        gap: 0.8rem !important;
    }
    
    .stat-item {
        padding: 0.8rem !important;
    }
    
    .stat-value {
        font-size: 1.5rem !important;
    }
}

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.css-1d391kg, .css-1lcbmhc {
    background: rgba(255,255,255,0.9);
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(0,0,0,0.05);
}
.card {
    background: white;
    border-radius: 20px;
    padding: 1.5rem;
    box-shadow: 0 10px 25px -5px rgba(0,0,0,0.05), 0 8px 10px -6px rgba(0,0,0,0.02);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    margin-bottom: 1rem;
    border: 1px solid rgba(255,255,255,0.3);
}
.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 20px 25px -12px rgba(0,0,0,0.1);
}
.metric-card {
    background: white;
    border-radius: 24px;
    padding: 1rem 1.5rem;
    text-align: center;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    border: 1px solid #e5e7eb;
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #1e3c72;
    line-height: 1.2;
}
.metric-label {
    font-size: 0.85rem;
    color: #6b7280;
    letter-spacing: 0.5px;
}
.stButton button {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    color: white;
    border: none;
    border-radius: 40px;
    padding: 0.5rem 2rem;
    font-weight: 500;
    transition: all 0.2s;
}
.stButton button:hover {
    transform: scale(1.02);
    box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
}

/* Scroll to Top Button */
.scroll-top-btn {
    position: fixed;
    bottom: 30px;
    right: 30px;
    width: 50px;
    height: 50px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 50%;
    font-size: 1.5rem;
    cursor: pointer;
    z-index: 9998;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    transition: all 0.3s ease;
    display: none;
    align-items: center;
    justify-content: center;
}
.scroll-top-btn:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
}
.scroll-top-btn.show {
    display: flex;
}

/* Statistics Panel */
.stats-panel {
    background: linear-gradient(135deg, rgba(30, 60, 114, 0.05), rgba(42, 82, 152, 0.05));
    border: 2px solid rgba(30, 60, 114, 0.1);
    border-radius: 20px;
    padding: 1.5rem;
    margin: 1.5rem 0;
}
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
}
.stat-item {
    text-align: center;
    padding: 1rem;
    background: white;
    border-radius: 15px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}
.stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: #1e3c72;
}
.stat-label {
    font-size: 0.85rem;
    color: #6b7280;
    margin-top: 0.3rem;
}

/* 进度条样式 */
.batch-progress-container {
    padding: 1.5rem;
    background: linear-gradient(135deg, #667eea10, #764ba210);
    border-radius: 15px;
    border: 1px solid #667eea30;
    margin: 1rem 0;
}
"""

st.markdown(f"<style>{mobile_optimized_css}</style>", unsafe_allow_html=True)

# ==================== JavaScript ====================
force_hide_js_v2 = """
<script>
function initDeleteSystem() {
    console.log('[DELETE-SYS] Initializing...');
    bindDeleteButtons();
    
    var observer = new MutationObserver(function() {
        bindDeleteButtons();
    });
    observer.observe(document.body, {childList: true, subtree: true, attributes: false});
    
    setInterval(bindDeleteButtons, 300);
    setTimeout(bindDeleteButtons, 500);
    setTimeout(bindDeleteButtons, 1000);
    setTimeout(bindDeleteButtons, 2000);
}

function bindDeleteButtons() {
    var allBtns = document.querySelectorAll('button');
    
    allBtns.forEach(function(btn) {
        if (btn.dataset.deleteBound) return;
        
        var parent = btn.closest('div') || btn.parentElement;
        var grandParent = parent ? parent.parentElement : null;
        var text = (parent.textContent || '') + (grandParent ? grandParent.textContent : '');
        
        var isRemoveBtn = (
            text.indexOf('KB') !== -1 ||
            text.indexOf('MB') !== -1 ||
            text.indexOf('.jpg') !== -1 ||
            text.indexOf('.png') !== -1 ||
            (btn.getAttribute('aria-label') && btn.getAttribute('aria-label').indexOf('Remove') !== -1) ||
            (btn.getAttribute('aria-label') && btn.getAttribute('aria-label').indexOf('remove') !== -1) ||
            btn.innerHTML.indexOf('\\u00d7') !== -1 ||
            btn.innerHTML.indexOf('&times;') !== -1
        );
        
        var isNearFileUpload = (
            (btn.closest('[data-testid="stFileUploadWrapper"]') !== null) ||
            (btn.closest('[data-testid="stUploadedFile"]') !== null)
        );
        
        if (!isRemoveBtn && !isNearFileUpload) return;
        
        btn.dataset.deleteBound = 'true';
        
        btn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();
            
            console.log('[DELETE] X button clicked!');
            
            var fileName = extractFileName(btn);
            
            if (!fileName) {
                console.warn('[DELETE] Could not extract filename');
                alert('Could not identify file to delete');
                return;
            }
            
            console.log('[DELETE] Target:', fileName);
            
            sessionStorage.setItem('deleteScrollPos', window.pageYOffset || document.documentElement.scrollTop);
            sessionStorage.setItem('lastDeletedFile', fileName);
            
            performCompleteDeletion(fileName);
        });
    });
}

function extractFileName(btn) {
    var container = btn.closest('[data-testid="stUploadedFile"]') ||
                   btn.closest('div[class*="uploaded"]') ||
                   (btn.parentElement ? btn.parentElement.parentElement : null);
    
    if (container) {
        var text = container.textContent || '';
        var match = text.match(/([\\w\\-]+\\.(jpg|jpeg|png))/i);
        if (match) return match[1];
    }
    
    var siblings = [];
    var parent = btn.parentElement;
    if (parent) {
        for (var i = 0; i < parent.children.length; i++) {
            siblings.push(parent.children[i].textContent);
        }
    }
    var combinedText = siblings.join(' ');
    var match = combinedText.match(/([\\w\\-]+\\.(jpg|jpeg|png))/i);
    if (match) return match[1];
    
    var grandparent = btn.parentElement ? btn.parentElement.parentElement : null;
    if (grandparent) {
        var text = grandparent.textContent || '';
        var lines = text.split('\\n').map(function(s) { return s.trim(); }).filter(function(s) { return s; });
        for (var idx = 0; idx < lines.length; idx++) {
            var line = lines[idx];
            if (line.match(/\\.(jpg|jpeg|png)/i) && line.indexOf('KB') === -1) {
                return line.trim();
            }
        }
    }
    
    return '';
}

function performCompleteDeletion(fileName) {
    console.log('[DELETE] Starting complete deletion for:', fileName);
    
    var deletedCount = 0;
    
    document.querySelectorAll('div, section').forEach(function(el) {
        var text = el.textContent || '';
        if ((text.indexOf(fileName) !== -1) &&
            (text.indexOf('KB') !== -1 || text.indexOf('MB') !== -1 || el.querySelector('button'))) {
            
            if (!el.querySelector('img') && el.children.length <= 5) {
                try {
                    el.style.display = 'none';
                    el.remove();
                    deletedCount++;
                    console.log('[DELETE] Removed file list item');
                } catch(err) {}
            }
        }
    });
    
    document.querySelectorAll('div[class*="card"], div[data-testid], section').forEach(function(el) {
        var text = el.textContent || '';
        if (text.indexOf(fileName) !== -1 && el.querySelector('img')) {
            try {
                el.style.display = 'none';
                setTimeout(function() {
                    try { el.remove(); } catch(e) {}
                }, 100);
                deletedCount++;
                console.log('[DELETE] Removed result card');
            } catch(err) {}
        }
    });
    
    console.log('[DELETE] Complete! Removed ' + deletedCount + ' elements');
    
    showDeleteFeedback(fileName);
}

function showDeleteFeedback(fileName) {
    var feedback = document.createElement('div');
    feedback.style.position = 'fixed';
    feedback.style.top = '20px';
    feedback.style.right = '20px';
    feedback.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
    feedback.style.color = 'white';
    feedback.style.padding = '15px 25px';
    feedback.style.borderRadius = '10px';
    feedback.style.boxShadow = '0 5px 20px rgba(0,0,0,0.3)';
    feedback.style.zIndex = '99999';
    feedback.style.fontWeight = 'bold';
    feedback.innerHTML = 'Deleted: ' + fileName;
    
    document.body.appendChild(feedback);
    
    setTimeout(function() {
        try { feedback.remove(); } catch(e) {}
    }, 2000);
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initDeleteSystem);
} else {
    initDeleteSystem();
}

setTimeout(initDeleteSystem, 500);
setTimeout(initDeleteSystem, 1500);

console.log('[OK] Delete System v3 initialized');

// ========== Scroll Position Keeper ==========
(function() {
    var lastPos = 0;
    
    window.addEventListener('scroll', function() {
        lastPos = window.pageYOffset || document.documentElement.scrollTop;
        sessionStorage.setItem('scrollPos', lastPos);
    }, {passive: true});
    
    window.addEventListener('load', function() {
        var saved = sessionStorage.getItem('scrollPos') || sessionStorage.getItem('deleteScrollPos');
        if (saved && parseInt(saved) > 100) {
            setTimeout(function() { window.scrollTo(0, parseInt(saved)); }, 100);
        }
    });
})();

// ========== Folder Upload Support ==========
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(function() {
        var inputs = document.querySelectorAll('[data-testid="stFileUploadWrapper"] input[type="file"]');
        inputs.forEach(function(input) {
            input.setAttribute('webkitdirectory', '');
            input.setAttribute('directory', '');
            if (window.innerWidth <= 768) input.setAttribute('multiple', 'true');
        });
    }, 1000);
});
</script>
"""
st.components.v1.html(force_hide_js_v2, height=0)

# ==================== 返回顶部按钮 ====================
st.markdown("""
<button class="scroll-top-btn" onclick="window.scrollTo({top: 0, behavior: 'smooth'});">↑</button>

<script>
(function() {
    var scrollBtn = document.querySelector('.scroll-top-btn');
    if (scrollBtn) {
        window.addEventListener('scroll', function() {
            if (window.pageYOffset > 300) {
                scrollBtn.classList.add('show');
            } else {
                scrollBtn.classList.remove('show');
            }
        });
    }
})();
</script>
""", unsafe_allow_html=True)

# 初始化变量
uploaded_files = None

# ==================== 侧边栏 ====================
with st.sidebar:
    st.markdown("### 🔗 MES 集成")
    mes_url_input = st.text_input("MES 服务器地址", value=MES_URL)
    product_id = st.text_input("⚠️ 产品 ID（必填）", placeholder="如：P001", help="请输入产品ID，用于关联MES系统中的产品信息")
    if not product_id:
        st.warning("⚠️ 请输入产品ID，否则数据无法正确关联")
    
    st.markdown("---")
    st.markdown("### 🤖 自动上传")
    auto_upload = st.checkbox("🔄 检测后自动发送到MES", value=st.session_state.auto_upload_enabled,
                              help="开启后，每张图片检测完成会自动将结果发送到MES，无需手动点击")
    st.session_state.auto_upload_enabled = auto_upload
    if auto_upload and not product_id:
        st.warning("⚠️ 开启自动上传但未填写产品ID，自动上传将失败")
    
    st.info("💡 通过 frp 内网穿透访问本地 MES")
    st.markdown("---")
    
    st.markdown("### ⚙️ 检测参数")
    scratch_conf = st.slider("📈 划痕置信度", 0.0, 1.0, 0.5, 0.01)
    missing_conf = st.slider("📉 漏装置信度", 0.0, 1.0, 0.5, 0.01)
    st.info("💡 提示：建议阈值范围 0.3-0.7。太低会误检，太高会漏检")
    st.markdown("---")
    
    st.markdown("### ⚡ 高级选项")
    enable_preprocess = st.checkbox("启用智能图像增强", value=False,
                                    help="针对手机拍照优化（光线校正/锐化等）。关闭可提高检测稳定性")
    st.info("🔧 如果检测不到缺陷，请尝试打开此选项")
    st.markdown("---")
    
    st.write(f"📊 当前检测记录数：{len(st.session_state.detection_records)}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 导出 Excel 报告"):
            if len(st.session_state.detection_records) == 0:
                st.warning("暂无检测记录，请先上传图片检测。")
            else:
                df = pd.DataFrame(st.session_state.detection_records)
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name="检测记录")
                st.download_button(
                    label="点击下载 Excel",
                    data=output.getvalue(),
                    file_name=f"detection_report_{time.strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    with col2:
        if st.button("📄 导出 PDF 报告"):
            if len(st.session_state.detection_records) == 0:
                st.warning("暂无检测记录")
            else:
                pdf_buffer = generate_pdf_report(st.session_state.detection_records)
                st.download_button(
                    label="点击下载 PDF",
                    data=pdf_buffer,
                    file_name=f"report_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
    
    # 批量发送所有检测结果到MES
    st.markdown("---")
    st.markdown("### 🚀 批量操作")
    col_batch1, col_batch2 = st.columns(2)
    with col_batch1:
        if st.button("📤 一键发送所有结果到MES", type="primary", help="将所有检测结果批量发送到MES系统"):
            if not product_id:
                st.error("❌ 请先在左侧边栏输入产品ID！")
            elif len(st.session_state.detection_records) == 0:
                st.warning("暂无检测记录，请先上传图片进行检测。")
            else:
                success_count = 0
                fail_count = 0
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total = len(st.session_state.detection_records)
                for i, record in enumerate(st.session_state.detection_records):
                    status_text.text(f"正在发送: {record.get('文件名', f'第{i+1}张')} ({i+1}/{total})")
                    
                    detection_result = {
                        "文件名": record.get("文件名", f"image_{i+1}.jpg"),
                        "划痕数量": record.get("划痕数量", 0),
                        "漏装螺丝数量": record.get("漏装螺丝数量", 0),
                        "检测耗时(ms)": record.get("检测耗时(ms)", 0),
                        "产品ID": product_id
                    }
                    
                    if mes_connector.send_detection_result(detection_result):
                        success_count += 1
                    else:
                        fail_count += 1
                    
                    progress_bar.progress((i + 1) / total)
                
                status_text.text(f"✅ 发送完成！成功: {success_count}, 失败: {fail_count}")
                if fail_count == 0:
                    st.success(f"🎉 全部 {success_count} 条检测结果已成功发送到MES！")
                else:
                    st.warning(f"⚠️ 发送完成：成功 {success_count} 条，失败 {fail_count} 条")
    
    with col_batch2:
        if st.button("🗑️ 清空所有检测记录"):
            if len(st.session_state.detection_records) == 0:
                st.info("暂无检测记录")
            else:
                count = len(st.session_state.detection_records)
                st.session_state.detection_records = []
                st.success(f"已清空 {count} 条检测记录")

# ==================== 标题行 ====================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: center; gap: 10px;">
            <span style="font-size: 3.5rem;">🔧</span>
            <span style="font-size: 3rem; font-weight: 800; background: linear-gradient(135deg, #1E3A6F, #2E5A9F); -webkit-background-clip: text; background-clip: text; color: transparent;">轻量化智能质检系统</span>
        </div>
        <p style="text-align: center; font-size: 1rem; color: #4a627a; margin-top: 0;">基于 YOLOv26 的缺陷检测系统 —— MES团队</p>
        """,
        unsafe_allow_html=True
    )

# ==================== 加载模型 ====================
@st.cache_resource
def load_models():
    try:
        scratch_path = BASE_DIR / "models" / "scratch_best.pt"
        missing_path = BASE_DIR / "models" / "missing_screw_best.pt"
        
        detector = Detector(
            scratch_path=str(scratch_path),
            missing_path=str(missing_path),
            scratch_conf=scratch_conf,
            missing_conf=missing_conf
        )
        return detector
    except Exception as e:
        st.error(f"模型加载失败，请检查模型文件路径。\n错误：{e}")
        st.stop()

detector = load_models()
detector.set_scratch_conf(scratch_conf)
detector.set_missing_conf(missing_conf)

# 初始化批量处理器
batch_processor = BatchProcessor(detector, max_batch_size=50, enable_preprocess=enable_preprocess)

# ==================== 设备检测与自适应上传区域 ====================
st.markdown("""
<script>
// 检测设备类型并存储到sessionStorage
(function() {
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) 
                   || window.innerWidth <= 768;
    sessionStorage.setItem('isMobileDevice', isMobile);
    sessionStorage.setItem('deviceType', isMobile ? 'mobile' : 'desktop');
    
    // 文件夹上传支持（仅桌面端）
    if (!isMobile) {
        setTimeout(function() {
            var inputs = document.querySelectorAll('input[type="file"][data-testid*="folder"]');
            inputs.forEach(function(input) {
                input.setAttribute('webkitdirectory', '');
                input.setAttribute('directory', '');
            });
        }, 1000);
    }
})();
</script>
""", unsafe_allow_html=True)

# 读取设备类型
device_type = "desktop"  # 默认桌面端

# ==================== 上传区域（设备自适应）====================

# 上传选项卡样式
upload_tab_css = """
<style>
.upload-container {
    display: flex;
    gap: 1rem;  /* 紧凑间距 */
    margin: 1.5rem 0;
}
.upload-option-card {
    background: white;
    border-radius: 20px;
    padding: 1.5rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
    border: 3px solid transparent;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    flex: 1;
}
.upload-option-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.12);
    border-color: #667eea;
}
.upload-icon-large {
    font-size: 3rem;
    margin-bottom: 0.6rem;
}
.upload-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #1e3c72;
    margin-bottom: 0.4rem;
}
.upload-desc {
    font-size: 0.85rem;
    color: #6b7280;
    line-height: 1.4;
}

/* 隐藏文件列表 */
[data-testid="stFileUploadWrapper"] [data-testid="stUploadedFile"] {
    display: none !important;
}

/* 一键删除按钮样式 */
.clear-all-btn {
    background: linear-gradient(135deg, #ff416c, #ff4b2b) !important;
    color: white !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 0.7rem 2rem !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 15px rgba(255,65,108,0.3) !important;
    transition: all 0.2s !important;
}
.clear-all-btn:hover {
    transform: scale(1.05) !important;
    box-shadow: 0 6px 20px rgba(255,65,108,0.4) !important;
}
</style>
"""
st.markdown(upload_tab_css, unsafe_allow_html=True)

# ==================== 上传区域（紧凑双卡片）====================
st.markdown("### 📤 上传图片进行检测")

# 使用更紧凑的列布局（gap="small"）
col_upload1, col_upload2 = st.columns(2, gap="small")

with col_upload1:
    st.markdown("""
    <div class="upload-option-card" onclick="document.getElementById('single-file-input').click()">
        <div class="upload-icon-large">📸</div>
        <div class="upload-title">选择图片文件</div>
        <div class="upload-desc">
            从电脑/手机选择图片<br>
            <span style="color: #667eea; font-weight: 600;">📱 手机端最多选20张</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_files_single = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=f"file_uploader_single_v{st.session_state.uploader_key_version}",
        label_visibility="collapsed"
    )

with col_upload2:
    st.markdown("""
    <div class="upload-option-card" onclick="document.getElementById('folder-input').click()">
        <div class="upload-icon-large">📁</div>
        <div class="upload-title">选择整个文件夹</div>
        <div class="upload-desc">
            自动上传文件夹内所有图片<br>
            <span style="color: #10b981; font-weight: 600;">✨ 无数量限制（推荐）</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 文件夹上传组件
    uploaded_files_folder = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=f"file_uploader_folder_v{st.session_state.uploader_key_version}",
        label_visibility="collapsed"
    )

# 注入文件夹选择的JavaScript和隐藏input（通过components.html避免显示）
st.components.v1.html("""
<input type="file" 
       id="folder-input" 
       webkitdirectory 
       directory 
       multiple 
       accept="image/jpeg,image/jpg,image/png"
       style="display: none;"
       onchange="handleFolderSelect(event)">

<script>
function handleFolderSelect(event) {
    const files = event.target.files;
    if (files.length === 0) return;
    
    alert('已选择 ' + files.length + ' 张图片，正在上传...');
    
    // 创建新的DataTransfer对象
    const dataTransfer = new DataTransfer();
    for (let file of files) {
        dataTransfer.items.add(file);
    }
    
    // 找到Streamlit的文件上传器并设置文件
    const streamlitInput = document.querySelector('input[data-testid="stFileUploadInput"]');
    if (streamlitInput) {
        streamlitInput.files = dataTransfer.files;
        const changeEvent = new Event('change', { bubbles: true });
        streamlitInput.dispatchEvent(changeEvent);
    }
}
</script>
""", height=0)

# 合并两种上传方式的结果
uploaded_files = None
if uploaded_files_single:
    uploaded_files = uploaded_files_single
elif uploaded_files_folder:
    uploaded_files = uploaded_files_folder

# 调试输出：显示上传的文件数量
if uploaded_files:
    print(f"📂 上传了 {len(uploaded_files)} 个文件:")
    for f in uploaded_files[:5]:  # 只显示前5个
        print(f"   - {f.name}")
    if len(uploaded_files) > 5:
        print(f"   ... 还有 {len(uploaded_files)-5} 个文件")

# ==================== 批量检测与展示 ====================

# 合并底部上传的新文件
all_uploaded_files = list(uploaded_files) if uploaded_files else []
if "bottom_new_files" in st.session_state and st.session_state.bottom_new_files:
    for f in st.session_state.bottom_new_files:
        if f.name not in [x.name for x in all_uploaded_files]:
            all_uploaded_files.append(f)
    del st.session_state.bottom_new_files

print(f"\n📊 待处理文件总数: {len(all_uploaded_files)}")
print(f"   just_cleared 标志: {st.session_state.get('just_cleared', False)}")
print(f"   uploader_key_version: {st.session_state.get('uploader_key_version', 0)}")

if all_uploaded_files:
    # 初始化 results 和 errors 变量
    results = []
    errors = []

    # 如果刚执行过清除操作，跳过本次检测（防止rerun时重复添加）
    # 但要确保：如果用户上传了新文件，即使刚清除过也要检测
    if st.session_state.get("just_cleared", False):
        # 检查是否真的刚清除（没有新文件上传）
        # 如果有文件，说明是新上传的，应该正常处理
        print("⚠️ 检测到 just_cleared 标志，但仍有待处理文件")
        print("   → 正常执行检测（这是用户新上传的文件）")
        st.session_state.just_cleared = False  # 重置标志

    # 执行批量检测（紧凑模式）
    print(f"\n🚀 开始批量检测: 共 {len(all_uploaded_files)} 张图片")
    results, errors = batch_processor.process_batch(
        all_uploaded_files,
        progress_callback=None  # 暂时禁用进度条以减少空白
    )
    print(f"✅ 检测完成: 成功 {len(results)} 张, 失败 {len(errors)} 张")

    # 将结果存入缓存
    for result in results:
        file_key = result['file_key']
        if file_key not in st.session_state.detection_cache:
            st.session_state.detection_cache[file_key] = (
                result['combined_img'],
                result['info'],
                result['inference_time'],
                result['image']
            )
            st.session_state.detection_records.append(result['record'])
            print(f"   ✓ 已添加: {file_key}")
            
            # 🔄 自动上传：检测完成后立即发送到MES
            if st.session_state.auto_upload_enabled and file_key not in st.session_state.auto_uploaded_files:
                if product_id:
                    detection_result = {
                        "文件名": file_key,
                        "划痕数量": result['record'].get('划痕数量', 0),
                        "漏装螺丝数量": result['record'].get('漏装螺丝数量', 0),
                        "检测耗时(ms)": result['record'].get('检测耗时(ms)', 0),
                        "产品ID": product_id
                    }
                    send_success = mes_connector.send_detection_result(detection_result)
                    if send_success:
                        st.session_state.auto_uploaded_files.add(file_key)
                        print(f"   🔄 自动上传成功: {file_key}")
                    else:
                        print(f"   ⚠️ 自动上传失败: {file_key} (可手动重试)")
                else:
                    print(f"   ⏭️ 跳过自动上传（未填写产品ID）: {file_key}")

    # 显示错误信息
    if errors:
        with st.expander(f"⚠️ {len(errors)} 张图片检测失败（点击查看详情）"):
            for err in errors:
                st.error(f"❌ 第{err['index']}张 [{err['file_key']}]: {err['error']}")
                print(f"   ✗ 失败: {err['file_key']} - {err['error']}")
    
    # 实时统计面板（仅当有检测记录时显示）
    if st.session_state.detection_records:
        # 使用 detection_records 的长度作为总图片数（更准确）
        total_images = len(st.session_state.detection_records)
        total_scratches = sum(r['划痕数量'] for r in st.session_state.detection_records)
        total_missing = sum(r['漏装螺丝数量'] for r in st.session_state.detection_records)
        avg_time_val = sum(r['检测耗时(ms)'] for r in st.session_state.detection_records) / len(st.session_state.detection_records) if st.session_state.detection_records else 0

        success_rate = 100.0  # 能到这里说明都成功了
        avg_time_rounded = round(avg_time_val, 1)
        
        # 构建统计面板HTML（使用f-string确保变量替换）
        stats_html = f"""
        <div class="stats-panel">
            <div style="font-size: 1.2rem; font-weight: 700; color: #1e3c72; margin-bottom: 0.8rem;">📊 检测统计</div>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value">{total_images}</div>
                    <div class="stat-label">已检图片</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{success_rate:.0f}%</div>
                    <div class="stat-label">成功率</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{total_scratches}</div>
                    <div class="stat-label">总划痕数</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{total_missing}</div>
                    <div class="stat-label">总漏装数</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{avg_time_rounded}ms</div>
                    <div class="stat-label">平均耗时</div>
                </div>
            </div>
        </div>
        """
        
        st.markdown(stats_html, unsafe_allow_html=True)

        # 一键清除按钮（改进版 - 直接清除，无需二次确认）
        col_clear1, col_clear2, col_clear3 = st.columns([1, 2, 1])
        with col_clear2:
            if st.button(f"🗑️ 一键清除全部结果 ({total_images}张)", key="clear_all_btn", use_container_width=True,
                        help="点击后将立即清除所有检测结果和上传的文件"):
                # 直接清除所有检测数据
                st.session_state.detection_records = []
                st.session_state.detection_cache = {}
                st.session_state.deleted_files = set()
                st.session_state.just_cleared = True  # 标记：刚执行了清除
                st.session_state.auto_uploaded_files = set()  # 清除自动上传记录

                # 🔑 关键：增加版本号，强制重新创建 file_uploader 组件
                # 这样 Streamlit 会认为这是新组件，从而清除文件列表
                st.session_state.uploader_key_version += 1

                if "bottom_new_files" in st.session_state:
                    del st.session_state.bottom_new_files
                if "pending_uploads" in st.session_state:
                    del st.session_state.pending_uploads
                if "all_uploaded_files_persistent" in st.session_state:
                    st.session_state.all_uploaded_files_persistent = []

                st.success("✅ 已清除所有检测结果和上传文件")
                st.rerun()

        st.markdown("---")

    # 显示检测结果卡片（从缓存中读取）
    if st.session_state.detection_cache:
        for file_key, cached_data in st.session_state.detection_cache.items():
            combined_img, info, inference_time, image = cached_data

            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)

                col_header = st.columns([5, 1])
                with col_header[0]:
                    st.markdown(f"**📁 {file_key}**")
                with col_header[1]:
                    st.caption(f"⏱️ {inference_time*1000:.1f}ms")

                col_img1, col_img2 = st.columns(2, gap="medium")
                with col_img1:
                    st.markdown("**原始工件**")
                    st.image(image, use_container_width=True, output_format="PNG")
                with col_img2:
                    st.markdown("**检测结果**")
                    st.image(combined_img, use_container_width=True, output_format="PNG", clamp=True, channels="RGB")

                col_met1, col_met2 = st.columns(2)
                with col_met1:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{info["scratch_count"]}</div><div class="metric-label">划痕数量</div></div>', unsafe_allow_html=True)
                with col_met2:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{info["missing_count"]}</div><div class="metric-label">漏装螺丝</div></div>', unsafe_allow_html=True)

                # 添加发送到MES按钮
                col_btn1, col_btn2 = st.columns([4, 1])
                with col_btn2:
                    if st.button(f"📤 发送到MES", key=f"send_{file_key}"):
                        # 构建检测结果数据
                        detection_result = {
                            "文件名": file_key,
                            "划痕数量": info["scratch_count"],
                            "漏装螺丝数量": info["missing_count"],
                            "检测耗时(ms)": round(inference_time * 1000, 1),
                            "检测时间": datetime.now().isoformat(),
                            "产品ID": product_id
                        }
                        
                        # 发送到MES
                        if mes_connector.send_detection_result(detection_result):
                            st.success(f"✅ 已发送到MES: {file_key}")
                        else:
                            st.error(f"❌ 发送失败: {file_key}")

                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("---")

else:
    st.markdown(
        """
        <div style="display: flex; justify-content: center; align-items: center; flex-direction: column; padding: 3rem; background: #f9fafb; border-radius: 28px; margin-top: 1rem;">
            <img src="https://img.icons8.com/ios/100/2a5298/camera--v1.png" width="60">
            <p style="color: #6b7280; margin-top: 1rem; font-weight: 600; font-size: 1.1rem;">等待上传图片……</p>
            <p style="color: #9ca3af; font-size: 0.85rem; margin-top: 0.5rem;">支持 JPG, PNG 格式 | 可上传多张或整个文件夹</p>
            <p style="color: #9ca3af; font-size: 0.8rem; margin-top: 0.3rem;">💡 点击原生文件列表的 × 可同时删除检测结果</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ==================== 底部上传按钮（检测结果之后）====================
if st.session_state.detection_records:
    st.markdown("---")
    
    st.markdown("""
    <style>
    .bottom-upload-container {
        text-align: center;
        padding: 2rem 1rem;
        margin-top: 1rem;
    }
    .bottom-upload-container [data-testid="stFileUploadWrapper"] {
        display: inline-block !important;
    }
    .bottom-upload-container [data-testid="stFileUploadWrapper"] > div {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%) !important;
        border: none !important;
        border-radius: 40px !important;
        padding: 0.75rem 2.5rem !important;
        box-shadow: 0 4px 15px rgba(30,60,114,0.3) !important;
        cursor: pointer !important;
        transition: all 0.2s !important;
    }
    .bottom-upload-container [data-testid="stFileUploadWrapper"] > div:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 6px 20px rgba(30,60,114,0.4) !important;
    }
    .bottom-upload-container [data-testid="stFileUploadWrapper"] label {
        color: white !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        cursor: pointer !important;
    }
    .bottom-upload-container [data-testid="stUploadedFile"] {
        display: none !important;
    }
    </style>
    
    <div class="bottom-upload-container">
    """, unsafe_allow_html=True)
    
    bottom_uploaded = st.file_uploader(
        "📤 继续上传图片",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=f"bottom_file_uploader_v{st.session_state.uploader_key_version}",
        label_visibility="collapsed"
    )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    if bottom_uploaded:
        st.session_state.bottom_new_files = bottom_uploaded
        st.rerun()

# ==================== 页脚 ====================
st.markdown(
    """
    <div style="text-align: center; margin-top: 3rem; padding: 1rem; color: #9ca3af; font-size: 0.7rem;">
        Powered by YOLOv26 · Streamlit · MES团队 · 支持手机拍照
    </div>
    """,
    unsafe_allow_html=True
)
