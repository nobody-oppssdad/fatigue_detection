import sys
import cv2
import torch
import time
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import QSound
from PIL import Image, ImageDraw, ImageFont
from model.faster_rcnn import get_faster_rcnn
from utils.transforms import get_val_transform
from torchvision.ops import nms
import os
import warnings
import traceback

warnings.filterwarnings("ignore")

# ===================== 全局配置（修复FP16+NMS兼容） =====================
CONFIG = {
    "img_size": (640, 480),  # 模型输入尺寸
    "conf_thresh": 0.3,  # 最优阈值（提升检出率）
    "nms_thresh": 0.4,  # 放宽NMS，避免误删小目标
    "alarm_interval": 2,  # 报警间隔
    "alarm_sound": "alarm.wav",
    "save_dir": "detection_results",
    "fatigue_frame_thresh": 3,  # 连续3帧疲劳才触发报警
    "fps_smooth_window": 5,  # FPS滑动平均窗口
    "use_fp16": True  # 保留FP16提速，但修复NMS兼容
}

# 创建保存目录
os.makedirs(CONFIG["save_dir"], exist_ok=True)

# 疲劳状态映射
FATIGUE_LEVEL = {
    "normal": ("正常", (0, 255, 0)),
    "mild": ("轻度疲劳", (255, 165, 0)),
    "fatigue": ("疲劳", (255, 0, 0)),
    "severe": ("极度疲劳", (128, 0, 128))
}

# 类别配置
CLS_CONFIG = {
    1: {"name": "闭眼", "color": (0, 0, 255)},
    2: {"name": "睁眼", "color": (0, 255, 0)},
    3: {"name": "闭嘴", "color": (255, 255, 0)},
    4: {"name": "张嘴", "color": (255, 0, 0)}
}


class DetectorThread(QThread):
    """检测线程（修复FP16+NMS兼容问题）"""
    frame_signal = pyqtSignal(np.ndarray)
    status_signal = pyqtSignal(str, str)
    fps_signal = pyqtSignal(float)
    alarm_signal = pyqtSignal(bool)
    img_result_signal = pyqtSignal(np.ndarray, str)

    def __init__(self, checkpoint_path, source=0, device="cuda"):
        super().__init__()
        self.checkpoint_path = checkpoint_path

        # ===================== 修复报错 =====================
        if isinstance(source, str):
            self.source = int(source) if source.isdigit() else source
        else:
            self.source = source
        # ===================================================

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 运行状态
        self.running = False
        self.frame_count = 0
        self.fps_list = []
        self.last_time = time.time()
        self.last_alarm_time = 0

        # 疲劳状态累计
        self.fatigue_frame_count = 0
        self.current_fatigue_level = "normal"

        # 模型和预处理
        self.model = self._load_model()
        self.transform = get_val_transform()

        # 预加载字体
        self.font_small = self._load_font(12)
        self.font_big = self._load_font(14)

    def _load_font(self, size):
        """加载中文字体（兼容多系统）"""
        font_paths = ["simhei.ttf", "msyh.ttc", "/System/Library/Fonts/PingFang.ttc", "Arial Unicode.ttf"]
        for path in font_paths:
            try:
                return ImageFont.truetype(path, size)
            except:
                continue
        return ImageFont.load_default()

    def _load_model(self):
        """加载模型（优化推理速度，兼容FP16）"""
        try:
            num_classes = 5
            model = get_faster_rcnn(num_classes).to(self.device)
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
            model.eval()

            # 推理优化：仅模型转半精度，NMS时张量转回float32
            torch.set_grad_enabled(False)
            if self.device.type == "cuda" and CONFIG["use_fp16"]:
                model.half()  # 模型参数转半精度（提速）
                print(f"✅ 模型已转FP16半精度（设备：{self.device}）")

            print(f"✅ 模型加载成功（设备：{self.device}）")
            return model
        except Exception as e:
            print(f"❌ 模型加载失败：{str(e)}")
            QMessageBox.critical(None, "错误", f"模型加载失败：{str(e)}")
            return None

    def _restore_box_coord(self, box, frame_shape):
        """坐标还原（精准适配实际帧尺寸）"""
        frame_h, frame_w = frame_shape
        model_w, model_h = CONFIG["img_size"]

        scale_x = frame_w / model_w
        scale_y = frame_h / model_h

        x1 = max(0, min(int(box[0] * scale_x), frame_w))
        y1 = max(0, min(int(box[1] * scale_y), frame_h))
        x2 = max(0, min(int(box[2] * scale_x), frame_w))
        y2 = max(0, min(int(box[3] * scale_y), frame_h))

        return (x1, y1, x2, y2)

    def _remove_duplicate_boxes(self, boxes, scores, labels):
        """去除重复框（核心修复：NMS前转float32）"""
        if len(boxes) == 0:
            return boxes, scores, labels

        # 关键修复：将半精度张量转为float32，兼容NMS
        boxes_float32 = torch.tensor(boxes, dtype=torch.float32)
        scores_float32 = torch.tensor(scores, dtype=torch.float32)

        # 执行NMS（现在支持float32）
        keep_idx = nms(boxes_float32, scores_float32, CONFIG["nms_thresh"])
        keep_idx = keep_idx.cpu().numpy()

        return boxes[keep_idx], scores[keep_idx], labels[keep_idx]

    def _judge_fatigue_level(self, labels):
        """判断疲劳等级（优化逻辑）"""
        has_closed_eye = 1 in labels
        has_open_eye = 2 in labels
        has_closed_mouth = 3 in labels
        has_open_mouth = 4 in labels

        if has_closed_eye:
            if has_open_mouth:
                return "severe"
            else:
                return "fatigue"
        elif has_open_mouth and not has_closed_eye:
            return "mild"
        else:
            return "normal"

    def _draw_text_opencv(self, frame, text, pos, color=(255, 255, 255), font_size=0.4):
        """OpenCV绘制中文（避免PIL转换）"""
        x, y = pos
        h, w = frame.shape[:2]

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        bbox = draw.textbbox((0, 0), text, font=self.font_small)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        text_x = max(0, min(x, w - text_w))
        text_y = max(text_h, min(y, h))

        draw.rectangle([text_x, text_y - text_h, text_x + text_w, text_y], fill=(0, 0, 0))
        draw.text((text_x, text_y - text_h), text, font=self.font_small, fill=color)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _draw_results(self, frame, boxes, scores, labels, fatigue_level, frame_shape):
        """绘制结果（优化版）"""
        frame_copy = frame.copy()
        h, w = frame_shape

        # 绘制检测框
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = self._restore_box_coord(box, frame_shape)
            cls_info = CLS_CONFIG.get(label, {"name": "未知", "color": (255, 255, 255)})

            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), cls_info["color"], 2)

            text = f"{cls_info['name']} {score:.2f}"
            text_y = y1 - 5 if y1 - 5 > 0 else y2 + 15
            frame_copy = self._draw_text_opencv(frame_copy, text, (x1, text_y), (255, 255, 255))

        # 绘制人头框和疲劳状态
        if len(boxes) > 0:
            restored_boxes = [self._restore_box_coord(b, frame_shape) for b in boxes]
            all_x1 = min([b[0] for b in restored_boxes])
            all_y1 = min([b[1] for b in restored_boxes])
            all_x2 = max([b[2] for b in restored_boxes])
            all_y2 = max([b[3] for b in restored_boxes])

            expand_w = (all_x2 - all_x1) * 0.2
            expand_h = (all_y2 - all_y1) * 0.3
            head_x1 = max(0, int(all_x1 - expand_w))
            head_y1 = max(0, int(all_y1 - expand_h))
            head_x2 = min(w, int(all_x2 + expand_w))
            head_y2 = min(h, int(all_y2 + expand_h))

            level_name, color_bgr = FATIGUE_LEVEL[fatigue_level]
            cv2.rectangle(frame_copy, (head_x1, head_y1), (head_x2, head_y2), color_bgr, 3)

            status_text = f"疲劳状态：{level_name}"
            status_y = head_y2 + 25 if (head_y2 + 25) < h else head_y1 - 20
            frame_copy = self._draw_text_opencv(
                frame_copy, status_text, (head_x1 + 10, status_y),
                color_bgr, font_size=0.5
            )

        # 绘制FPS
        if not hasattr(self, 'is_img_mode') or not self.is_img_mode:
            fps_text = f"FPS: {np.mean(self.fps_list):.1f}"
            frame_copy = self._draw_text_opencv(frame_copy, fps_text, (10, 30), (255, 255, 0))

        return frame_copy

    def detect_image(self, img_path):
        """图片检测（修复FP16兼容）"""
        self.is_img_mode = True
        try:
            frame = cv2.imread(img_path)
            if frame is None:
                QMessageBox.warning(None, "警告", f"无法读取图片：{img_path}")
                return

            frame_input = cv2.resize(frame, CONFIG["img_size"], interpolation=cv2.INTER_LINEAR)
            frame_shape = (frame.shape[0], frame.shape[1])
            img_pil = Image.fromarray(cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB))
            img_tensor, _ = self.transform(img_pil, None)

            # FP16兼容：输入张量转半精度（如果启用）
            if self.device.type == "cuda" and CONFIG["use_fp16"]:
                img_tensor = img_tensor.half()

            img_tensor = img_tensor.unsqueeze(0).to(self.device)

            # 推理
            with torch.no_grad():
                pred = self.model(img_tensor)[0]

            # 后处理：将模型输出转回float32（核心修复）
            pred["boxes"] = pred["boxes"].float()
            pred["scores"] = pred["scores"].float()

            # 过滤低置信度框
            mask = pred["scores"] >= CONFIG["conf_thresh"]
            boxes = pred["boxes"][mask].cpu().numpy()
            scores = pred["scores"][mask].cpu().numpy()
            labels = pred["labels"][mask].cpu().numpy()
            boxes, scores, labels = self._remove_duplicate_boxes(boxes, scores, labels)

            # 判断疲劳等级
            fatigue_level = self._judge_fatigue_level(labels) if len(boxes) > 0 else "normal"

            # 绘制结果
            frame_with_result = self._draw_results(frame, boxes, scores, labels, fatigue_level, frame_shape)

            # 保存结果
            img_name = os.path.basename(img_path)
            save_path = os.path.join(CONFIG["save_dir"], f"detected_{img_name}")
            cv2.imwrite(save_path, frame_with_result)

            # 推送信号
            self.img_result_signal.emit(frame_with_result, save_path)
            level_name, color_bgr = FATIGUE_LEVEL[fatigue_level]
            color_hex = f"#{color_bgr[2]:02x}{color_bgr[1]:02x}{color_bgr[0]:02x}"
            self.status_signal.emit(level_name, color_hex)
            self.alarm_signal.emit(fatigue_level == "severe")

            # 报警
            if fatigue_level == "severe":
                try:
                    QSound.play(CONFIG["alarm_sound"])
                except:
                    QApplication.beep()

        except Exception as e:
            QMessageBox.critical(None, "错误", f"图片检测失败：{str(e)}\n{traceback.format_exc()}")

    def run(self):
        """视频/摄像头检测（核心修复FP16+NMS兼容）"""
        self.is_img_mode = False
        self.running = True

        # 摄像头参数优化
        cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["img_size"][0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["img_size"][1])

        if not cap.isOpened():
            print(f"❌ 无法打开检测源：{self.source}")
            self.running = False
            return

        # 获取摄像头实际分辨率
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_shape = (frame_h, frame_w)
        print(f"📹 摄像头分辨率：{frame_w}×{frame_h}")

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            # 计算FPS
            current_time = time.time()
            self.fps_list.append(1 / (current_time - self.last_time))
            if len(self.fps_list) > CONFIG["fps_smooth_window"]:
                self.fps_list.pop(0)
            self.last_time = current_time
            self.fps_signal.emit(np.mean(self.fps_list))

            # 预处理
            frame_input = cv2.resize(frame, CONFIG["img_size"], interpolation=cv2.INTER_LINEAR)
            img_pil = Image.fromarray(cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB))
            img_tensor, _ = self.transform(img_pil, None)

            # FP16兼容：输入张量转半精度
            if self.device.type == "cuda" and CONFIG["use_fp16"]:
                img_tensor = img_tensor.half()

            img_tensor = img_tensor.unsqueeze(0).to(self.device)

            # 推理
            with torch.no_grad():
                pred = self.model(img_tensor)[0]

            # 核心修复：将模型输出的boxes/scores转回float32
            pred["boxes"] = pred["boxes"].float()
            pred["scores"] = pred["scores"].float()

            # 后处理
            mask = pred["scores"] >= CONFIG["conf_thresh"]
            boxes = pred["boxes"][mask].cpu().numpy()
            scores = pred["scores"][mask].cpu().numpy()
            labels = pred["labels"][mask].cpu().numpy()
            boxes, scores, labels = self._remove_duplicate_boxes(boxes, scores, labels)

            # 疲劳状态累计
            frame_fatigue_level = self._judge_fatigue_level(labels) if len(boxes) > 0 else "normal"
            if frame_fatigue_level != "normal":
                self.fatigue_frame_count += 1
                if self.fatigue_frame_count >= CONFIG["fatigue_frame_thresh"]:
                    self.current_fatigue_level = frame_fatigue_level
            else:
                self.fatigue_frame_count = 0
                self.current_fatigue_level = "normal"

            # 推送状态
            level_name, color_bgr = FATIGUE_LEVEL[self.current_fatigue_level]
            color_hex = f"#{color_bgr[2]:02x}{color_bgr[1]:02x}{color_bgr[0]:02x}"
            self.status_signal.emit(level_name, color_hex)

            # 报警逻辑
            is_alarm = self.current_fatigue_level == "severe"
            self.alarm_signal.emit(is_alarm)
            if is_alarm and (time.time() - self.last_alarm_time) > CONFIG["alarm_interval"]:
                try:
                    QSound.play(CONFIG["alarm_sound"])
                except:
                    QApplication.beep()
                self.last_alarm_time = time.time()
                print(f"🔔 连续{CONFIG['fatigue_frame_thresh']}帧极度疲劳，触发报警（FPS：{np.mean(self.fps_list):.1f}）")

            # 绘制结果
            frame_with_result = self._draw_results(frame, boxes, scores, labels, self.current_fatigue_level,
                                                   frame_shape)
            self.frame_signal.emit(frame_with_result)

            self.frame_count += 1

        # 释放资源
        cap.release()
        torch.cuda.empty_cache()

    def stop(self):
        """停止检测"""
        self.running = False
        self.fatigue_frame_count = 0
        self.current_fatigue_level = "normal"


class FatigueDetectionUI(QMainWindow):
    """主界面（新增：选择模型按钮，其他完全不变）"""

    def __init__(self):
        super().__init__()
        self.checkpoint_path = None  # 模型路径
        self.source = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.detector_thread = None

        # 高DPI适配
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

        self.init_ui()

    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("疲劳检测系统")
        self.setFixedSize(CONFIG["img_size"][0] + 80, CONFIG["img_size"][1] + 260)
        self.setStyleSheet("background-color: #f5f5f5;")

        # 中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(30, 20, 30, 20)
        layout.setSpacing(15)

        # ===================== 【新增：模型选择区域】 =====================
        model_layout = QHBoxLayout()
        self.model_label = QLabel("未选择模型")
        self.model_btn = QPushButton("选择模型文件")
        self.model_btn.setStyleSheet("padding:6px 15px; font-size:14px;")
        self.model_btn.clicked.connect(self.select_model)
        model_layout.addWidget(QLabel("模型："))
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(self.model_btn)
        layout.addLayout(model_layout)
        # =================================================================

        # 显示区域
        self.display_label = QLabel()
        self.display_label.setFixedSize(CONFIG["img_size"][0], CONFIG["img_size"][1])
        self.display_label.setStyleSheet("""
            border: 3px solid #444;
            border-radius: 8px;
            background-color: #000;
            color: #fff;
            font-size: 14px;
        """)
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setText("请先选择模型！")
        layout.addWidget(self.display_label)

        # 状态区域
        status_layout = QHBoxLayout()
        status_layout.setSpacing(30)

        self.status_label = QLabel("疲劳状态：未检测")
        self.status_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #888;
            min-width: 150px;
        """)
        status_layout.addWidget(self.status_label)

        self.fps_label = QLabel("FPS：0.0")
        self.fps_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #ffcc00;
            min-width: 100px;
        """)
        status_layout.addWidget(self.fps_label)

        self.alarm_label = QLabel("报警状态：正常")
        self.alarm_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #00cc00;
            min-width: 150px;
        """)
        status_layout.addWidget(self.alarm_label)

        layout.addLayout(status_layout)

        # 按钮区域
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(20)

        self.start_btn = QPushButton("启动摄像头检测")
        self.start_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                padding: 10px 30px;
                border-radius: 8px;
                background-color: #0099cc;
                color: white;
                border: none;
            }
            QPushButton:hover {
                background-color: #0077aa;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666;
            }
        """)
        self.start_btn.clicked.connect(self.start_camera_detection)
        self.start_btn.setEnabled(False)  # 未选模型时禁用
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("停止检测")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                padding: 10px 30px;
                border-radius: 8px;
                background-color: #cc3300;
                color: white;
                border: none;
            }
            QPushButton:hover {
                background-color: #aa2200;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666;
            }
        """)
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)

        self.select_img_btn = QPushButton("选择图片检测")
        self.select_img_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                padding: 10px 30px;
                border-radius: 8px;
                background-color: #009900;
                color: white;
                border: none;
            }
            QPushButton:hover {
                background-color: #007700;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666;
            }
        """)
        self.select_img_btn.clicked.connect(self.select_image)
        self.select_img_btn.setEnabled(False)  # 未选模型时禁用
        btn_layout.addWidget(self.select_img_btn)

        layout.addLayout(btn_layout)

        # 保存提示
        self.save_label = QLabel("")
        self.save_label.setStyleSheet("font-size: 12px; color: #666;")
        self.save_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.save_label)

    # ===================== 【新增：选择模型函数】 =====================
    def select_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型", "", "模型文件 (*.pth *.pt)")
        if not path:
            return
        self.checkpoint_path = path
        self.model_label.setText(os.path.basename(path))
        self.start_btn.setEnabled(True)
        self.select_img_btn.setEnabled(True)
        QMessageBox.information(self, "成功", "模型路径已选择！")

    def start_camera_detection(self):
        """启动摄像头检测"""
        if not self.checkpoint_path:
            QMessageBox.warning(self, "提示", "请先选择模型！")
            return

        self.display_label.clear()
        self.save_label.setText("")

        self.detector_thread = DetectorThread(self.checkpoint_path, self.source, self.device)
        if self.detector_thread.model is None:
            return

        self.detector_thread.frame_signal.connect(self.update_display, Qt.QueuedConnection)
        self.detector_thread.status_signal.connect(self.update_status, Qt.QueuedConnection)
        self.detector_thread.fps_signal.connect(self.update_fps, Qt.QueuedConnection)
        self.detector_thread.alarm_signal.connect(self.update_alarm_status, Qt.QueuedConnection)

        self.detector_thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.select_img_btn.setEnabled(False)

    def select_image(self):
        """选择图片检测"""
        if not self.checkpoint_path:
            QMessageBox.warning(self, "提示", "请先选择模型！")
            return

        self.stop_detection()

        img_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp)"
        )

        if img_path:
            if self.detector_thread is None or not self.detector_thread.isRunning():
                self.detector_thread = DetectorThread(self.checkpoint_path, self.source, self.device)
                if self.detector_thread.model is None:
                    return
                self.detector_thread.img_result_signal.connect(self.update_display, Qt.QueuedConnection)
                self.detector_thread.img_result_signal.connect(self.update_save_info, Qt.QueuedConnection)
                self.detector_thread.status_signal.connect(self.update_status, Qt.QueuedConnection)
                self.detector_thread.alarm_signal.connect(self.update_alarm_status, Qt.QueuedConnection)

            self.detector_thread.detect_image(img_path)

    def update_display(self, frame, save_path=None):
        """更新显示"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

        pixmap = QPixmap.fromImage(q_img).scaled(
            self.display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.display_label.setPixmap(pixmap)

    def update_save_info(self, frame, save_path):
        """更新保存提示"""
        self.save_label.setText(f"✅ 检测结果已保存：{save_path}")

    def stop_detection(self):
        """停止检测"""
        if self.detector_thread and self.detector_thread.isRunning():
            self.detector_thread.stop()
            self.detector_thread.wait()

        self.display_label.clear()
        self.display_label.setText("请选择检测模式：\n1. 启动摄像头检测\n2. 选择图片检测")
        self.status_label.setText("疲劳状态：未检测")
        self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #888;")
        self.fps_label.setText("FPS：0.0")
        self.alarm_label.setText("报警状态：正常")
        self.alarm_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #00cc00;")
        self.save_label.setText("")

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.select_img_btn.setEnabled(True)

    def update_status(self, level_name, color_hex):
        """更新疲劳状态"""
        self.status_label.setText(f"疲劳状态：{level_name}")
        self.status_label.setStyleSheet(f"""
            font-size: 18px;
            font-weight: bold;
            color: {color_hex};
            min-width: 150px;
        """)

    def update_fps(self, fps):
        """更新FPS"""
        self.fps_label.setText(f"FPS：{fps:.1f}")

    def update_alarm_status(self, is_alarm):
        """更新报警状态"""
        if is_alarm:
            self.alarm_label.setText("报警状态：极度疲劳！")
            self.alarm_label.setStyleSheet("""
                font-size: 18px;
                font-weight: bold;
                color: #cc0000;
                min-width: 150px;
            """)
        else:
            self.alarm_label.setText("报警状态：正常")
            self.alarm_label.setStyleSheet("""
                font-size: 18px;
                font-weight: bold;
                color: #00cc00;
                min-width: 150px;
            """)

    def closeEvent(self, event):
        """关闭窗口"""
        self.stop_detection()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FatigueDetectionUI()
    window.show()
    sys.exit(app.exec_())