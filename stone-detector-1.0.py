import sys
import os
import random
import time
import threading
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QLabel, QPushButton, QFileDialog, QSlider, QSpinBox, 
                           QScrollArea, QFrame, QStatusBar, QCheckBox, QGroupBox)
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
import pyautogui
import keyboard
import ctypes
import json
from ultralytics import YOLO

# Yönetici olarak çalıştırma kontrolü
def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

if not is_admin():
    ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
    sys.exit()

class YOLOWorker(QThread):
    update_frame = pyqtSignal(np.ndarray)
    update_status = pyqtSignal(str)
    stone_detected = pyqtSignal(tuple)  # (x, y) koordinatları
    
    def __init__(self, model_path="yolov8n.pt"):
        super().__init__()
        self.running = False
        self.paused = False
        self.model = None
        self.model_path = model_path
        self.confidence = 0.5
        self.click_interval = 3  # Varsayılan tıklama aralığı (saniye)
        
    def load_model(self):
        try:
            self.update_status.emit("Model yükleniyor...")
            # Eğer model dosyası yoksa yeni bir model oluştur
            if not os.path.exists(self.model_path):
                self.model = YOLO("yolov8n.pt")
                self.update_status.emit("Varsayılan YOLOv8n modeli yüklendi")
            else:
                self.model = YOLO(self.model_path)
                self.update_status.emit(f"Model başarıyla yüklendi: {self.model_path}")
            return True
        except Exception as e:
            self.update_status.emit(f"Model yüklenirken hata oluştu: {str(e)}")
            return False
        
    def set_confidence(self, value):
        self.confidence = value
        
    def set_click_interval(self, seconds):
        self.click_interval = seconds

    def detect_stones(self, frame):
        if self.model is None:
            if not self.load_model():
                return frame, None
                
        # YOLOv8 ile tespit et
        results = self.model(frame, conf=self.confidence)
        annotated_frame = results[0].plot()
        
        # En yüksek güven puanına sahip sonucu bul
        best_detection = None
        best_confidence = 0
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # YOLOv8 varsayılan modelleri kullanıyorsak sınıfları kontrol etmemiz gerekebilir
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                if conf > best_confidence:
                    best_confidence = conf
                    # Tam ortasını hesapla
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    best_detection = (center_x, center_y)
                    
        return annotated_frame, best_detection

    def run(self):
        self.running = True
        
        if not self.load_model():
            self.running = False
            return
            
        last_stone_click = time.time()
        last_z_press = time.time()
        z_interval = random.uniform(0, 3)  # 0-3 saniye arası
        
        move_directions = ['w', 'a', 's', 'd']
        current_direction = 0
        move_count = 0
        
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue
                
            # Ekran görüntüsü al (tam ekran)
            screenshot = pyautogui.screenshot()
            
            # OpenCV formatına dönüştür (performans optimizasyonu için)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Metin taşlarını tespit et
            annotated_frame, stone_position = self.detect_stones(frame)
            
            # Tespiti ekrana göster
            self.update_frame.emit(annotated_frame)
            
            current_time = time.time()
            
            # Metin taşı tespit edildiğinde ve tıklama aralığı geçtiyse tıkla
            if stone_position and (current_time - last_stone_click) >= self.click_interval:
                center_x, center_y = stone_position
                
                # İmleci metin taşının üzerine getir
                pyautogui.moveTo(center_x, center_y)
                
                # 0-1 saniye arasında rastgele bekle
                wait_time = random.uniform(0, 1)
                self.update_status.emit(f"İmleç metin taşının üzerinde, {wait_time:.1f} saniye bekleniyor...")
                time.sleep(wait_time)
                    
                # Taşa tıkla (tam ortasına)
                pyautogui.click()
                self.update_status.emit(f"Metin taşı tespit edildi! Tam ortasına tıklandı: ({center_x}, {center_y}) - {self.click_interval} saniye sonra tekrar tıklanacak")
                self.stone_detected.emit((center_x, center_y))
                last_stone_click = current_time
            
            # Metin taşı tespit edildiğinde ama tıklama aralığı geçmediyse bekle
            elif stone_position:
                remaining = self.click_interval - (current_time - last_stone_click)
                self.update_status.emit(f"Metin taşı tespit edildi! Tıklamaya kalan süre: {remaining:.1f} saniye")
                time.sleep(0.2)  # CPU yükünü azaltmak için kısa bekleme
                
            # Taş bulunamadığında hareket et
            else:
                # Rotasyon hareketleri
                if move_count % 10 == 0:  # Her 10 adımda bir yön değiştir
                    current_direction = (current_direction + 1) % 4
                
                # Hareket tuşuna bas
                keyboard.press(move_directions[current_direction])
                time.sleep(0.3)
                keyboard.release(move_directions[current_direction])
                
                # Belirli aralıklarla çevreyi tara (E ve Q tuşları)
                if move_count % 5 == 0:
                    scan_key = 'e' if move_count % 10 == 0 else 'q'
                    keyboard.press(scan_key)
                    time.sleep(0.2)
                    keyboard.release(scan_key)
                
                move_count += 1
                self.update_status.emit(f"Metin taşı aranıyor... ({move_directions[current_direction]} tuşu ile hareket)")
            
            # Rastgele Z tuşuna basma
            if current_time - last_z_press > z_interval:
                keyboard.press('z')
                time.sleep(0.1)
                keyboard.release('z')
                last_z_press = current_time
                z_interval = random.uniform(0, 5)  # Yeni rastgele aralık
                self.update_status.emit(f"Z tuşuna basıldı (sonraki basma: {z_interval:.1f} saniye sonra)")
            
            time.sleep(0.05)  # CPU kullanımını azaltmak için kısa bekleme

class MetinBotGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Metin2 Stone Detector - Katz")
        self.setMinimumSize(1000, 700)
        
        # Config dosya yolu
        self.config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot_config.json")
        
        # Style için
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121212;
                color: #e0e0e0;
            }
            QLabel {
                color: #e0e0e0;
                font-size: 12px;
                margin: 2px;
            }
            QPushButton {
                background-color: #1565C0;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 4px;
                font-weight: bold;
                margin: 4px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
            QPushButton:disabled {
                background-color: #424242;
                color: #9E9E9E;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #424242;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #1565C0;
                border: 1px solid #1565C0;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            QSpinBox {
                background-color: #1E1E1E;
                color: #e0e0e0;
                border: 1px solid #333333;
                border-radius: 4px;
                padding: 4px;
                margin: 2px;
            }
            QFrame#lineFrame {
                background-color: #333333;
                margin: 10px 0px;
            }
            QStatusBar {
                background-color: #1E1E1E;
                color: #e0e0e0;
                padding: 4px;
            }
            QGroupBox {
                border: 1px solid #333333;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                font-weight: bold;
                color: #e0e0e0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                color: #1976D2;
            }
            QCheckBox {
                color: #e0e0e0;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid #999999;
                background-color: #1E1E1E;
                border-radius: 2px;
            }
            QCheckBox::indicator:checked {
                border: 1px solid #1565C0;
                background-color: #1976D2;
                border-radius: 2px;
            }
        """)
        
        # Ana widget ve layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(12)  # Ana bölümler arası boşluk
        
        # Sol panel (kontroller)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)  # Elemanlar arası boşluk
        left_layout.setContentsMargins(10, 15, 10, 15)  # Kenar boşlukları
        left_panel.setMaximumWidth(300)
        
        # Model seçimi (GroupBox içinde)
        model_group = QGroupBox("Model Ayarları")
        model_layout = QVBoxLayout(model_group)
        model_layout.setSpacing(8)
        
        model_path_layout = QHBoxLayout()
        self.model_label = QLabel("YOLOv8 Model:")
        self.model_path = QLabel("Varsayılan (yolov8n.pt)")
        self.model_path.setWordWrap(True)
        model_path_layout.addWidget(self.model_label)
        model_path_layout.addWidget(self.model_path)
        
        self.browse_model_btn = QPushButton("Model Seç")
        self.browse_model_btn.clicked.connect(self.browse_model)
        
        model_layout.addLayout(model_path_layout)
        model_layout.addWidget(self.browse_model_btn)
        left_layout.addWidget(model_group)
        
        # Bot ayarları (GroupBox içinde)
        bot_settings_group = QGroupBox("Bot Ayarları")
        bot_settings_layout = QVBoxLayout(bot_settings_group)
        bot_settings_layout.setSpacing(8)
        
        # Tıklama aralığı ayarı
        click_interval_layout = QHBoxLayout()
        click_interval_label = QLabel("Tıklama Aralığı (sn):")
        self.click_interval_spinbox = QSpinBox()
        self.click_interval_spinbox.setMinimum(1)
        self.click_interval_spinbox.setMaximum(60)
        self.click_interval_spinbox.setValue(3)  # Varsayılan 3 saniye
        self.click_interval_spinbox.valueChanged.connect(self.update_click_interval)
        click_interval_layout.addWidget(click_interval_label)
        click_interval_layout.addWidget(self.click_interval_spinbox)
        bot_settings_layout.addLayout(click_interval_layout)
        
        # Algılama ayarları
        conf_layout = QVBoxLayout()
        conf_header = QHBoxLayout()
        conf_label = QLabel("Algılama Eşik Değeri:")
        self.conf_value = QLabel("0.50")
        self.conf_value.setAlignment(Qt.AlignRight)
        conf_header.addWidget(conf_label)
        conf_header.addWidget(self.conf_value)
        
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(1)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(50)  # Varsayılan %50
        self.conf_slider.valueChanged.connect(self.update_conf_value)
        
        conf_layout.addLayout(conf_header)
        conf_layout.addWidget(self.conf_slider)
        bot_settings_layout.addLayout(conf_layout)
        
        # Arayüz ayarları
        ui_layout = QVBoxLayout()
        self.preview_checkbox = QCheckBox("Oyun Önizlemesini Göster")
        self.preview_checkbox.setChecked(True)
        self.preview_checkbox.stateChanged.connect(self.toggle_preview)
        ui_layout.addWidget(self.preview_checkbox)
        bot_settings_layout.addLayout(ui_layout)
        
        left_layout.addWidget(bot_settings_group)
        
        # Ayarlar butonu grubu
        settings_group = QGroupBox("Ayarlar Yönetimi")
        settings_layout = QHBoxLayout(settings_group)
        
        # Ayarları Kaydet Butonu
        self.save_settings_btn = QPushButton("Kaydet")
        self.save_settings_btn.clicked.connect(self.save_settings)
        settings_layout.addWidget(self.save_settings_btn)
        
        # Ayarları Yükle Butonu
        self.load_settings_btn = QPushButton("Yükle")
        self.load_settings_btn.clicked.connect(self.load_settings)
        settings_layout.addWidget(self.load_settings_btn)
        
        left_layout.addWidget(settings_group)
        
        # Kısayollar (GroupBox içinde)
        shortcuts_group = QGroupBox("Kısayol Tuşları")
        shortcuts_layout = QVBoxLayout(shortcuts_group)
        shortcuts_info = QLabel("F1: Başlat\nF2: Duraklat\nESC: Durdur")
        shortcuts_info.setAlignment(Qt.AlignCenter)
        shortcuts_layout.addWidget(shortcuts_info)
        left_layout.addWidget(shortcuts_group)
        
        # Bot kontrol düğmeleri (GroupBox içinde)
        ctrl_group = QGroupBox("Bot Kontrolü")
        ctrl_layout = QVBoxLayout(ctrl_group)
        
        self.start_btn = QPushButton("Başlat (F1)")
        self.start_btn.clicked.connect(self.start_bot)
        self.start_btn.setMinimumHeight(40)
        
        pause_stop_layout = QHBoxLayout()
        self.pause_btn = QPushButton("Duraklat (F2)")
        self.pause_btn.clicked.connect(self.pause_bot)
        self.pause_btn.setEnabled(False)
        
        self.stop_btn = QPushButton("Durdur (ESC)")
        self.stop_btn.clicked.connect(self.stop_bot)
        self.stop_btn.setEnabled(False)
        
        pause_stop_layout.addWidget(self.pause_btn)
        pause_stop_layout.addWidget(self.stop_btn)
        
        ctrl_layout.addWidget(self.start_btn)
        ctrl_layout.addLayout(pause_stop_layout)
        left_layout.addWidget(ctrl_group)
        
        # Sağ panel (görüntüleme)
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        self.right_layout.setContentsMargins(10, 15, 10, 15)
        
        # Görüntü etiketi için grup kutusu
        display_group = QGroupBox("Oyun Önizleme")
        display_layout = QVBoxLayout(display_group)
        
        # Görüntü etiketi
        self.image_label = QLabel("Bot başlatıldığında görüntü burada gösterilecek")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid #333333; background-color: #1E1E1E;")
        self.image_label.setMinimumHeight(400)
        
        # Kaydırma alanı
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.image_label)
        display_layout.addWidget(self.scroll_area)
        
        # Durum mesajları için grup kutusu
        status_group = QGroupBox("Bot Durumu")
        status_layout = QVBoxLayout(status_group)
        
        # Log durumu
        self.log_label = QLabel("Durum: Hazır")
        self.log_label.setWordWrap(True)
        self.log_label.setMinimumHeight(40)
        self.log_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        status_layout.addWidget(self.log_label)
        
        # Ana düzene grupları ekle
        self.right_layout.addWidget(display_group, 4)  # 4 birim yükseklik
        self.right_layout.addWidget(status_group, 1)   # 1 birim yükseklik
        
        # Ana düzene panelleri ekle
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(self.right_panel, 3)
        
        # Durum çubuğu
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Bot hazır")
        
        # Kısayol tuşları için global hook
        keyboard.add_hotkey('f1', self.start_bot)
        keyboard.add_hotkey('f2', self.pause_bot)
        keyboard.add_hotkey('esc', self.stop_bot)
        
        # YOLO worker ve değişkenler
        self.yolo_worker = YOLOWorker()
        self.yolo_worker.update_frame.connect(self.update_display)
        self.yolo_worker.update_status.connect(self.update_status)
        self.yolo_worker.stone_detected.connect(self.stone_detected)
        
        # Ayarları yükle (varsa)
        self.load_settings()

    def toggle_preview(self, state):
        # Önizleme ekranını aç/kapa
        display_group = self.findChild(QGroupBox, "")
        if display_group:
            for i in range(self.right_layout.count()):
                item = self.right_layout.itemAt(i)
                if item and item.widget() and isinstance(item.widget(), QGroupBox) and "Önizleme" in item.widget().title():
                    item.widget().setVisible(state == Qt.Checked)
                    # Pencere boyutunu güncelle
                    if state == Qt.Checked:
                        self.setMinimumSize(1000, 700)
                    else:
                        self.setMinimumSize(800, 300)
                    break
        
        # Config'e kaydet
        self.save_settings()

    def update_display(self, frame):
        # Önizleme kapalıysa güncelleme
        if not self.preview_checkbox.isChecked():
            return
            
        # OpenCV frame'i QImage'e dönüştür (performans optimizasyonu)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
        
        # QImage'i QLabel'a uygula
        pixmap = QPixmap.fromImage(image)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), self.image_label.height(), 
                                              Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_status(self, message):
        self.log_label.setText(f"Durum: {message}")
        self.status_bar.showMessage(message)
        
    def stone_detected(self, pos):
        x, y = pos
        self.status_bar.showMessage(f"Metin taşı tespit edildi! Konumu: ({x}, {y})")

    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "YOLOv8 Model Seç", "", "PT Files (*.pt)")
        if file_path:
            self.yolo_worker.model_path = file_path
            self.model_path.setText(os.path.basename(file_path))
            self.status_bar.showMessage(f"Model seçildi: {file_path}")

    def update_conf_value(self):
        value = self.conf_slider.value() / 100.0
        self.conf_value.setText(f"{value:.2f}")
        self.yolo_worker.set_confidence(value)
        
    def update_click_interval(self):
        value = self.click_interval_spinbox.value()
        self.yolo_worker.set_click_interval(value)
        self.status_bar.showMessage(f"Tıklama aralığı {value} saniye olarak ayarlandı")

    def save_settings(self):
        settings = {
            "model_path": self.yolo_worker.model_path,
            "confidence": self.conf_slider.value() / 100.0,
            "click_interval": self.click_interval_spinbox.value(),
            "show_preview": self.preview_checkbox.isChecked()
        }
        
        try:
            with open(self.config_file, "w") as f:
                json.dump(settings, f)
            self.update_status(f"Ayarlar kaydedildi: {self.config_file}")
        except Exception as e:
            self.update_status(f"Ayarlar kaydedilirken hata oluştu: {str(e)}")
            
    def load_settings(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    settings = json.load(f)
                
                # Model yolunu ayarla
                if "model_path" in settings:
                    self.yolo_worker.model_path = settings["model_path"]
                    self.model_path.setText(os.path.basename(settings["model_path"]))
                
                # Güven değerini ayarla
                if "confidence" in settings:
                    conf_value = int(settings["confidence"] * 100)
                    self.conf_slider.setValue(conf_value)
                    self.conf_value.setText(f"{settings['confidence']:.2f}")
                    self.yolo_worker.set_confidence(settings["confidence"])
                
                # Tıklama aralığını ayarla
                if "click_interval" in settings:
                    self.click_interval_spinbox.setValue(settings["click_interval"])
                    self.yolo_worker.set_click_interval(settings["click_interval"])
                
                # Önizlemeyi ayarla
                if "show_preview" in settings:
                    self.preview_checkbox.setChecked(settings["show_preview"])
                    self.toggle_preview(Qt.Checked if settings["show_preview"] else Qt.Unchecked)
                
                self.update_status("Ayarlar yüklendi")
                return True
            except Exception as e:
                self.update_status(f"Ayarlar yüklenirken hata oluştu: {str(e)}")
                return False
        else:
            self.update_status("Ayar dosyası bulunamadı, varsayılan ayarlar kullanılıyor")
            return False

    def start_bot(self):
        if self.yolo_worker.isRunning() and self.yolo_worker.paused:
            # Eğer duraklatılmışsa devam et
            self.yolo_worker.paused = False
            self.update_status("Bot devam ediyor...")
        elif not self.yolo_worker.isRunning():
            # Eğer çalışmıyorsa başlat
            self.yolo_worker.start()
            self.update_status("Bot başlatıldı...")
            
        # Düğme durumlarını güncelle
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)

    def pause_bot(self):
        if self.yolo_worker.isRunning():
            self.yolo_worker.paused = True
            self.update_status("Bot duraklatıldı")
            
            # Düğme durumlarını güncelle
            self.start_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)

    def stop_bot(self):
        if self.yolo_worker.isRunning():
            self.yolo_worker.running = False
            self.yolo_worker.wait()
            self.update_status("Bot durduruldu")
            
            # Düğme durumlarını güncelle
            self.start_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)

    def closeEvent(self, event):
        # Çıkmadan önce ayarları kaydet
        self.save_settings()
        
        # Uygulamayı kapatmadan önce thread'i durdur
        if self.yolo_worker.isRunning():
            self.yolo_worker.running = False
            self.yolo_worker.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MetinBotGUI()
    window.show()
    sys.exit(app.exec_())