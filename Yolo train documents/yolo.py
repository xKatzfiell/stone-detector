from ultralytics import YOLO  

if __name__ == '__main__':   
    # Önceden eğitilmiş model kullanın (transfer öğrenme)
    model = YOLO('yolov8n.pt')

    # Veri artırma seçenekleriyle eğitim
    results = model.train(
        data='C:/Users/Katzfiell/Desktop/stone/data.yaml',
        epochs=300,        # Daha fazla epoch
        imgsz=640,
        patience=300,       # Sabırlı olun
        batch=8,           # Küçük batch size
        augment=True,      # Veri artırma açık
        degrees=20,        # 20 dereceye kadar döndürme
        translate=0.2,     # Kaydırma
        workers=6,
        scale=0.2,         # Ölçekleme
        fliplr=0.5,        # %50 olasılıkla yatay çevirme
        mosaic=1.0,        # Mozaik artırma
        mixup=0.2,         # Karıştırma
        copy_paste=0.2,    # Kopyala-yapıştır
        save=True,
        save_period=50     # Her 50 epoch'ta model kaydet
    )
