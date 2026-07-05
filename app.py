from flask import Flask, request, jsonify, render_template, send_from_directory
import cv2
import numpy as np
from ultralytics import YOLO
import sqlite3
import json
from datetime import datetime
import os
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB limit

# Создание папки для загрузок
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Загрузка модели YOLOv8 (nano версия для скорости)
print("Загрузка модели YOLOv8n...")
model = YOLO('yolov8n.pt')
print("Модель загружена успешно!")

# Максимальное количество зарядных мест
MAX_CHARGING_SPOTS = 10

# Инициализация базы данных
def init_db():
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            input_image TEXT NOT NULL,
            output_image TEXT NOT NULL,
            cars_count INTEGER NOT NULL,
            total_objects INTEGER NOT NULL,
            detections_json TEXT NOT NULL,
            processing_time REAL NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/process', methods=['POST'])
def process_image():
    start_time = datetime.now()
    
    # Проверка наличия файла
    if 'image' not in request.files:
        return jsonify({'error': 'Файл не загружен'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400
    
    # Генерация уникального ID запроса
    request_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Сохранение загруженного файла
    input_filename = f'input_{request_id}_{timestamp}.jpg'
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
    file.save(input_path)
    
    # Чтение изображения
    img = cv2.imread(input_path)
    if img is None:
        return jsonify({'error': 'Не удалось прочитать изображение'}), 400
    
    # Инференс модели
    results = model(img)
    
    # Получение обнаруженных объектов
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        
        detections.append({
            'class': class_name,
            'class_id': class_id,
            'confidence': round(confidence, 3),
            'bbox': [x1, y1, x2, y2]
        })
    
    # Фильтрация: оставляем только автомобили (class_id = 2 в COCO)
    cars = [d for d in detections if d['class_id'] == 2]  # 'car'
    
    # Визуализация результатов
    output_img = results[0].plot()
    output_filename = f'output_{request_id}_{timestamp}.jpg'
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    cv2.imwrite(output_path, output_img)
    
    # Расчет времени обработки
    processing_time = (datetime.now() - start_time).total_seconds()
    
    # Сохранение в историю
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO requests 
        (request_id, timestamp, input_image, output_image, cars_count, total_objects, detections_json, processing_time)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        request_id,
        datetime.now().isoformat(),
        input_filename,
        output_filename,
        len(cars),
        len(detections),
        json.dumps(detections),
        round(processing_time, 3)
    ))
    conn.commit()
    conn.close()
    
    return jsonify({
        'success': True,
        'request_id': request_id,
        'cars_count': len(cars),
        'total_objects': len(detections),
        'free_spots': max(0, MAX_CHARGING_SPOTS - len(cars)),
        'output_image': output_filename,
        'input_image': input_filename,
        'processing_time': round(processing_time, 3),
        'detections': detections
    })

@app.route('/history', methods=['GET'])
def get_history():
    limit = request.args.get('limit', 50, type=int)
    
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, request_id, timestamp, input_image, output_image, cars_count, total_objects, processing_time
        FROM requests
        ORDER BY id DESC
        LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            'id': row[0],
            'request_id': row[1],
            'timestamp': row[2],
            'input_image': row[3],
            'output_image': row[4],
            'cars_count': row[5],
            'total_objects': row[6],
            'processing_time': row[7]
        })
    
    return jsonify(history)

@app.route('/stats', methods=['GET'])
def get_stats():
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    
    # Общее количество запросов
    cursor.execute('SELECT COUNT(*) FROM requests')
    total_requests = cursor.fetchone()[0]
    
    # Среднее количество автомобилей
    cursor.execute('SELECT AVG(cars_count) FROM requests')
    avg_cars = cursor.fetchone()[0] or 0
    
    # Максимальное количество автомобилей
    cursor.execute('SELECT MAX(cars_count) FROM requests')
    max_cars = cursor.fetchone()[0] or 0
    
    # Запросы за последние 24 часа
    cursor.execute('''
        SELECT COUNT(*) FROM requests 
        WHERE datetime(timestamp) > datetime('now', '-1 day')
    ''')
    last_24h = cursor.fetchone()[0]
    
    conn.close()
    
    return jsonify({
        'total_requests': total_requests,
        'avg_cars': round(avg_cars, 2),
        'max_cars': max_cars,
        'last_24h': last_24h,
        'max_spots': MAX_CHARGING_SPOTS
    })

@app.route('/delete_history', methods=['DELETE'])
def delete_history():
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM requests')
    conn.commit()
    conn.close()
    return jsonify({'success': True, 'message': 'История очищена'})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚗 УЧЕТ ЭЛЕКТРОМОБИЛЕЙ НА ЗАРЯДНОЙ СТАНЦИИ")
    print("="*50)
    print(f"📍 Максимальное количество мест: {MAX_CHARGING_SPOTS}")
    print(f"🔗 Откройте в браузере: http://localhost:5000")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)