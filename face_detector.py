# сюда пишет Маша
"""
Модуль обнаружения лиц для проекта face-recognition-app
Автор: Участник 2
Функции: Обнаружение лиц, рисование прямоугольников, подготовка данных для распознавания
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Optional, Dict
import os

class FaceDetector:
    """
    Класс для обнаружения лиц на видео с камеры
    """
    
    def __init__(self, cascade_path: str = None):
        """
        Инициализация детектора лиц
        
        Args:
            cascade_path: Путь к файлу каскада Хаара.
                         Если None, используется встроенный в OpenCV.
        """
        self.cascade_path = cascade_path
        self.face_cascade = self._load_cascade()
        
        # Статистика
        self.total_faces_detected = 0
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Данные для других модулей
        self.face_regions = []  # Список текущих областей лиц
        self.face_images = []   # Список изображений лиц
        self.last_detection_time = 0
        
        print("✅ Детектор лиц инициализирован")
    
    def _load_cascade(self):
        """Загрузка каскада Хаара для обнаружения лиц"""
        try:
            if self.cascade_path and os.path.exists(self.cascade_path):
                cascade = cv2.CascadeClassifier(self.cascade_path)
            else:
                # Используем встроенный каскад OpenCV
                cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
            
            if cascade.empty():
                # Альтернативный путь
                cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                if cascade.empty():
                    raise ValueError("Не удалось загрузить каскад Хаара")
            
            return cascade
        except Exception as e:
            print(f"⚠️ Ошибка загрузки каскада: {e}")
            print("Скачайте файл с: https://github.com/opencv/opencv/tree/master/data/haarcascades")
            raise
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Обнаружение лиц на кадре
        
        Args:
            frame: Входной кадр (BGR)
            
        Returns:
            Список прямоугольников (x, y, width, height) для каждого лица
        """
        # Конвертируем в оттенки серого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Улучшаем контраст для лучшего обнаружения
        gray = cv2.equalizeHist(gray)
        
        # Обнаружение лиц
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Сохраняем информацию для других модулей
        self.face_regions = faces.tolist() if len(faces) > 0 else []
        self.face_images = []
        
        # Извлекаем изображения лиц с отступами
        for (x, y, w, h) in self.face_regions:
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            face_img = frame[y1:y2, x1:x2]
            if face_img.size > 0:
                # Приводим к стандартному размеру для распознавания
                standard_size = (160, 160)
                face_img_resized = cv2.resize(face_img, standard_size)
                self.face_images.append(face_img_resized)
        
        # Обновляем статистику
        self.total_faces_detected += len(faces)
        self.frame_count += 1
        self.last_detection_time = time.time()
        
        return faces
    
    def draw_faces(self, frame: np.ndarray, faces: List[Tuple], 
                   names: List[str] = None, confidence: List[float] = None) -> np.ndarray:
        """
        Рисование прямоугольников вокруг лиц с возможностью добавления имен
        
        Args:
            frame: Исходный кадр
            faces: Список прямоугольников лиц
            names: Список имен для каждого лица (опционально)
            confidence: Список уверенностей распознавания (опционально)
            
        Returns:
            Кадр с нарисованными прямоугольниками
        """
        frame_copy = frame.copy()
        
        for i, (x, y, w, h) in enumerate(faces):
            # Выбираем цвет в зависимости от имени
            if names and i < len(names):
                if names[i] == "Unknown" or names[i] == "Неизвестный":
                    color = (0, 0, 255)  # Красный для неизвестных
                else:
                    color = (0, 255, 0)  # Зеленый для известных
            else:
                color = (0, 255, 0)  # Зеленый по умолчанию
            
            thickness = 2
            
            # Рисуем прямоугольник
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), color, thickness)
            
            # Подготавливаем текст для отображения
            if names and i < len(names):
                name_text = names[i]
                if confidence and i < len(confidence):
                    name_text += f" ({confidence[i]:.2f})"
                
                # Рисуем фон для текста
                text_size = cv2.getTextSize(name_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(frame_copy, 
                             (x, y - 25), 
                             (x + text_size[0] + 10, y), 
                             color, 
                             -1)
                
                # Рисуем текст
                cv2.putText(frame_copy, name_text, (x + 5, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                # Без имени - просто номер
                cv2.putText(frame_copy, f'Face #{i+1}', (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Рисуем точку в центре лица
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(frame_copy, (center_x, center_y), 3, (255, 255, 0), -1)
        
        return frame_copy
    
    def get_face_regions(self) -> List[Tuple[int, int, int, int]]:
        """Получение текущих регионов лиц"""
        return self.face_regions
    
    def get_face_images(self) -> List[np.ndarray]:
        """Получение текущих изображений лиц"""
        return self.face_images
    
    def get_face_data(self) -> Dict:
        """
        Получение полных данных о текущих лицах
        
        Returns:
            Словарь с данными о лицах
        """
        return {
            'count': len(self.face_regions),
            'regions': self.face_regions,
            'images': self.face_images,
            'timestamp': time.time(),
            'frame_count': self.frame_count,
            'total_detected': self.total_faces_detected
        }
    
    def calculate_fps(self) -> float:
        """Расчет FPS (кадров в секунду)"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        if elapsed_time > 0:
            self.fps = self.frame_count / elapsed_time
        
        return self.fps
    
    def get_statistics(self) -> Dict[str, any]:
        """Получение статистики"""
        return {
            'fps': self.calculate_fps(),
            'current_faces': len(self.face_regions),
            'total_faces': self.total_faces_detected,
            'frame_count': self.frame_count,
            'uptime': time.time() - self.start_time
        }
    
    def reset_statistics(self):
        """Сброс статистики"""
        self.total_faces_detected = 0
        self.frame_count = 0
        self.start_time = time.time()
    
    def save_faces(self, output_dir: str = "detected_faces") -> List[str]:
        """
        Сохранение текущих лиц в файлы
        
        Args:
            output_dir: Директория для сохранения
            
        Returns:
            Список путей к сохраненным файлам
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        for i, face_img in enumerate(self.face_images):
            if face_img is not None and face_img.size > 0:
                filename = os.path.join(output_dir, f"face_{timestamp}_{i+1}.jpg")
                cv2.imwrite(filename, face_img)
                saved_paths.append(filename)
        
        return saved_paths
    
    def preprocess_for_recognition(self, face_image: np.ndarray) -> np.ndarray:
        """
        Предобработка изображения лица для распознавания
        
        Args:
            face_image: Изображение лица
            
        Returns:
            Предобработанное изображение
        """
        # Конвертация в оттенки серого
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Нормализация
        normalized = gray.astype('float32') / 255.0
        
        # Расширение размерности для нейронных сетей
        if len(normalized.shape) == 2:
            normalized = np.expand_dims(normalized, axis=-1)
        
        return normalized

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def create_detector(config: Dict = None) -> FaceDetector:
    """
    Фабричная функция для создания детектора
    
    Args:
        config: Конфигурация детектора
        
    Returns:
        Экземпляр FaceDetector
    """
    if config and 'cascade_path' in config:
        return FaceDetector(cascade_path=config['cascade_path'])
    return FaceDetector()

def test_detection_on_image(image_path: str, show_result: bool = True):
    """
    Тестирование обнаружения на изображении
    
    Args:
        image_path: Путь к изображению
        show_result: Показывать результат
    """
    detector = FaceDetector()
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Не удалось загрузить изображение: {image_path}")
        return
    
    faces = detector.detect_faces(image)
    
    print(f"Обнаружено лиц: {len(faces)}")
    
    if show_result:
        result = detector.draw_faces(image, faces)
        cv2.imshow('Face Detection Test', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return faces

def benchmark_detector(detector: FaceDetector, test_frames: int = 100):
    """
    Бенчмарк производительности детектора
    
    Args:
        detector: Детектор для тестирования
        test_frames: Количество тестовых кадров
    """
    # Создаем тестовые кадры
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    start_time = time.time()
    
    for i in range(test_frames):
        detector.detect_faces(test_frame)
    
    end_time = time.time()
    
    total_time = end_time - start_time
    fps = test_frames / total_time
    
    print(f"Бенчмарк завершен:")
    print(f"  Кадров: {test_frames}")
    print(f"  Время: {total_time:.2f} сек")
    print(f"  FPS: {fps:.2f}")
    
    return fps

# ==================== ТОЧКА ВХОДА ДЛЯ ТЕСТИРОВАНИЯ ====================

if __name__ == "__main__":
    print("=" * 50)
    print("Тестирование модуля обнаружения лиц")
    print("=" * 50)
    
    # Тест 1: Создание детектора
    print("\n1. Тестирование создания детектора...")
    try:
        detector = FaceDetector()
        print("✅ Детектор создан успешно")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        exit(1)
    
    # Тест 2: Бенчмарк
    print("\n2. Запуск бенчмарка производительности...")
    fps = benchmark_detector(detector, 50)
    
    # Тест 3: Проверка работы с камерой
    print("\n3. Тестирование с веб-камерой (5 секунд)...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Не удалось открыть камеру")
    else:
        start = time.time()
        while time.time() - start < 5:
            ret, frame = cap.read()
            if ret:
                faces = detector.detect_faces(frame)
                frame_with_faces = detector.draw_faces(frame, faces)
                cv2.imshow('Test Camera', frame_with_faces)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Тест с камерой завершен")
    
    print("\n" + "=" * 50)
    print("Все тесты завершены")
    print("=" * 50)

from face_detector import FaceDetector, create_detector

# Создание детектора
detector = FaceDetector()

# Использование
while True:
    ret, frame = cap.read()
    faces = detector.detect_faces(frame)
    frame_with_faces = detector.draw_faces(frame, faces)
    
    # Получение данных для распознавания
    face_data = detector.get_face_data()
