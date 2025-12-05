# сюда пишет Маша
import cv2
import numpy as np
import time
from typing import List, Tuple, Optional

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
        
        # Параметры для участника 3
        self.face_regions = []  # Список текущих областей лиц
        self.face_images = []   # Список изображений лиц (для передачи участнику 3)
        
        print("✅ Детектор лиц инициализирован")
    
    def _load_cascade(self):
        """Загрузка каскада Хаара для обнаружения лиц"""
        if self.cascade_path:
            cascade = cv2.CascadeClassifier(self.cascade_path)
        else:
            # Используем встроенный каскад OpenCV
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        
        if cascade.empty():
            raise ValueError("❌ Не удалось загрузить каскад Хаара. Проверьте путь к файлу.")
        
        return cascade
    
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
            scaleFactor=1.1,      # На сколько уменьшаем изображение
            minNeighbors=5,       # Минимальное количество соседей
            minSize=(40, 40),     # Минимальный размер лица
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Сохраняем информацию для участника 3
        self.face_regions = faces.tolist() if len(faces) > 0 else []
        self.face_images = []
        
        # Извлекаем изображения лиц
        for (x, y, w, h) in self.face_regions:
            # Добавляем небольшой отступ
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            face_img = frame[y1:y2, x1:x2]
            if face_img.size > 0:
                self.face_images.append(face_img)
        
        # Обновляем статистику
        self.total_faces_detected += len(faces)
        self.frame_count += 1
        
        return faces
    
    def draw_faces(self, frame: np.ndarray, faces: List[Tuple]) -> np.ndarray:
        """
        Рисование прямоугольников вокруг лиц
        
        Args:
            frame: Исходный кадр
            faces: Список прямоугольников лиц
            
        Returns:
            Кадр с нарисованными прямоугольниками
        """
        frame_copy = frame.copy()
        
        for i, (x, y, w, h) in enumerate(faces):
            # Рисуем зеленый прямоугольник
            color = (0, 255, 0)  # Зеленый (BGR)
            thickness = 2
            
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), color, thickness)
            
            # Добавляем номер лица (для отладки)
            cv2.putText(frame_copy, f'Face #{i+1}', (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Рисуем точку в центре лица
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(frame_copy, (center_x, center_y), 3, (0, 0, 255), -1)
        
        return frame_copy
    
    def calculate_fps(self) -> float:
        """Расчет FPS (кадров в секунду)"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        if elapsed_time > 0:
            self.fps = self.frame_count / elapsed_time
        
        return self.fps
    
    def add_debug_info(self, frame: np.ndarray) -> np.ndarray:
        """
        Добавление отладочной информации на кадр
        
        Args:
            frame: Входной кадр
            
        Returns:
            Кадр с отладочной информацией
        """
        frame_copy = frame.copy()
        
        # Расчет FPS
        fps = self.calculate_fps()
        
        # Добавляем информацию
        info_lines = [
            f"FPS: {fps:.1f}",
            f"Faces: {len(self.face_regions)}",
            f"Total detected: {self.total_faces_detected}",
            "Press 'q' to quit",
            "Press 's' to save face"
        ]
        
        y_offset = 30
        for line in info_lines:
            cv2.putText(frame_copy, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25
        
        return frame_copy
    
    def get_face_data(self) -> dict:
        """
        Получение данных о текущих лицах для передачи участнику 3
        
        Returns:
            Словарь с данными о лицах:
            {
                'count': количество лиц,
                'regions': координаты лиц,
                'images': изображения лиц,
                'timestamp': время обнаружения
            }
        """
        return {
            'count': len(self.face_regions),
            'regions': self.face_regions,
            'images': self.face_images,
            'timestamp': time.time(),
            'frame_count': self.frame_count
        }
    
    def reset_statistics(self):
        """Сброс статистики"""
        self.total_faces_detected = 0
        self.frame_count = 0
        self.start_time = time.time()

# ==================== ФУНКЦИИ ДЛЯ ЭКСПОРТА ====================

def run_face_detection(camera_id: int = 0, window_name: str = "Face Detector"):
    """
    Запуск обнаружения лиц с веб-камеры
    
    Args:
        camera_id: ID камеры (0 - встроенная камера)
        window_name: Название окна
    """
    # Инициализация детектора
    detector = FaceDetector()
    
    # Открытие видеопотока
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ Ошибка: не удалось открыть камеру {camera_id}")
        return
    
    # Настройка параметров камеры (опционально)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("🚀 Запуск обнаружения лиц...")
    print("📌 Горячие клавиши:")
    print("   'q' - выход")
    print("   's' - сохранить текущие лица")
    print("   'r' - сбросить статистику")
    print("   'c' - сделать скриншот")
    
    while True:
        # Захват кадра
        ret, frame = cap.read()
        if not ret:
            print("❌ Ошибка при захвате кадра")
            break
        
        # Обнаружение лиц
        faces = detector.detect_faces(frame)
        
        # Рисование прямоугольников
        frame_with_faces = detector.draw_faces(frame, faces)
        
        # Добавление отладочной информации
        frame_with_info = detector.add_debug_info(frame_with_faces)
        
        # Получение данных для участника 3
        face_data = detector.get_face_data()
        
        # Отображение результата
        cv2.imshow(window_name, frame_with_info)
        
        # Обработка клавиш
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):  # Выход
            break
        elif key == ord('s'):  # Сохранение лиц
            save_detected_faces(detector)
        elif key == ord('r'):  # Сброс статистики
            detector.reset_statistics()
            print("📊 Статистика сброшена")
        elif key == ord('c'):  # Скриншот
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, frame_with_info)
            print(f"📸 Скриншот сохранен: {filename}")
    
    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Программа завершена")

def save_detected_faces(detector: FaceDetector, save_dir: str = "detected_faces"):
    """
    Сохранение обнаруженных лиц в файлы
    
    Args:
        detector: Объект детектора лиц
        save_dir: Директория для сохранения
    """
    import os
    
    # Создание директории, если не существует
    os.makedirs(save_dir, exist_ok=True)
    
    face_data = detector.get_face_data()
    
    if face_data['count'] == 0:
        print("⚠️ Нет лиц для сохранения")
        return
    
    # Сохранение каждого лица
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    for i, face_img in enumerate(face_data['images']):
        if face_img is not None and face_img.size > 0:
            filename = os.path.join(save_dir, f"face_{timestamp}_{i+1}.jpg")
            cv2.imwrite(filename, face_img)
            print(f"💾 Лицо #{i+1} сохранено: {filename}")

def test_with_image(image_path: str):
    """
    Тестирование обнаружения лиц на изображении
    
    Args:
        image_path: Путь к изображению
    """
    detector = FaceDetector()
    
    # Загрузка изображения
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Не удалось загрузить изображение: {image_path}")
        return
    
    # Обнаружение лиц
    faces = detector.detect_faces(image)
    
    # Рисование прямоугольников
    result = detector.draw_faces(image, faces)
    
    # Добавление информации
    result = detector.add_debug_info(result)
    
    # Отображение результата
    print(f"📊 Обнаружено лиц: {len(faces)}")
    
    cv2.imshow('Face Detection Test', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ==================== ТОЧКА ВХОДА ====================

if __name__ == "__main__":
    # Демонстрация работы модуля
    print("=" * 50)
    print("🎭 Модуль обнаружения лиц (Участник 2)")
    print("=" * 50)
    
    # Запуск с веб-камеры
    run_face_detection()
    
    # Для тестирования на изображении раскомментируйте:
    # test_with_image("test_image.jpg")
