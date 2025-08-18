from datetime import datetime
import json
import cv2
import numpy as np
import pytz
import torch
from ultralytics.utils.metrics import box_iou
from confluent_kafka import Producer

import logging
import logging.handlers
logger = logging.getLogger('metric_67_gs25')

# Biến toàn cục để lưu trữ instance Producer
_producer = None
_producer_config = {
    'bootstrap.servers': 'hcm.gateway01.cxview.ai:9094',
    'client.id': 'python-producer',
    'security.protocol': 'SSL',
    'ssl.ca.location': './cert/ca-root.pem',
    'ssl.certificate.location': './cert/ca-cert.pem',
    'ssl.key.location': './cert/ca-key.pem',
    'retries': 100,  # Thêm retry để tăng độ tin cậy
    'retry.backoff.ms': 1000,  # Chờ 1 giây giữa các lần thử
    'debug': 'security'
}
def setup_logger(
    logger_name='metric_67_gs25',
    log_file='output/metric_67_gs25.log',
    level=logging.DEBUG,
    max_bytes=10*1024*1024,
    backup_count=5,
    log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
):
    """
    Thiết lập logger với console và file handler (rotation).
    
    Args:
        logger_name (str): Tên của logger.
        log_file (str): Đường dẫn file log.
        level (int): Mức độ log.
        max_bytes (int): Kích thước tối đa của file log.
        backup_count (int): Số file backup tối đa.
        log_format (str): Định dạng log.
    
    Returns:
        logging.Logger: Đối tượng logger đã được cấu hình.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False
    
    formatter = logging.Formatter(log_format)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def get_zone_coords(frame, zone_relative):
    height, width = frame.shape[:2]
    points_abs = [(int(x * width), int(y * height)) for x, y in zone_relative]
    return np.array([points_abs], dtype=np.int32)

def is_box_in_zone(box, zone_coords, score):
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    # Kiểm tra xem tâm của box có nằm trong vùng không
    if cv2.pointPolygonTest(zone_coords[0], (center_x, center_y), False) >= 0:
        return True
    # Kiểm tra IoU như một phương pháp dự phòng
    box_area = [(x1, y1, x2, y2)]
    zone_points = zone_coords[0]
    zone_x = [point[0] for point in zone_points]
    zone_y = [point[1] for point in zone_points]
    zone_x1, zone_y1 = min(zone_x), min(zone_y)
    zone_x2, zone_y2 = max(zone_x), max(zone_y)
    zone_area = [(zone_x1, zone_y1, zone_x2, zone_y2)]
    box_xyxy = np.array(box_area)
    zone_xyxy = np.array(zone_area)
    iou = box_iou(torch.tensor(box_xyxy), torch.tensor(zone_xyxy)).item()
    return iou > score

def rescale(frame, img_size, x_min, y_min, x_max, y_max):
    scale_x = frame.shape[1] / img_size
    scale_y = frame.shape[0] / img_size
    return x_min * scale_x, y_min * scale_y, x_max * scale_x, y_max * scale_y


def get_producer():
    """Lấy hoặc khởi tạo Producer một lần duy nhất."""
    global _producer
    if _producer is None:
        try:
            _producer = Producer(_producer_config)
            logger.info("Kafka Producer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Kafka Producer: {str(e)}")
            raise
    return _producer

def send_time_to_kafka(zone_id, customer_id, interaction_quantity, bootstrap_servers="hcm.gateway01.cxview.ai:9094", topic="production-gatewayhn-peoplecount"):
    """Gửi thông điệp đến Kafka topic, sử dụng Producer chung."""
    # Lấy timestamp theo múi giờ Asia/Ho_Chi_Minh
    dt = datetime.now(pytz.timezone("Asia/Ho_Chi_Minh")).timestamp()
    timestamp = int(dt)
    data = {
        "box_id": "d43050fd-7296-4409-ad54-757af392a0d2",   #str
        "metric": 6, 
        "customer_id": customer_id, #str
        "zone_id" : zone_id,
        "interaction_quantity": interaction_quantity, #int
        "timestamp": timestamp
    }
    def delivery_report(err, msg):
        """Hàm callback để báo cáo trạng thái gửi tin nhắn."""
        if err is not None:
            logger.error(f"Error sending message: {err}")
        else:
            logger.info(f"Completed to send zone_id: {zone_id}, customer: {customer_id}, interaction_quantity: {interaction_quantity}, Topic: {msg.topic()}, Partition: {msg.partition()}, Offset: {msg.offset()}")

    try:
        # Lấy Producer
        producer = get_producer()
        # Chuyển dữ liệu thành JSON string
        message = json.dumps(data)
        # Gửi tin nhắn đến topic Kafka
        producer.produce(topic, value=message.encode('utf-8'), callback=delivery_report)
        # Gọi poll để xử lý callback, không gọi flush thường xuyên để tối ưu hiệu suất
        producer.poll(0)
    except Exception as e:
        logger.error(f"Error sending message to Kafka: {str(e)}")
        raise

def close_producer():
    """Đóng Producer khi ứng dụng kết thúc."""
    global _producer
    if _producer is not None:
        try:
            _producer.flush()  # Đảm bảo tất cả thông điệp được gửi trước khi đóng
            logger.info("Kafka Producer closed successfully")
        except Exception as e:
            logger.error(f"Error closing Kafka Producer: {str(e)}")
        finally:
            _producer = None

def make_hand(box, side="right"):
    x1, y1, x2, y2 = box
    a = x2 - x1
    b = y2 - y1
    x1_hand = int(x1 + (a * 0.8))
    y1_hand = int(b * 0.2 + y1)
    x2_hand = int(x2)
    y2_hand = int(b * 0.4 + y1)

    return (x1_hand, y1_hand, x2_hand, y2_hand) 
def extract_feature(extractor, image, box):
    x1, y1, x2, y2 = box
    crop = image[y1:y2, x1:x2]
    if crop.size > 0:
        # Chuyển đổi hình ảnh crop sang tensor và đảm bảo nó nằm trên CPU
        crop_tensor = torch.from_numpy(crop).permute(2, 0, 1).float().unsqueeze(0)
        if torch.cuda.is_available():
            crop_tensor = crop_tensor.cuda()  # Di chuyển tensor sang GPU nếu có
        feature = extractor(crop_tensor)
        # Di chuyển feature về CPU trước khi chuyển sang NumPy
        feature = feature.cpu().detach().numpy()
        return feature, crop
    return None, None

def cosine_similarity(feat1, feat2):
    # Làm phẳng mảng nếu là 2D
    feat1 = feat1.flatten() if len(feat1.shape) > 1 else feat1
    feat2 = feat2.flatten() if len(feat2.shape) > 1 else feat2
    # Tính tích vô hướng và chuẩn
    dot_product = np.dot(feat1, feat2)
    norm_product = np.linalg.norm(feat1) * np.linalg.norm(feat2)
    # Tránh chia cho 0
    return dot_product / norm_product if norm_product != 0 else 0.0

