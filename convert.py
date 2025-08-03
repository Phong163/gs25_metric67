from ultralytics import YOLO

# Tải mô hình YOLO11n-seg
model = YOLO(r"C:\Users\OS\Desktop\gs25_metric67\weights\yolo11n-seg.pt")  # Tự động tải từ Ultralytics release

# Xuất mô hình sang ONNX
model.export(format="onnx", imgsz=[480, 480], opset=12)  # opset=12 được khuyến nghị