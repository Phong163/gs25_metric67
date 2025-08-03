import queue
import threading
import time
import cv2

class RTSPStream:
    def __init__(self, url, cam_id):
        self.url = url
        self.cam_id = cam_id
        self.running = True
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.frame_queue = queue.Queue(maxsize=10)
        self.thread = threading.Thread(target=self.read_stream)
        self.thread.daemon = True

    def read_stream(self):
        retries = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print(f"Cam {self.cam_id}: Don't read frame, try reconnect... ({retries + 1})")
                time.sleep(2)
                self.cap.release()
                self.cap = cv2.VideoCapture(self.url)
                retries += 1
                continue
            retries = 0
            if self.frame_queue.full():
                self.frame_queue.get()
            self.frame_queue.put(frame)
        self.cap.release()

    def start(self):
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def get_frame(self):
        try:
            return True, self.frame_queue.get(timeout=1)
        except queue.Empty:
            return False, None