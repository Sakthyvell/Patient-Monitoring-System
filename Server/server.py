import cv2
import imagezmq
from time import sleep
from datetime import datetime
image_hub = imagezmq.ImageHub()

while True:  # show streamed images until Ctrl-C
    rpi_name, image = image_hub.recv_image()
    dt_string = datetime.now().strftime("%d%m%Y_%H:%M:%S")
    filename = f"/home/sakthy1497/Documents/ED_Project/Patient_Monitoring_System/Server/Images/Patient1_{dt_string}.png"
    cv2.imwrite(filename, image)
    cv2.waitKey(1)
    image_hub.send_reply(b'OK')
    print(f"Image received at {dt_string}")
    sleep(3)