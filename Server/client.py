from imutils.video import VideoStream
import imagezmq
import argparse
import socket
import time
from datetime import datetime
import logging

parser = argparse.ArgumentParser(description="Client Program to send image to Server")
parser.add_argument('-ip','--ipadd', type=str, metavar="", required=True, help="IP Address of the Server")
args = parser.parse_args()
sender = imagezmq.ImageSender(connect_to=args.ip)

rpiName = socket.gethostname()
vs = VideoStream(usePiCamera=True, resolution=(640, 480)).start()
time.sleep(3)
logging.basicConfig(filename="PMS_Patient1.log", filemode='a', datefmt='%H:%M:%S', level=logging.DEBUG)

while True:
    frame = vs.read()
    sender.send_image(rpiName, frame)
    time.sleep(1)
    print(f"Image sent at {datetime.now()}")
    logging.info(f"Image sent at {datetime.now()} to {args.ip }")
