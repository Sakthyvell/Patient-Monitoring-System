<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css"
    integrity="sha512-KfkfwYDsLkIlwQp6LFnl8zNdLGxu9YAA1QvwINks4PhcElQSvqcyVLLD9aMhXd13uQjoXtEKNosOWaZqXgel0g=="
    crossorigin="anonymous" referrerpolicy="no-referrer" />
<link href="https://fonts.googleapis.com/css2?family=Quicksand:wght@400;700&display=swap" rel="stylesheet">

<div align="center" style="font-family: 'QuickSand';">
    <h2 style="font-family: 48px;">Patient Monitoring System</h2>
    <p style="font-family: 24px;">System for monitoring patient facial emotion using Raspberry Pi Cam and Remote Server</p>
    <p align="center">
        <img src="https://img.shields.io/github/languages/count/sakthyvell/Patient-Monitoring-System" alt="">
        <img src="https://img.shields.io/github/languages/top/sakthyvell/Patient-Monitoring-System" alt="">
        <img src="https://img.shields.io/github/last-commit/sakthyvell/Patient-Monitoring-System" alt="">
        <img src="https://img.shields.io/badge/development-completed-blue" alt="">
    </p>
</div>

<hr>
<br>

### Installation
#### Recommended : Install virtualenv
```bash
$ pip install virtualenv
$ virtualenv venv
```
#### Install dependencies
```bash
$ pip install -r requirements.txt
```

### Running the program
#### Running the subprogram in raspberry pi (to send image captured by cam)
```bash
$ python Server/client.py
```
#### Running thesubprogram in remote server (to receive image sent by rpi)
```bash
$ python Server/server.py
```
#### Running emotion detection model to capture
```bash
$ python face_capture/face_capture.py
```

This program is part of the source code for the study submitted in [2020 IEEE Student Conference on Research and Development (SCOReD)](https://ieeexplore.ieee.org/abstract/document/9250950).
