xhost +
sudo nvidia-docker run -it -v /home/super/TSC:/TSC -v /tmp/.X11-unix:/tmp/.X11-unix:ro --privileged -e DISPLAY=unix$DISPLAY --shm-size=16G 4495 /bin/bash
