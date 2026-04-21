sudo apt-get update && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y libglvnd0 libegl1 libegl-mesa0 libgl1 libglx0 libopengl0 libgles2
sudo apt-get -o Dpkg::Options::="--force-unsafe-io" -f install -y
dpkg -l | grep -E "^ii\s+(libegl1|libgl1\s|libgles2\s|libglvnd0|libegl-mesa0)" || true
ls -ld /usr/share/glvnd /usr/share/glvnd/egl_vendor.d && ls -l /usr/share/glvnd/egl_vendor.d
sudo apt-get -o Dpkg::Options::="--force-unsafe-io" install -y libegl1 libgl1 libgles2
ls -ld /etc/glvnd /etc/glvnd/egl_vendor.d 2>&1 || true && ls -l /usr/share/glvnd/egl_vendor.d 2>&1 || true
sudo mkdir -p /etc/glvnd && if [ ! -e /etc/glvnd/egl_vendor.d ]; then sudo ln -s /usr/share/glvnd/egl_vendor.d /etc/glvnd/egl_vendor.d; fi && ls -ld /etc/glvnd/egl_vendor.d && ls -l /etc/glvnd/egl_vendor.d