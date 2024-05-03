sudo apt-get update
cd ~
sudo apt install -y python3.9-venv
python3.9 -m venv bv_venv
. ~/bv_venv/bin/activate

pip install -U pip  # Yes, really needed.
  # NOTE: doesn't work when in requirements.txt -> cyclic dep
pip install "jax[tpu]==0.4.25" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install flax==0.8.3