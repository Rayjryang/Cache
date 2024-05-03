sudo apt-get update
sudo apt install -y python3.8-venv
python3 -m venv bv_venv
. ~/deit_jax/bv_venv/bin/activate

pip install -U pip  # Yes, really needed.
  # NOTE: doesn't work when in requirements.txt -> cyclic dep
pip install "jax[tpu]==0.4.6" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install flax==0.7.0