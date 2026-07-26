# Unsloth arrives one of two ways.
# PREFERRED (offline-safe, the route the 32.22 exemplar used): attach a utility-script notebook that
# pip-installed unsloth==2025.9.7 + the flash-attn wheel with --target on the CURRENT Kaggle image
# (fork sorokin/pip-install-unsloth-flash-patch, swap the cp311 flash-attn wheel URL for the cp312
# build matching this image's torch, run it once with internet, attach YOUR fork). Its tree lands on
# sys.path automatically and the import below just works, no pip here at all.
# FALLBACK (interactive with internet only): plain pip install.
try:
    import unsloth  # noqa: F401
    print('unsloth already importable (utility-script route), version', unsloth.__version__)
except ImportError:
    print('unsloth not on path, falling back to pip (needs internet)')
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'unsloth', 'unsloth_zoo'], check=True)
