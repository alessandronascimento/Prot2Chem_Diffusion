from setuptools import setup, find_packages

setup(
    name="prot2chemdiff",
    version="0.1.0",
    description="A Latent Diffusion model for target-aware molecule generation.",
    author="NascimentoLab",
    packages=find_packages(),
    install_requires=[
        "torch>=1.13.0",
        "pytorch-lightning>=2.0.0",
        "transformers>=4.0.0",
        "rdkit",
        "selfies",
        "huggingface-hub",
        "diffusers",
        "tqdm",
        "pandas"
    ],
)