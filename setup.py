from setuptools import setup, find_packages

setup(
    name='dispatcher',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "pydantic",
        "requests",
    ],
    entry_points={
        "console_scripts": [
            "dispatcher-server=dispatcher.server:main",
        ],
    },
)
