from setuptools import setup, find_packages

setup(
    name="dispatcher",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn>=0.34,<0.35",
        "pydantic",
        "requests",
    ],
    extras_require={
        "dev": [
            "pytest",
            "responses",
            "pytest-cov",
        ],
    },
    entry_points={
        "console_scripts": [
            # generic task-runner CLI
            "dispatcher-task-run = dispatcher.taskmanager.cli:main",

            # starts the FastAPI/uvicorn server
            "dispatcher-server = dispatcher.server:main",

        ],
    },
)
