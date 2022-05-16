from setuptools import find_packages, setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="ml_project",
    packages=find_packages(),
    version="0.1.0",
    description="homework 1 for MADE course on ML in production 2021",
    author="Roman Novokshanov",
    entry_points={
        "console_scripts": [
            "ml_project_train = train_pipeline:train_pipeline_command",
            "ml_project_predict = predict_pipeline:predict_pipeline_command",
        ]
    },
    install_requires=required,
    license="MIT",
)

# syntax for entry point
# <name> = [<package>.[<subpackage>.]]<module>[:<object>.<object>]
