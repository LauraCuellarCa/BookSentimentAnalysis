from setuptools import setup, find_packages

setup(
    name="bookalytics",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "nltk",
        "scikit-learn",
        "textblob",
        "spacy",
        "torch",
        "transformers",
        "matplotlib",
        "pdfplumber",
        "PyPDF2",
        "fastapi",
        "uvicorn",
        "python-multipart",
        "aiofiles",
    ],
    entry_points={
        "console_scripts": [
            "bookalytics=bookalytics.main:start",
        ],
    },
    author="BookAlytics Team",
    author_email="info@bookalytics.com",
    description="Book sentiment and emotion analysis tool",
    keywords="nlp, books, sentiment, emotion, analysis",
    python_requires=">=3.7",
) 