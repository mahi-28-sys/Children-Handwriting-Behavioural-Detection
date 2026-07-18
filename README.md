# Children Handwriting Behavioural Detection

A Flask-based Machine Learning web application that analyzes children's handwriting images to predict behavioural traits using handwriting feature extraction and machine learning techniques.

## Features

- Upload handwriting images
- Image preprocessing
- Handwriting feature extraction
- Behaviour prediction using Machine Learning
- Interactive web interface

## Technologies Used

- Python
- Flask
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- HTML
- CSS

## Project Pipeline

1. Preprocess handwriting images.
2. Extract handwriting features (`feature_extractor.py`).
3. Perform clustering (`clustering.py`).
4. Train the machine learning model (`train_model.py`).
5. Run the Flask web application (`app.py`) for behaviour prediction.

## Project Structure

```
Children-Handwriting-Behavioural-Detection
│
├── src/
│   ├── app.py
│   ├── feature_extractor.py
│   ├── clustering.py
│   ├── train_model.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── static/
│   └── templates/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/mahi-28-sys/Children-Handwriting-Behavioural-Detection.git
```

Navigate to the project directory:

```bash
cd Children-Handwriting-Behavioural-Detection
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python src/app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```
