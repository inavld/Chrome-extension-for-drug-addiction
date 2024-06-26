# SoberSurfer: a Chrome extension that uses an AI model and blurs triggering content

This extension uses a Support Vector Machine to classify and blur addiction-related text content that might have negative impact on individuals recovering from drug addiction.

### Dependencies

Python 3.11.9 was used for this repository. Activate the virtual environment, called `thesis_code` to have access to the other dependencies.

### Executing program

To use the extension, firstly run the `app.py` file that starts the Flask server where the SVM is hosted. 
Secondly, open your Chrome extensions page and upload SoberSurfer by clicking `Load unpacked` and turning it on.

### Examples:

![image](https://github.com/inavld/Chrome-extension-for-drug-addiction/assets/130556930/4607bf3f-0cb8-4671-8394-24361b61c34f)

![image](https://github.com/inavld/Chrome-extension-for-drug-addiction/assets/130556930/bdb5edc1-f5f9-4a1b-9677-a6d636f11bb0)

### Reults of the model
![image](https://github.com/inavld/Chrome-extension-for-drug-addiction/assets/130556930/b3d89833-7a5b-4bc8-b29d-6e7876c457ec)
![image](https://github.com/inavld/Chrome-extension-for-drug-addiction/assets/130556930/8ca95795-1953-41b3-8c3f-64b0917b9a4d)
![image](https://github.com/inavld/Chrome-extension-for-drug-addiction/assets/130556930/8e711ccc-0cef-41d5-9cc3-4a271d7113e0)

## Authors

Ina Maria Vlad

## Acknowledgments
* [Kaggle notebook](https://www.kaggle.com/code/rahulvv/lstm-machine-learning-models-89-accuracy/notebook)

* [Reddit dataset](https://www.kaggle.com/datasets/prakharrathi25/reddit-data-huge)
* [Dash drug data](https://zenodo.org/records/4278895)
