# Sentiment-Analysis-of-Movie-Reviews

Overview
This project focuses on the pivotal area of Natural Language Processing (NLP), specifically sentiment analysis, to understand human emotions and opinions expressed in movie reviews. Utilizing the Large Movie Review Dataset, the goal is to accurately classify sentiments as positive or negative. This work is part of the sentiment analysis competition designed to explore and advance domain-specific sentiment analysis techniques.

Goal
The main aim of this competition project is to refine sentiment analysis algorithms for movie reviews, accurately classifying sentiments to capture the nuances of human emotions in text. We employ a Long Short-Term Memory (LSTM) network to achieve high precision and recall in sentiment classification.

Dataset
The dataset is based on the Large Movie Review Dataset by Andrew L. Maas et al. (2011), comprising extensive movie reviews for sentiment analysis. The dataset is split into training and test sets, with the objective of developing a model that can classify review sentiments as either positive (1) or negative (0).

Model
The project utilizes a Keras Sequential model incorporating an LSTM layer, specifically tailored for sequence prediction problems. This model structure is chosen for its effectiveness in capturing long-term dependencies in text data, crucial for understanding the context and sentiment of reviews.

Evaluation
Model performance is evaluated using the F1-score, balancing precision and recall to provide a comprehensive measure of model accuracy. The evaluation metrics are crucial for optimizing the model to perform well in real-world sentiment analysis tasks.

Installation and Usage
Ensure you have Python and the necessary packages installed. You can install all required dependencies using:


pip install -r requirements.txt
To train the model and predict sentiment analysis, run:


python sentiment_analysis_model.py
File Descriptions
sentiment_analysis_model.py: The main script defining the LSTM model, data preprocessing, training, and evaluation.
train.csv & test.csv: Dataset files for training and testing the model. (Note: Due to dataset size or privacy, these files might be linked instead of uploaded directly.)
requirements.txt: Lists all dependencies required to run the project.
Contributing
Contributions to improve the model or extend the project are welcome. Please feel free to fork the repository and submit pull requests.

License
This project is open-source and available under the MIT License.

Acknowledgements
Large Movie Review Dataset by Andrew L. Maas et al., ACL 2011.
Keras and TensorFlow for providing the deep learning framework.
Citation
If you use this project or the dataset in your research, please cite the following:

bibtex
@inproceedings{maas-EtAl:2011:ACL-HLT2011,
  author = {Maas, Andrew L. and Daly, Raymond E. and Pham, Peter T. and Huang, Dan and Ng, Andrew Y. and Potts, Christopher},
  title = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  year = {2011},
  pages = {142--150},
  address = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics}
}
