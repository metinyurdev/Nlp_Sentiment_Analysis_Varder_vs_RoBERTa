# 🥗 Sentiment Analysis on Amazon Fine Food Reviews  

![Sentiment Analysis](https://img.shields.io/badge/Sentiment%20Analysis-NLP-blueviolet?style=flat-square)  
![Python](https://img.shields.io/badge/Made%20With-Python-blue?style=flat-square&logo=python)  
![Jupyter Notebook](https://img.shields.io/badge/Platform-Jupyter%20Notebook-orange?style=flat-square&logo=jupyter)  
![Transformers](https://img.shields.io/badge/Model-RoBERTa-green?style=flat-square)  
![Lexicon-based](https://img.shields.io/badge/Model-VADER-yellow?style=flat-square)  

## 📌 Project Overview  

In this project, we perform **sentiment analysis** on the [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) dataset, comparing **VADER** (a rule-based model) with **RoBERTa** (a transformer-based deep learning model).  

The goal is to evaluate how these models perform in classifying sentiments, analyze their strengths and weaknesses, and provide insights through **various visualizations**.  

## 🗃 Dataset Information  

This dataset contains over **500,000** food reviews from Amazon, spanning **1999 to 2012**. It includes user information, product details, ratings, and plain text reviews.  

**Dataset Contents:**  
- 📄 `Reviews.csv` → Contains the full dataset  
- 🗃 `database.sqlite` → Contains the `Reviews` table  

🔗 **Dataset Link:** [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)  

## 🔬 Models Used  

| Model  | Type | Strengths |
|--------|------|-----------|
| 📝 **VADER** | Lexicon-based | Fast & lightweight, performs well on short texts |
| 🤖 **RoBERTa** | Transformer-based | High accuracy, deep contextual understanding |

## 📊 Results Summary  

| Metric      | VADER  | RoBERTa  |
|------------|--------|----------|
| **Accuracy**   | 0.0865 | 0.8290   |
| **Precision**  | 0.7743 | 0.8317   |
| **Recall**     | 0.0865 | 0.8290   |
| **F1 Score**   | 0.0235 | 0.8289   |

🎯 **Key Findings:**  
- RoBERTa **outperforms VADER** in all metrics, making it a powerful choice for sentiment analysis.  
- VADER has **high precision for positive sentiments**, making it useful in some scenarios.  
- Multiple visualizations (bar plots, heatmaps, line graphs, stacked bar plots) were used for better insights.  

📌 **Takeaways for Practitioners:**  
- Use **RoBERTa** for **high-accuracy, deep learning-based sentiment analysis**.  
- Use **VADER** for **lightweight, lexicon-based sentiment classification**.  
- Always consider **dataset size and resource availability** when choosing a model.  

## 🛠 Technologies & Libraries Used  

📌 **Programming Language:**  
- 🐍 Python  

📌 **Libraries:**  

| Library | Description |
|---------|------------|
| 📊 `pandas` | Data manipulation & analysis |
| 🔢 `numpy` | Numerical computations |
| 📉 `matplotlib` | Data visualization |
| 📊 `seaborn` | Statistical data visualization |
| 📖 `nltk` | Natural Language Processing (NLP) |
| ⏳ `tqdm` | Progress bar for loops |
| 🤗 `transformers` | Pre-trained NLP models |
| 📔 `jupyter` | Interactive Python notebooks |
| 🎛 `ipywidgets` | Interactive widgets for notebooks |
| 🔥 `torch` | Deep learning framework |
| 🖼 `torchvision` | Image processing for PyTorch |
| 🎵 `torchaudio` | Audio processing for PyTorch |

## 🎯 Conclusion  

This project provides a comprehensive **comparison between lexicon-based (VADER) and transformer-based (RoBERTa) sentiment analysis models** on the Amazon Fine Food Reviews dataset.  

🔍 **Key Insights:**  
- **RoBERTa significantly outperforms VADER** in accuracy, recall, and F1-score, making it a strong choice for sentiment classification.  
- **VADER demonstrates high precision for positive sentiments**, proving useful for quick and lightweight applications.  
- **Visualizations** such as bar plots, heatmaps, line graphs, and stacked bar plots helped analyze model performance effectively.  

🚀 **Future Enhancements:**  
✅ Fine-tune RoBERTa on domain-specific data for improved performance.  
✅ Experiment with other transformer-based models like **BERT, DistilBERT, and XLNet**.  
✅ Extend the analysis to different Amazon product categories for broader insights.  
✅ Implement **hyperparameter tuning** for optimizing RoBERTa performance.  

1️⃣ **Clone the Repository:**  
```bash
git clone https://github.com/metinyurdev/nlp_sentiment_analysis__varder_vs_roberta
cd nlp_sentiment_analysis__varder_vs_roberta
```

2️⃣ Install Required Libraries:
```bash
pip install -r requirements.txt
```

3️⃣ Open Jupyter Notebook:
```bash
jupyter notebook
```

4️⃣ Run Each Cell in Sentiment_Analysis.ipynb 📒


💡 **Final Thoughts:**  
This project serves as a useful guide for those exploring **sentiment analysis in NLP**. Depending on the requirements (speed vs. accuracy), users can choose between

## 📌 Author & Contact  

- **Author:** Metin YURDUSEVEN  
- **📧 Contact:** metin.yrdsvn@gmail.com  
- **📝 License:** MIT  

🚀 *Feel free to contribute, fork, or star this project!* 🌟  

