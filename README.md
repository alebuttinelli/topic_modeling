# Italian Legal-BERT Topic Modelling Pipeline

This repository contains a complete, three-script Python pipeline for training, visualising, and analysing a **BERTopic** model on a corpus of Italian legal texts.

The pipeline is built to be modular:
1.  **Train:** It uses a specialised legal-BERT model (`dlicari/lsg16k-Italian-Legal-BERT-SC`) to compute embeddings and train the topic model.
2.  **Visualise:** It generates a suite of interactive HTML plots to explore the model's results.
3.  **Analyse:** It provides a script to analyse model uncertainty and find ambiguous documents.

## The Pipeline Explained

This project is not a single script but a **data pipeline**. The scripts are designed to be run in order, with each script consuming the 'artefacts' (saved files) from the previous one.



1.  **`train_bertopic.py`:** This is the main training script. It loads a raw text file, computes embeddings, trains the BERTopic model, and saves all the necessary artefacts (the model, embeddings, probabilities, topics, and metadata) to a `run` directory.
2.  **`visualize_bertopic.py`:** This script loads the artefacts from a training `run` directory. It then generates interactive Plotly visualisations (e.g., `visualize_topics`, `visualize_documents`) and saves them as HTML files.
3.  **`analyze_uncertainty.py`:** This script also loads artefacts from a training `run`. It identifies documents with high uncertainty (e.g., low maximum probability or high entropy) and saves the analysis to a single CSV file for review.

---

## Tech Stack

* **Core:** Python 3.10+
* **AI & NLP:** `PyTorch`, `Hugging Face Transformers`
* **Topic Modelling:** `bertopic`, `umap-learn`, `hdbscan`
* **Data Handling:** `pandas`, `numpy`, `scipy`, `pyarrow`
* **Visualisation:** `plotly`

---

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    ```
    *Activate it:*
    * **macOS/Linux:** `source venv/bin/activate`
    * **Windows:** `venv\Scripts\activate`

3.  **Install the dependencies:**
    This will install all required libraries from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage: Running the Pipeline

Run the scripts from your terminal in the following order.

### Step 1: Train the Model

First, run `train_bertopic.py`. You must provide an input file (`-i`) and a directory (`-o`) where the results will be saved.

```bash
python train_bertopic.py \
    --input ./data/my_legal_texts.csv \
    --output_dir ./runs/run_01_legal_model
    ```

### Step 2: Visualise the Results
Next, run visualize_bertopic.py. Point it to the run directory you just created (-r) and specify an output directory for your plots (-o).

```bash
python visualize_bertopic.py \
    --run_dir ./runs/run_01_legal_model \
    --output_dir ./visuals/run_01_plots
    ```

This creates a new folder ./visuals/run_01_plots containing interactive HTML files like visualize_topics.html and visualize_documents.html.

### Step 3: Analyse Uncertainty
Finally, you can analyse the model's uncertainty. This script also reads from the run directory (-r) and saves a single CSV file (-o).

```bash
python analyze_uncertainty.py \
    --run_dir ./runs/run_01_legal_model \
    --output_file ./analysis/run_01_uncertainty.csv \
    --threshold 0.5
    ```

This creates the file ./analysis/run_01_uncertainty.csv, which lists all documents that the model was "unsure" about, sorted by how uncertain they are. You can change the --threshold (default is 0.5).

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
