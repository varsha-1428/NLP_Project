
# BERT-Based Word Embeddings for Software Story Point Estimation

**Course:** Natural Language Processing\
**Problem Statement 2:** Investigate the effectiveness of BERT-based
word embeddings for predicting story points of later issues from the
same project using chronological splitting.

## 1. Project Overview

This project treats software story-point estimation as a regression
task. The input is the title and description of a software issue, and
the target is its recorded story-point value.

The experiment uses **BERT contextual token-level embeddings** with a
**three-layer 1D CNN** as the prediction model. It follows a
within-project setting using the LSST Data Management project (`DM`)
from the TAWOS dataset. Issues are ordered by creation date so that the
model is evaluated on later issues.

## 2. Dataset

-   **Dataset:** TAWOS (The Agile Work Order System)
-   **Selected project:** LSST Data Management (`DM`)
-   **Dataset source:** [TAWOS dataset record
    (DOI)](https://doi.org/10.5522/04/21308124)
-   **Prepared records used:** 14,782 issues
-   **Input text:** issue title combined with description
-   **Target:** `Story_Point`
-   **Creation-date range:** 4 March 2014 -- 22 October 2020

The prepared CSV contains the columns:

`Issue_Key`, `Title`, `Description_Text`, `Creation_Date`,
`Estimation_Date`, `Story_Point`

The notebook expects this prepared file to be named
**`NLP_Project_dataset.csv`**. The dataset file is not embedded in the
notebook; obtain and prepare it separately, and upload it when prompted.

### Preprocessing performed

1.  Loaded the prepared DM project CSV.
2.  Replaced missing titles/descriptions with empty strings.
3.  Combined title and description into a single `Text` field.
4.  Converted creation and estimation dates to datetime and story points
    to numeric values.
5.  Removed rows with invalid creation dates, invalid story-point
    labels, or empty combined text.
6.  Sorted the records by creation date.
7.  Removed boundary quote characters from the title and description
    text before constructing the final text field.

The prepared data used in the recorded run contains 14,782 rows. Of
these, 12,992 have a description and 1,790 have a missing description;
the title is retained for those issues.

## 3. Methodology

### 3.1 Chronological Development/Test Split

The records are sorted by `Creation_Date` and divided chronologically:

  Partition         Number of issues      Share
  --------------- ------------------ ----------
  Development                 11,825        80%
  Held-out test                2,957        20%
  **Total**               **14,782**   **100%**

The development partition covers 4 March 2014 to 7 November 2019. The
held-out test partition covers 7 November 2019 to 22 October 2020.

### 3.2 Five-Fold Temporal Validation

The development data is evaluated using `TimeSeriesSplit` with five
splits. Each validation block occurs after its corresponding training
block, and the training window expands over successive folds.

  Fold     Training issues   Validation issues
  ------ ----------------- -------------------
  1                  1,975               1,970
  2                  3,945               1,970
  3                  5,915               1,970
  4                  7,885               1,970
  5                  9,855               1,970

The final held-out test set is not used for model selection.

### 3.3 BERT Embedding Extraction

-   Pretrained model: `bert-base-uncased`
-   Tokenizer: BERT WordPiece tokenizer
-   Maximum sequence length: 128 tokens
-   Hidden dimension: 768
-   Number of Transformer layers: 12
-   BERT parameters: frozen; BERT is used as a feature extractor
-   Embedding representation per issue: `128 × 768`
-   Embeddings are stored as `float16` arrays to reduce storage
    requirements.

The attention mask is supplied during BERT embedding extraction.

### 3.4 CNN Regression Model

The extracted token embeddings are passed to a one-dimensional CNN:

  Layer      Configuration
  ---------- ----------------------------------------------
  Conv1D 1   768 → 256 channels, kernel size 3, padding 1
  ReLU       Activation
  Conv1D 2   256 → 128 channels, kernel size 3, padding 1
  ReLU       Activation
  Conv1D 3   128 → 64 channels, kernel size 3, padding 1
  ReLU       Activation
  Pooling    Adaptive max pooling to sequence length 1
  Output     Linear layer, 64 → 1

### 3.5 Training Configuration

  Setting                   Value
  ------------------------- --------------------
  Optimizer                 AdamW
  Learning rate             0.0005
  Weight decay              0.0001
  Loss                      L1 loss
  Batch size                32
  Maximum epochs per fold   25
  Early-stopping patience   5
  Fold random seed          `42 + fold number`
  Final training seed       42
  Final training epochs     6

For each fold, target values are centered using the training-fold median
and scaled using the training-fold interquartile range. The final model
is trained on all development data for the median of the best epoch
counts selected during cross-validation.

## 4. Baselines

Two baselines are included for comparison:

1.  **Development median baseline:** predicts the median story point
    from the relevant training data for every issue. The full
    development-set median used for final testing is 2.0.
2.  **TF-IDF + Ridge regression:** uses unigram and bigram TF-IDF
    features (`min_df=2`, maximum 150,000 features, sublinear term
    frequency) and Ridge regression on `log1p` story-point targets. The
    regularization parameter is selected using development temporal
    validation; the selected value was `alpha=1.0`.

## 5. Evaluation Metrics

The notebook reports:

-   **Mean Absolute Error (MAE):** average absolute difference between
    actual and predicted story points.
-   **Median Absolute Error (MdAE):** median of the absolute prediction
    errors.

Lower values indicate better performance.

## 6. Results

### 6.1 BERT-CNN Five-Fold Temporal Validation

  Fold                MAE         MdAE   Best epoch
  ---------- ------------ ------------ ------------
  1                4.6669       1.7122            8
  2                4.3481       1.3746            8
  3                4.4094       1.5082            5
  4                4.2324       1.4023            3
  5                3.6958       1.5348            6
  **Mean**     **4.2705**   **1.5064**          ---

### 6.2 Median Baseline Five-Fold Temporal Validation

  Fold                MAE         MdAE
  ---------- ------------ ------------
  1                5.4015       1.9000
  2                5.1980       2.0000
  3                5.4619       1.5000
  4                5.0205       1.0000
  5                4.3602       1.0000
  **Mean**     **5.0884**   **1.4800**

### 6.3 Final Held-Out Test Results

  Model                              Test MAE    Test MdAE
  ------------------------------ ------------ ------------
  **BERT-CNN**                     **3.6281**       1.0887
  TF-IDF + Ridge (`alpha=1.0`)         4.0977       1.5014
  Development median baseline          4.5224   **1.0000**

BERT-CNN obtains the lowest test MAE among the evaluated models. The
median baseline obtains the lowest test MdAE. Therefore, the BERT-CNN
model improves average absolute error, while the median baseline remains
slightly better on the median absolute error metric.

### 6.4 Error Analysis

The test-set story-point distribution is right-skewed. Errors increase
for issues with larger actual story-point values. The notebook groups
test issues into story-point ranges and compares BERT-CNN and
median-baseline MAE:

  -----------------------------------------------------------------------
  Actual           Number of issues       BERT-CNN MAE    Median baseline
  story-point                                                         MAE
  range                                                
  -------------- ------------------ ------------------ ------------------
  0--2                        1,801             1.0826             0.9021

  3--5                          569             2.3376             1.7768

  6--10                         350             5.3127             5.7237

  11--20                         92            10.2420            13.2452

  Above 20                      145            32.0469            51.8277
  -----------------------------------------------------------------------

This analysis shows that larger story-point issues are more difficult to
estimate. BERT-CNN has lower MAE than the median baseline in the higher
story-point ranges shown, while the median baseline performs better in
the lower ranges.

## 7. Limitations

-   The experiment uses one software project; results may not generalize
    to other projects.
-   The story-point labels are strongly right-skewed, with relatively
    few high-effort issues.
-   BERT inputs are truncated or padded to 128 tokens.
-   BERT is frozen and is not fine-tuned on the project-specific data.
-   The CNN uses the fixed-length token embeddings; the attention mask
    is used by BERT during embedding extraction but is not passed
    through the CNN pooling stage.

## 8. Repository Contents

  -----------------------------------------------------------------------
  File / Folder                       Description
  ----------------------------------- -----------------------------------
  `NLP_Project_Test1.ipynb`           Main notebook containing
                                      preprocessing, chronological
                                      splitting, BERT embedding
                                      extraction, CNN training,
                                      baselines, evaluation, and plots

  `README.md`                         Project description and
                                      reproduction instructions

  `NLP_Project_dataset.csv`           Prepared input dataset expected by
                                      the notebook (add separately if
                                      permitted)
  -----------------------------------------------------------------------

The notebook also generates intermediate embedding arrays, labels, fold
CSVs, model weights, prediction CSVs, result CSVs, and plot images
during execution. These are generated artifacts and need not be
committed unless required.

## 9. How to Run

The notebook was run in **Google Colab with a Tesla T4 GPU**. A GPU is
recommended because BERT embedding extraction is memory- and
compute-intensive.

1.  Open `NLP_Project_Test1.ipynb` in Google Colab.
2.  Run the installation and import cells.
3.  When the upload prompt appears, upload the prepared dataset CSV
    named `NLP_Project_dataset.csv`.
4.  Run the notebook cells in order.
5.  Allow the notebook to download `bert-base-uncased` from Hugging Face
    during the embedding extraction stage.
6.  Continue through temporal validation, final model training, baseline
    evaluation, and plotting.

The notebook installs `transformers`, `scikit-learn`, `pandas`, `numpy`,
`matplotlib`, and `seaborn`. PyTorch is used for BERT and the CNN; it is
available in the Colab runtime used for the recorded experiment.

**Note:** BERT embedding extraction creates large files (approximately
2.17 GB for the development embeddings in the recorded run). Ensure
sufficient Google Drive/Colab storage and runtime memory. If a Colab
session disconnects, rerun the notebook from the relevant setup stage
and regenerate missing intermediate files.

