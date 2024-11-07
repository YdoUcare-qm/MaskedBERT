# BERT for Masked Language Modeling

A BERT model that was **trained** using the **Yelp Review Full** dataset for Masked Language Modeling (MLM). The tokenizer used is the **pretrained `bert-base-uncased`** model. Check how well the model fills in the masked tokens.

## Overview

- **Dataset:** [Yelp Dataset](https://huggingface.co/datasets/Yelp/yelp_review_full)
- **Tokenizer:** Pretrained `bert-base-uncased` BertTokenizer
- **Model:** `BertForMaskedLM`
- **Training:** Trained for 1 epoch using `Yelp Review Full dataset`.

## Directory Structure

- `model/`: Contains the fine-tuned BERT model.
- `tokenizer/`: Contains the tokenizer used for pre-processing the data.

## Usage

1. Clone the repo:
   ```bash
   git clone https://github.com/YdoUcare-qm/MaskedBERT.git
  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Load the model and tokenizer for inference:
   ```python
   import torch
   from transformers import BertTokenizer, BertForMaskedLM

   # Load tokenizer
   tokenizer = BertTokenizer.from_pretrained('tokenizer')

   # Load model
   model = BertForMaskedLM.from_pretrained('model')

   # Example usage
   input_text = "This restaurant is [MASK] good!"
   inputs = tokenizer(input_text, return_tensors='pt')

   # Get logits
   with torch.no_grad():
       outputs = model(**inputs)
       logits = outputs.logits

   # Get the index of [MASK] token
   mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]

   # Extract logits for the [MASK] position and get top 5 predictions
   mask_token_logits = logits[0, mask_token_index, :]
   top_k = 5
   top_k_tokens = torch.topk(mask_token_logits, top_k, dim=1).indices[0].tolist()

   # Convert token IDs back to words
   predicted_tokens = [tokenizer.decode([token]) for token in top_k_tokens]
   print(f"Top {top_k} predictions for the masked word: {predicted_tokens}")
## Refer the Notebooks Section for understanding Training workflows and trying out the model on colab
