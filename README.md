To jointly address the Named Entity Recognition (NER) and Part-of-Speech (POS) tagging tasks, we employ BERT as the backbone architecture and attach two separate classification heads to its output representations.

Since the model relies on the BERT tokenizer, words are often split into multiple subword tokens. Consequently, proper alignment between token embeddings and their corresponding labels becomes essential.

To handle this issue, we assign the original label only to the first subword token of each word, while the remaining subword tokens are assigned a label value of -1. However, special attention must be paid to the -1 label, as including it in the optimization process can negatively affect convergence and prevent the model from reaching the desired performance. Therefore, these labels are ignored during loss computation using the ignore_index mechanism in the loss function.

In addition, during inference, the predicted labels must be converted back to the original word-level format. To address this alignment problem, we implemented the functions align_tags(), align_logits(), and align_predictions_to_tokens().
