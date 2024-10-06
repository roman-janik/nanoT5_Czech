# Author: Roman Janík
# Script for creating a Hugging Face fast tokenizer.
#

import hydra
import os

from datasets import load_dataset
from utils import get_tokenizer
from hydra.utils import get_original_cwd



def get_training_corpus():
    c4_czech_dataset = load_dataset("allenai/c4", "cs", trust_remote_code=True, streaming=True)
    dataset = c4_czech_dataset["train"]
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield samples["text"]


@hydra.main(config_path="configs", config_name="default", version_base='1.1')
def main(args):
    old_tokenizer = get_tokenizer(args)
    training_corpus = get_training_corpus()

    tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, args.tokenizer.vocab_size)
    old_tokenizer.save_pretrained(os.path.join(get_original_cwd(), args.tokenizer.save_path))


if __name__ == '__main__':
    main()
