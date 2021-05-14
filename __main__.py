import sys

from models import FuzzyRec


def main():

    embeddings_dir_path = '...path to meta-path-based embeddings...'
    train_rating_file_path = '...path to rating data file for training...'
    test_rating_file_path = '...path to rating data file for testing...'

    user_emb_dim = 300
    user_metapaths = ['...list of meta-path for users...']

    item_emb_dim = 300
    item_metapaths = ['...list of meta-path for items...']

    max_iterations = 60

    alpha_hyper_param = .01
    beta_hyper_param = 1
    lambda_constant = 1.0

    model = FuzzyRec(
        train_rating_file_path,
        test_rating_file_path,
        embeddings_dir_path,
        user_emb_dim,
        user_metapaths,
        item_emb_dim,
        item_metapaths,
        max_iterations,
        alpha_hyper_param,
        beta_hyper_param,
        lambda_constant
    )

    model.run()


if __name__ == "__main__":
    sys.exit(main())