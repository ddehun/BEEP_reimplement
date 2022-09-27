from argparse import ArgumentParser


MIMIC_TASKNAME_MAP = {"MP_IN": "mortality"}


def get_parser():
    parser = ArgumentParser()

    """
    path fname
    """
    # Dataset Path
    parser.add_argument("--mimic_fname", type=str, default="./MIMIC_PREPROCESSED/{}_adm_{}.csv", help="args.task and split name")
    parser.add_argument("--retrieved_abstract_fname", type=str, default="./data/abstracts/{}.pubmed.texts_and_dates.pkl")
    # Experimental path
    parser.add_argument("--model_path", type=str, default="logs{}", help="args.seed")
    # Experiments for retriever/reranker
    parser.add_argument("--retriever_exp_path", type=str, default="./{}/retriever/vanilla/", help="args.seed")
    parser.add_argument("--reranker_exp_path", type=str, default="./{}/reranker/vanilla/", help="args.seed")
    # Experiments for predictor
    parser.add_argument("--task", type=str, choices=["MP_IN", "LOS_WEEKS"], default="MP_IN")
    parser.add_argument("--num_predictor_labels", type=int, default=2)
    parser.add_argument("--predictor_exp_name", type=str, default="vanilla")
    parser.add_argument("--num_doc_for_augment", type=int, default=5)
    parser.add_argument("--augment_strategy", type=str, choices=["avg", "wavg", "svote", "wvote"])
    parser.add_argument(
        "--predictor_exp_path", type=str, default="./{}/predictor/task-{}.{}/", help="args.seed,args.task and args.predictor_exp_name"
    )

    """
    Pickle files
    """
    # Bi-encoder로 인코딩한 abstracts 저장
    parser.add_argument("--encoded_abstract_fname", type=str, default="./data/abstracts/{}-retrieved/encoded_pubmed.pck", help="args.task")
    # Bi-encoder로 점수매긴 거 저장
    parser.add_argument(
        "--biencoder_retrieved_abstract_pck_path",
        type=str,
        default="./data/abstracts/{}-retrieved/pubmed.abstract.biencoder.{}.pck",
        help="args.task and mimic example id",
    )
    # Reranker로 점수 매긴거 저장
    parser.add_argument(
        "--reranker_abstract_score_pck_path",
        type=str,
        default="./data/abstracts/{}-reranked/pubmed.abstract.reranked.{}.pck",
        help="args.task and mimic example id",
    )

    # 학습 데이터 tokenize한거 저장
    parser.add_argument("--retriever_ids_pck_path", type=str, default="./pickled/trec.retriever.{}.ids.pck")
    parser.add_argument("--reranker_ids_pck_path", type=str, default="./pickled/trec.reranker.{}.ids.pck")
    parser.add_argument(
        "--predictor_ids_pck_path",
        type=str,
        default="./pickled/task-{}.trec.predictor.input-{}.{}.ids.pck",
        help="args.task, args.num_doc_for_augment and split name",
    )

    # Method-specific arguments
    # Arguments for retrieval-augmentation
    parser.add_argument("--num_first_retrieval", type=int, default=1000)

    # pre-train LM namecard
    parser.add_argument("--biencoder_lm_ckpt", type=str, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    parser.add_argument("--reranker_lm_ckpt", type=str, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    parser.add_argument("--predictor_lm_ckpt", type=str, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

    # General arguments for training and testing
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrieval_margin", type=float, default=5.0)
    parser.add_argument("--grad_norm", type=float, default=1.0)

    args = parser.parse_args()
    args.casual_task_name = MIMIC_TASKNAME_MAP[args.task]
    args.model_path = args.model_path.format(args.seed)  # e.g., log42
    args.retriever_exp_path = args.retriever_exp_path.format(args.model_path)
    args.reranker_exp_path = args.reranker_exp_path.format(args.model_path)
    args.predictor_exp_path = args.predictor_exp_path.format(args.model_path, args.task, args.predictor_exp_name)
    args.retrieved_abstract_fname = args.retrieved_abstract_fname.format(args.casual_task_name)

    return args
