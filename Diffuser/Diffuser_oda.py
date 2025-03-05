
import argparse
from argparse import ArgumentParser
import os
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.loggers.base import DummyLogger
import pytorch_lightning as pl
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI
import numpy as np

from os_sfda.Diffuser.datasets import CustomDataset
from os_sfda.Diffuser.datasets import GMM_dataset
from os_sfda.Diffuser.clusternet_modules.clusternetasmodel import ClusterNetModel
from os_sfda.Diffuser.utils import check_args, cluster_acc


'''
borrowed from https://github.com/BGU-CS-VIL/DeepDPM  
'''

def parse_minimal_args(parser):
    # Dataset parameters
    parser.add_argument("--dir", default="./pretrained_embeddings/umap_embedded_datasets/MNIST/",
                        help="dataset directory")
    parser.add_argument("--dataset", default="custom")
    # Training parameters
    parser.add_argument(
        "--lr", type=float, default=0.002, help="learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="input batch size for training"#128
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1, help="number of jobs to run in parallel"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device for computation (default: cpu)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run training without Neptune Logger"
    )
    parser.add_argument(
        "--tag", type=str, default="MNIST_UMAPED",
    )
    parser.add_argument(
        "--max_epochs", type=int, default=500,
    )
    parser.add_argument(
        "--limit_train_batches", type=float, default=1., help="used for debugging"
    )
    parser.add_argument(
        "--limit_val_batches", type=float, default=1., help="used for debugging"
    )
    parser.add_argument(
        "--save_checkpoints", type=bool, default=False
    )
    parser.add_argument(
        "--exp_name", type=str, default="default_exp"
    )
    parser.add_argument(
        "--use_labels_for_eval",
        action="store_true",
        help="whether to use labels for evaluation"
    )
    return parser


def run_on_embeddings_hyperparams(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False,allow_abbrev=False)
    parser.add_argument(
        "--init_k", default=1, type=int, help="number of initial clusters"
    )
    parser.add_argument(
        "--clusternet_hidden",
        type=int,
        default=50,
        help="The dimensions of the hidden dim of the clusternet. Defaults to 50.",
    )
    parser.add_argument(
        "--clusternet_hidden_layer_list",
        type=int,
        nargs="+",
        default=[50],
        help="The hidden layers in the clusternet. Defaults to [50, 50].",
    )
    parser.add_argument(
        "--transform_input_data",
        type=str,
        default="normalize",
        choices=["normalize", "min_max", "standard", "standard_normalize", "None", None],
        help="Use normalization for embedded data",
    )
    parser.add_argument(
        "--cluster_loss_weight",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--init_cluster_net_weights",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--when_to_compute_mu",
        type=str,
        choices=["once", "every_epoch", "every_5_epochs"],
        default="every_epoch",
    )
    parser.add_argument(
        "--how_to_compute_mu",
        type=str,
        choices=["kmeans", "soft_assign"],
        default="soft_assign",
    )
    parser.add_argument(
        "--how_to_init_mu",
        type=str,
        choices=["kmeans", "soft_assign", "kmeans_1d"],
        default="kmeans",
    )
    parser.add_argument(
        "--how_to_init_mu_sub",
        type=str,
        choices=["kmeans", "soft_assign", "kmeans_1d"],
        default="kmeans_1d",
    )
    parser.add_argument(
        "--log_emb_every",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--log_emb",
        type=str,
        default="never",
        choices=["every_n_epochs", "only_sampled", "never"]
    )
    parser.add_argument(
        "--train_cluster_net",
        type=int,
        default=300,
        help="Number of epochs to pretrain the cluster net",
    )
    parser.add_argument(
        "--cluster_lr",
        type=float,
        default=0.0008,
    )
    parser.add_argument(
        "--subcluster_lr",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--lr_scheduler", type=str, default="StepLR", choices=["StepLR", "None", "ReduceOnP"]
    )
    parser.add_argument(
        "--start_sub_clustering",
        type=int,
        default=45,#45
    )
    parser.add_argument(
        "--subcluster_loss_weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--start_splitting",
        type=int,
        default=55,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--softmax_norm",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--subcluster_softmax_norm",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--split_prob",
        type=float,
        default=None,
        help="Split with this probability even if split rule is not met.  If set to None then the probability that will be used is min(1,H).",
    )
    parser.add_argument(
        "--merge_prob",
        type=float,
        default=None,
        help="merge with this probability even if merge rule is not met. If set to None then the probability that will be used is min(1,H).",
    )
    parser.add_argument(
        "--init_new_weights",
        type=str,
        default="same",
        choices=["same", "random", "subclusters"],
        help="How to create new weights after split. Same duplicates the old cluster's weights to the two new ones, random generate random weights and subclusters copies the weights from the subclustering net",
    )
    parser.add_argument(
        "--start_merging",
        type=int,
        default=55,
        help="The epoch in which to start consider merge proposals",
    )
    parser.add_argument(
        "--merge_init_weights_sub",
        type=str,
        default="highest_ll",
        help="How to initialize the weights of the subclusters of the merged clusters. Defaults to same",
    )
    parser.add_argument(
        "--split_init_weights_sub",
        type=str,
        default="random",
        choices=["same_w_noise", "same", "random"],
        help="How to initialize the weights of the subclusters of the merged clusters. Defaults to same",
    )
    parser.add_argument(
        "--split_every_n_epochs",
        type=int,
        default=10,
        help="Example: if set to 10, split proposals will be made every 10 epochs",
    )
    parser.add_argument(
        "--split_merge_every_n_epochs",
        type=int,
        default=30,
        help="Example: if set to 10, split proposals will be made every 10 epochs",
    )
    parser.add_argument(
        "--merge_every_n_epochs",
        type=int,
        default=10,
        help="Example: if set to 10, merge proposals will be made every 10 epochs",
    )
    parser.add_argument(
        "--raise_merge_proposals",
        type=str,
        default="brute_force_NN",
        help="how to raise merge proposals",
    )
    parser.add_argument(
        "--cov_const",
        type=float,
        default=0.005,
        help="gmms covs (in the Hastings ratio) will be torch.eye * cov_const",
    )
    parser.add_argument(
        "--freeze_mus_submus_after_splitmerge",
        type=int,
        default=5,
        help="Numbers of epochs to freeze the mus and sub mus following a split or a merge step",
    )
    parser.add_argument(
        "--freeze_mus_after_init",
        type=int,
        default=5,
        help="Numbers of epochs to freeze the mus and sub mus following a new initialization",
    )
    parser.add_argument(
        "--use_priors",
        type=int,
        default=1,
        help="Whether to use priors when computing model's parameters",
    )
    parser.add_argument("--prior", type=str, default="NIW", choices=["NIW", "NIG"])
    parser.add_argument(
        "--pi_prior", type=str, default="uniform", choices=["uniform", None]
    )
    parser.add_argument(
        "--prior_dir_counts",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--prior_kappa",
        type=float,
        default=0.0001,
    )
    parser.add_argument(
        "--NIW_prior_nu",
        type=float,
        default=None,
        help="Need to be at least codes_dim + 1",
    )
    parser.add_argument(
        "--prior_mu_0",
        type=str,
        default="data_mean",
    )
    parser.add_argument(
        "--prior_sigma_choice",
        type=str,
        default="isotropic",
        choices=["iso_005", "iso_001", "iso_0001", "data_std"],
    )
    parser.add_argument(
        "--prior_sigma_scale",
        type=float,
        default=".005",
    )
    parser.add_argument(
        "--prior_sigma_scale_step",
        type=float,
        default=1.,
        help="add to change sigma scale between alternations"
    )
    parser.add_argument(
        "--compute_params_every",
        type=int,
        help="How frequently to compute the clustering params (mus, sub, pis)",
        default=1,
    )
    parser.add_argument(
        "--start_computing_params",
        type=int,
        help="When to start to compute the clustering params (mus, sub, pis)",
        default=25,
    )
    parser.add_argument(
        "--cluster_loss",
        type=str,
        help="What kind og loss to use",
        default="KL_GMM_2",
        choices=["diag_NIG", "isotropic", "KL_GMM_2"],
    )
    parser.add_argument(
        "--subcluster_loss",
        type=str,
        help="What kind og loss to use",
        default="isotropic",
        choices=["diag_NIG", "isotropic", "KL_GMM_2"],
    )

    parser.add_argument(
        "--ignore_subclusters",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--log_metrics_at_train",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--gpus",
        default=None
    )
    parser.add_argument(
        "--evaluate_every_n_epochs",
        type=int,
        default=5,
        help="How often to evaluate the net"
    )
    return parser


def train_cluster_net(all_fea,all_label,semisup_label,init_k,txt_input,train_state): #
    parser2 = argparse.ArgumentParser(description="Only_for_embbedding",allow_abbrev=False)
    parser2 = parse_minimal_args(parser2)
    parser2 = run_on_embeddings_hyperparams(parser2)
    args2 = parser2.parse_args()
    args2.train_cluster_net = args2.max_epochs

    if args2.dataset == "synthetic":
        dataset_obj = GMM_dataset(args2)
    else:
        dataset_obj = CustomDataset(args2)


    if train_state=='do_infer':
        train_loader, val_loader = dataset_obj.get_loaders(all_fea, all_label,train_state)
    elif train_state=='only_optimize' or train_state=='only_optimize_w_semi':
        train_loader, val_loader = dataset_obj.get_loaders(all_fea, semisup_label,train_state)


    tags = ['umap_embbeded_dataset']

    args2.offline = True
    if args2.offline:
        logger = DummyLogger()
    else:
        logger = NeptuneLogger(
            api_key='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmOGIxZjUyZi1iNzk1LTQwMjctYTcxMS0yMjBmMTU4ODIxMjUifQ==',
            project_name='wfl/os_sfda/Diffuser',
            experiment_name=args2.exp_name,
            params=vars(args2),
            tags=tags
        )

    check_args(args2, dataset_obj.data_dim)

    if isinstance(logger, NeptuneLogger):
        if logger.api_key == 'your_API_token':
            print("No Neptune API token defined!")
            print("Please define Neptune API token or run with the --offline argument.")
            print("Running without logging...")
            logger = DummyLogger()

    # Main body
    if args2.seed:
        pl.utilities.seed.seed_everything(args2.seed)
    args2.init_k = init_k

    #print & save txt
    txt_save_dir = txt_input
    log_str = 'init_K: {:.2f}'.format(args2.init_k)
    with open(txt_save_dir, "a") as f:
        f.write(log_str + '\n')
    print(log_str)
    if train_state=='do_infer':
        args2.max_epochs = 500
        train_loader, val_loader = dataset_obj.get_loaders(all_fea, all_label,train_state)
    elif train_state=='only_optimize' or train_state=='only_optimize_w_semi':
        args2.max_epochs = 100
        args2.start_splitting = 300
        args2.start_sub_clustering = 1000
        args2.cluster_lr = 0.0008
        if train_state=='only_optimize_w_semi':
            args2.cluster_lr = 0.0008
        train_loader, val_loader = dataset_obj.get_loaders(all_fea, semisup_label,train_state)
    model = ClusterNetModel(hparams=args2, input_dim=dataset_obj.data_dim, init_k=args2.init_k)
    if args2.save_checkpoints:
        from pytorch_lightning.callbacks import ModelCheckpoint
        checkpoint_callback = ModelCheckpoint(dirpath=f"./saved_models/{args.dataset}/{args.exp_name}")
        # if not os.path.exists(f'./saved_models/{args.dataset}'):
        #     os.makedirs(f'./saved_models/{args.dataset}')
        # os.makedirs(f'./saved_models/{args.dataset}/{args.exp_name}')
    else:
        checkpoint_callback = False
    trainer = pl.Trainer(logger=logger, max_epochs=args2.max_epochs, gpus=args2.gpus, num_sanity_val_steps=0,
                         checkpoint_callback=checkpoint_callback, limit_train_batches=args2.limit_train_batches,
                         limit_val_batches=args2.limit_val_batches)
    trainer.fit(model, train_loader, val_loader)



    print("Finished training!")

    # evaluate last model
    args2.use_labels_for_eval = True
    dataset = dataset_obj.get_train_data(all_fea, all_label,train_state)
    data = dataset.data
    net_pred = model(data).argmax(axis=1).cpu().numpy()
    if args2.use_labels_for_eval:
        # evaluate model using labels
        labels = dataset.targets.numpy()
        acc = np.round(cluster_acc(labels, net_pred), 5)
        nmi = np.round(NMI(net_pred, labels), 5)
        ari = np.round(ARI(net_pred, labels), 5)

        # print 'model.cluster_net.class_fc2.out_features'
        print(f"NMI: {nmi}, ARI: {ari}, acc: {acc}, final K: {len(np.unique(net_pred))}, inf_K: {model.cluster_net.class_fc2.out_features}")
        print(net_pred)
        with open(txt_save_dir, "a") as f:
            f.write(f"NMI: {nmi}, ARI: {ari}, acc: {acc}, final K: {len(np.unique(net_pred))}, inf_K: {model.cluster_net.class_fc2.out_features}\n")
            f.write("********************************\n")

    return net_pred,model.cluster_net.class_fc2.out_features,len(np.unique(net_pred))

