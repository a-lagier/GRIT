import logging
import time
from os.path import join

import numpy as np
import torch
from torch_geometric.graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt, get_ckpt_epoch, get_ckpt_path
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch

from grit.loss.subtoken_prediction_loss import subtoken_cross_entropy
from grit.utils import cfg_to_dict, flatten_dict, make_wandb_name, mlflow_log_cfgdict
import warnings
from collections import defaultdict














































import numpy as np
import torch
from torch_geometric.utils import to_dense_adj
from torch_scatter import scatter
import matplotlib.pyplot as plt


def aggregate_k_hop(edge_index, angles, k, alpha=2.):
    enhanced_angles = angles.clone()

    f = lambda k : np.log(k + 1.)
    # f = lambda k : np.exp(-k/2)
    
    k_paths = edge_index
    row, col = k_paths
    enhanced_angles += scatter(angles[col] * f(1), row, dim=0)

    for k_ in range(2, k + 1):
        edge1_idx, edge2_idx = torch.where(k_paths[1][:, None] == edge_index[0][None, :])

        path_sources = k_paths[0][edge1_idx]
        path_targets = edge_index[1][edge2_idx]
        
        k_paths = torch.stack([path_sources, path_targets], dim=0)

        # Delete duplicates
        k_paths = torch.tensor(np.unique(k_paths.cpu().numpy(), axis=1))
        
        if torch.cuda.is_available():
            k_paths = k_paths.cuda()

        row, col = k_paths
        enhanced_angles += scatter(angles[col] * f(k), row, dim=0)

    return enhanced_angles


def display_grids(batch):
    n = batch.num_nodes
    d_angles = 10
    k = 5

    adj = to_dense_adj(batch.edge_index).view(n,n)

    adj_k = adj.clone()
    for _  in range(k-1):
        adj_k += adj_k @ adj

    std = 0.5
    angles = torch.normal(0., std, size=(n, d_angles))


    def similarity_func(X, sim_func="dot"):
        # X of dimension (number of nodes, dim_angles)
        if sim_func == "dot":
            return X @ X.transpose(0,1)
        elif sim_func == "norm":
            norm = X.pow(2).sum(dim=1)
            norm_1 = norm.view(-1, 1).repeat_interleave(n, dim=-1)
            return norm_1 + norm_1.transpose(0,1) -2 * (X @ X.transpose(0,1))


    angles_similarity = similarity_func(angles, sim_func="norm")

    alpha = 1/10.
    enhanced_angles_10 = aggregate_k_hop(batch.edge_index, angles, k=k, alpha=alpha)
    enhanced_angles_10_similarity = similarity_func(enhanced_angles_10, sim_func="norm")

    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0,0].imshow(adj)
    ax[0,0].set_title("Adjacency matrix")

    ax[1,0].imshow(angles_similarity)
    ax[1,0].set_title("Angles similarity")

    ax[0,1].imshow(adj_k)
    ax[0,1].set_title(f"Adjacency matrix with k={k}")

    ax[1,1].imshow(enhanced_angles_10_similarity)
    ax[1,1].set_title(f"Enhanced angles similarity k={k}")
    plt.show()






































def train_epoch(logger, loader, model, optimizer, scheduler, batch_accumulation, overfit=False):
    model.train()
    optimizer.zero_grad()
    time_start = time.time()
    for iter, batch in enumerate(loader):
        # ipdb.set_trace()
        batch.split = 'train'
        batch.to(torch.device(cfg.device))
        pred, true = model(batch)

        if cfg.dataset.name == 'ogbg-code2':
            loss, pred_score = subtoken_cross_entropy(pred, true)
            _true = true
            _pred = pred_score
        else:
            loss, pred_score = compute_loss(pred, true)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)
        loss.backward()
        # Parameters update after accumulating gradients for given num. batches.
        if overfit or ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(loader)):
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name)
        time_start = time.time()
        if overfit:
            print(batch)
            break


@torch.no_grad()
def eval_epoch(logger, loader, model, split='val'):
    model.eval()
    time_start = time.time()
    for batch in loader:
        batch.split = split
        batch.to(torch.device(cfg.device))
        if cfg.gnn.head == 'inductive_edge':
            pred, true, extra_stats = model(batch)
        else:
            pred, true = model(batch)
            extra_stats = {}
        if cfg.dataset.name == 'ogbg-code2':
            loss, pred_score = subtoken_cross_entropy(pred, true)
            _true = true
            _pred = pred_score
        else:
            loss, pred_score = compute_loss(pred, true)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            lr=0, time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name,
                            **extra_stats)
        time_start = time.time()



@register_train('custom')
def custom_train(loggers, loaders, model, optimizer, scheduler, overfit=False):
    """
    Customized training pipeline.
    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler
        overfit: True if the training aim to overfit the model on one instance
    """
    # loggers[0].tb_writer.add_graph(model.model)


    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler,
                                cfg.train.epoch_resume)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch %s', start_epoch)

    if cfg.wandb.use:
        try:
            import wandb
        except:
            raise ImportError('WandB is not installed.')
        if cfg.wandb.name == '':
            wandb_name = make_wandb_name(cfg)
        else:
            wandb_name = cfg.wandb.name
        run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project,
                         name=wandb_name)
        run.config.update(cfg_to_dict(cfg))

    if cfg.mlflow.use:
        try:
            import mlflow
        except:
            raise ImportError('MLflow is not installed.')
        if cfg.mlflow.name == '':
            MLFLOW_NAME = make_wandb_name(cfg)
        else:
            MLFLOW_NAME = cfg.mlflow.name

        if cfg.name_tag != '':
            MLFLOW_NAME = MLFLOW_NAME + '-' + cfg.name_tag

        experiment = mlflow.set_experiment(cfg.mlflow.project)
        mlflow.start_run(run_name=MLFLOW_NAME)
        mlflow.pytorch.log_model(model, "model")
        mlflow_log_cfgdict(cfg_to_dict(cfg), mlflow_func=mlflow)
        if cfg.get('cfg_file', None) is not None: mlflow.log_artifact(cfg.cfg_file) # log the whole config-file



    num_splits = len(loggers)
    split_names = ['val', 'test']
    full_epoch_times = []
    perf = [[] for _ in range(num_splits)]

    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        # with torch.autograd.detect_anomaly():
        start_time = time.perf_counter()
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler,
                    cfg.optim.batch_accumulation, overfit=False)

        # if cur_epoch % 10 == 0:
        #     for (_, b) in enumerate(loaders[0]):
        #         display_grids(b)
        #         break

        perf[0].append(loggers[0].write_epoch(cur_epoch))

        # debug = True
        # if debug:
        #     tb_writer = loggers[0].tb_writer
        #     for k,v in model.named_buffers():
        #         if "running" in k:
        #             tb_writer.add_text(k, str(v), global_step=cur_epoch)

        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                try:
                    eval_epoch(loggers[i], loaders[i], model,
                               split=split_names[i - 1])
                    perf[i].append(loggers[i].write_epoch(cur_epoch))
                except Exception as e:
                    print(e)
                    perf[i].append(perf[i][-1])
                    # reset the running_mean and running_var for batchnorm is extreme results occurs
                    warnings.warn(f"NaN occurs on the {split_names[i - 1]}_loss, reset the running-mean and running-var for BN")
                    for k, v in model.named_buffers():
                        if "running_mean" in k:
                            v.data = torch.zeros_like(v.data)
                        if "running_var" in k:
                            v.data = torch.ones_like(v.data) * 1e3
        else:
            for i in range(1, num_splits):
                perf[i].append(perf[i][-1])

        val_perf = perf[1]
        if cfg.optim.scheduler == 'reduce_on_plateau' or "timm" in cfg.optim.scheduler:
            scheduler.step(val_perf[-1]['loss'])
        else:
            scheduler.step()
        full_epoch_times.append(time.perf_counter() - start_time)
        # Checkpoint with regular frequency (if enabled).
        if cfg.train.enable_ckpt and not cfg.train.ckpt_best \
                and is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)

        if cfg.wandb.use:
            run.log(flatten_dict(perf), step=cur_epoch)

        if cfg.mlflow.use:
            mlflow.log_metrics(flatten_dict(perf), step=cur_epoch)

        # Log current best stats on eval epoch.
        if is_eval_epoch(cur_epoch):
            best_epoch = np.array([vp['loss'] for vp in val_perf]).argmin()
            best_epoch_loss = best_epoch

            best_train = best_val = best_test = ""
            if cfg.metric_best != 'auto':
                # Select again based on val perf of `cfg.metric_best`.
                m = cfg.metric_best
                best_epoch = getattr(np.array([vp[m] for vp in val_perf]),
                                     cfg.metric_agg)()

                # if cfg.get("mv_metric_best", False):
                #     mv_len = cfg.get("mv_len", 10)
                #     mv_metric = np.array([vp[m] for vp in val_perf])
                #     if len(mv_metric) > mv_len:
                #         mv_metric = np.array([np.mean(mv_metric[max(i-mv_len, 0):i+1]) for i in range(len(mv_metric))])
                #     best_epoch = getattr(mv_metric, cfg.metric_agg)()


                if cfg.best_by_loss:
                    best_epoch = best_epoch_loss

                if m in perf[0][best_epoch]:
                    best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
                else:
                    best_train = f"train_{m}: {0:.4f}"
                best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
                best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

                if cfg.wandb.use or cfg.mlflow.use:
                    bstats = {"best/epoch": best_epoch}
                    for i, s in enumerate(['train', 'val', 'test']):
                        bstats[f"best/{s}_loss"] = perf[i][best_epoch]['loss']
                        if m in perf[i][best_epoch]:
                            bstats[f"best/{s}_{m}"] = perf[i][best_epoch][m]
                            if cfg.wandb.use:
                                run.summary[f"best_{s}_perf"] = \
                                    perf[i][best_epoch][m]
                            if cfg.mlflow.use:
                                mlflow.log_metric(f"best_{s}_perf", perf[i][best_epoch][m])

                        for x in ['hits@1', 'hits@3', 'hits@10', 'hits@20', 'mrr']:
                            if x in perf[i][best_epoch]:
                                bstats[f"best/{s}_{x}"] = perf[i][best_epoch][x]
                    if cfg.wandb.use:
                        run.log(bstats, step=cur_epoch)
                        run.summary["full_epoch_time_avg"] = np.mean(full_epoch_times)
                        run.summary["full_epoch_time_sum"] = np.sum(full_epoch_times)

                    if cfg.mlflow.use:
                        mlflow.log_metrics(bstats, step=cur_epoch)
                        mlflow.log_metric("full_epoch_time_avg", np.mean(full_epoch_times))
                        mlflow.log_metric("full_epoch_time_sum", np.sum(full_epoch_times))


            # Checkpoint the best epoch params (if enabled).
            if cfg.train.enable_ckpt and cfg.train.ckpt_best and \
                    best_epoch == cur_epoch:
                if cur_epoch < cfg.optim.num_warmup_epochs:
                    pass
                else:
                    save_ckpt(model, optimizer, scheduler, cur_epoch)
                if cfg.train.ckpt_clean:  # Delete old ckpt each time.
                    clean_ckpt()
            logging.info(
                f"> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s "
                f"(avg {np.mean(full_epoch_times):.1f}s) | "
                f"Best so far: epoch {best_epoch}\t"
                f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
                f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t" 
                f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}\n"
                f"-----------------------------------------------------------"
            )

        if cfg.optim.get('early_stop_by_lr', False):
            lr = scheduler.get_last_lr()[0]
            if lr <= cfg.optim.min_lr:
                break
        elif cfg.optim.get("early_stop_by_perf", False):
            if (cur_epoch - best_epoch) > cfg.optim.patience:
                break

    if cfg.train.enable_ckpt and cfg.train.ckpt_best and cfg.mlflow.use:
        mlflow.log_artifact(get_ckpt_path(get_ckpt_epoch(best_epoch)))

    logging.info(f"Avg time per epoch: {np.mean(full_epoch_times):.2f}s")
    logging.info(f"Total train loop time: {np.sum(full_epoch_times) / 3600:.2f}h")


    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()
    # close wandb
    if cfg.wandb.use:
        run.finish()
        run = None

    if cfg.mlflow.use:
        mlflow.end_run()
        run = None

    logging.info('Task done, results saved in %s', cfg.run_dir)


@register_train('inference-only')
def inference_only(loggers, loaders, model, optimizer=None, scheduler=None):
    """
    Customized pipeline to run inference only.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    num_splits = len(loggers)
    split_names = ['train', 'val', 'test']
    perf = [[] for _ in range(num_splits)]
    cur_epoch = 0
    start_time = time.perf_counter()

    for i in range(0, num_splits):
        eval_epoch(loggers[i], loaders[i], model,
                   split=split_names[i])
        perf[i].append(loggers[i].write_epoch(cur_epoch))

    best_epoch = 0
    best_train = best_val = best_test = ""
    if cfg.metric_best != 'auto':
        # Select again based on val perf of `cfg.metric_best`.
        m = cfg.metric_best
        if m in perf[0][best_epoch]:
            best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
        else:
            # Note: For some datasets it is too expensive to compute
            # the main metric on the training set.
            best_train = f"train_{m}: {0:.4f}"
        best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
        best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

    logging.info(
        f"> Inference | "
        f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
        f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
        f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
    )
    logging.info(f'Done! took: {time.perf_counter() - start_time:.2f}s')
    for logger in loggers:
        logger.close()


@register_train('PCQM4Mv2-inference')
def ogblsc_inference(loggers, loaders, model, optimizer=None, scheduler=None):
    """
    Customized pipeline to run inference on OGB-LSC PCQM4Mv2.

    Args:
        loggers: Unused, exists just for API compatibility
        loaders: List of loaders
        model: GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    from ogb.lsc import PCQM4Mv2Evaluator
    evaluator = PCQM4Mv2Evaluator()

    num_splits = 3
    split_names = ['valid', 'test-dev', 'test-challenge']
    assert len(loaders) == num_splits, "Expecting 3 particular splits."

    # Check PCQM4Mv2 prediction targets.
    logging.info(f"0 ({split_names[0]}): {len(loaders[0].dataset)}")
    assert(all([not torch.isnan(d.y)[0] for d in loaders[0].dataset]))
    logging.info(f"1 ({split_names[1]}): {len(loaders[1].dataset)}")
    assert(all([torch.isnan(d.y)[0] for d in loaders[1].dataset]))
    logging.info(f"2 ({split_names[2]}): {len(loaders[2].dataset)}")
    assert(all([torch.isnan(d.y)[0] for d in loaders[2].dataset]))

    model.eval()
    for i in range(num_splits):
        all_true = []
        all_pred = []
        for batch in loaders[i]:
            batch.to(torch.device(cfg.device))
            pred, true = model(batch)
            all_true.append(true.detach().to('cpu', non_blocking=True))
            all_pred.append(pred.detach().to('cpu', non_blocking=True))
        all_true, all_pred = torch.cat(all_true), torch.cat(all_pred)

        if i == 0:
            input_dict = {'y_pred': all_pred.squeeze(),
                          'y_true': all_true.squeeze()}
            result_dict = evaluator.eval(input_dict)
            logging.info(f"{split_names[i]}: MAE = {result_dict['mae']}")  # Get MAE.
        else:
            input_dict = {'y_pred': all_pred.squeeze()}
            evaluator.save_test_submission(input_dict=input_dict,
                                           dir_path=cfg.run_dir,
                                           mode=split_names[i])


@ register_train('log-attn-weights')
def log_attn_weights(loggers, loaders, model, optimizer=None, scheduler=None):
    """
    Customized pipeline to inference on the test set and log the attention
    weights in Transformer modules.

    Args:
        loggers: Unused, exists just for API compatibility
        loaders: List of loaders
        model (torch.nn.Module): GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    import os.path as osp
    from torch_geometric.loader.dataloader import DataLoader
    from grit.utils import unbatch, unbatch_edge_index

    start_time = time.perf_counter()

    # The last loader is a test set.
    l = loaders[-1]
    # To get a random sample, create a new loader that shuffles the test set.
    loader = DataLoader(l.dataset, batch_size=l.batch_size,
                        shuffle=True, num_workers=0)

    output = []
    # batch = next(iter(loader))  # Run one random batch.
    for b_index, batch in enumerate(loader):
        bsize = batch.batch.max().item() + 1  # Batch size.
        if len(output) >= 128:
            break
        print(f">> Batch {b_index}:")

        X_orig = unbatch(batch.x.cpu(), batch.batch.cpu())
        batch.to(torch.device(cfg.device))
        model.eval()
        model(batch)

        # Unbatch to individual graphs.
        X = unbatch(batch.x.cpu(), batch.batch.cpu())
        edge_indices = unbatch_edge_index(batch.edge_index.cpu(),
                                          batch.batch.cpu())
        graphs = []
        for i in range(bsize):
            graphs.append({'num_nodes': len(X[i]),
                           'x_orig': X_orig[i],
                           'x_final': X[i],
                           'edge_index': edge_indices[i],
                           'attn_weights': []  # List with attn weights in layers from 0 to L-1.
                           })

        # Iterate through GPS layers and pull out stored attn weights.
        for l_i, (name, module) in enumerate(model.layers.named_children()):
            if hasattr(module, 'attn_weights'):
                print(l_i, name, module.attn_weights.shape)
                for g_i in range(bsize):
                    # Clip to the number of nodes in this graph.
                    # num_nodes = graphs[g_i]['num_nodes']
                    # aw = module.attn_weights[g_i, :num_nodes, :num_nodes]
                    aw = module.attn_weights[g_i]
                    graphs[g_i]['attn_weights'].append(aw.cpu())
        output += graphs

    logging.info(
        f"[*] Collected a total of {len(output)} graphs and their "
        f"attention weights for {len(output[0]['attn_weights'])} layers.")

    # Save the graphs and their attention stats.
    save_file = osp.join(cfg.run_dir, 'graph_attn_stats.pt')
    logging.info(f"Saving to file: {save_file}")
    torch.save(output, save_file)

    logging.info(f'Done! took: {time.perf_counter() - start_time:.2f}s')




