def run_test(network, data_loader, device, eval_mode=True):
    targets_all = []
    preds_all = []
    if eval_mode:
        network.eval()
    for idx, (feat_batch, targ_batch) in enumerate(tqdm.tqdm(data_loader)):
        feat = feat_batch.permute(0, 2, 1).contiguous()
        pred = network(feat).cpu().detach().numpy()
        targets_all.append(targ_batch.detach().numpy())
        preds_all.append(pred.squeeze())
    targets_all = np.concatenate(targets_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)
    return targets_all, preds_all