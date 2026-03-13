import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from matchms import calculate_scores
from matchms import Spectrum
from concurrent.futures import ThreadPoolExecutor
import time


def _run_single_bootstrap(
    b,
    spectra_binned,
    global_bins_arr,
    internal_ids,
    id_to_index,
    similarity_metric,
    seed,
    k,
    track_bins=False,
):
    rng = np.random.default_rng(seed + b)

    N = len(spectra_binned)
    P = len(global_bins_arr)

    pair_sim_sum = defaultdict(float)
    pair_counts = defaultdict(int)
    edge_support = Counter()

    sampled_indices = rng.integers(0, P, size=P)
    sampled_bins_arr = np.unique(global_bins_arr[sampled_indices])

    spectra_boot = []

    for idx, (internal_id, spec) in enumerate(zip(internal_ids, spectra_binned)):
        mz = spec.peaks.mz
        intens = spec.peaks.intensities

        mask = np.isin(mz, sampled_bins_arr)
        mz_kept = mz[mask]
        int_kept = intens[mask]

        if mz_kept.size == 0:
            dummy_mz = float(global_bins_arr.max() + 1000.0 + idx)
            mz_kept = np.array([dummy_mz], dtype="float32")
            int_kept = np.array([1.0], dtype="float32")

        meta = {**spec.metadata, "internal_id": internal_id}

        spectra_boot.append(
            Spectrum(
                mz=mz_kept.astype("float32"),
                intensities=int_kept.astype("float32"),
                metadata=meta,
            )
        )

    sim_matrix = np.zeros((N, N), dtype=float)

    scores = calculate_scores(
        references=spectra_boot,
        queries=spectra_boot,
        similarity_function=similarity_metric,
        is_symmetric=True,
    )

    for item in scores:
        if len(item) == 3:
            ref_spec, qry_spec, sim_val = item
        else:
            (ref_spec, qry_spec), sim_val = item

        i = id_to_index[ref_spec.metadata["internal_id"]]
        j = id_to_index[qry_spec.metadata["internal_id"]]

        if i == j:
            continue

        sim = float(sim_val[0])
        sim_matrix[i, j] = sim
        sim_matrix[j, i] = sim

        key = tuple(sorted((internal_ids[i], internal_ids[j])))
        pair_sim_sum[key] += sim
        pair_counts[key] += 1

    # mutual kNN
    k_eff = min(k, N - 1)
    all_knn = []

    for i in range(N):
        row = sim_matrix[i].copy()
        row[i] = -np.inf
        knn_i = np.argpartition(row, -k_eff)[-k_eff:]
        knn_i = knn_i[np.argsort(row[knn_i])[::-1]]
        all_knn.append(set(knn_i))

    for i in range(N):
        for j in all_knn[i]:
            if i < j and i in all_knn[j]:
                key = tuple(sorted((internal_ids[i], internal_ids[j])))
                edge_support[key] += 1

    result = {
        "b": b,
        "pair_sim_sum": pair_sim_sum,
        "pair_counts": pair_counts,
        "edge_support": edge_support,
    }

    if track_bins:
        sampled_arr = np.unique(sampled_bins_arr)
        missing_arr = np.setdiff1d(global_bins_arr, sampled_arr)
        result["sampled_bins"] = sampled_arr
        result["missing_bins"] = missing_arr

    return result


def run_bootstrap_batch(
    batch_bootstrap_ids,
    spectra_binned,
    global_bins_arr,
    internal_ids,
    id_to_index,
    similarity_metric,
    seed,
    k,
    B,
    collect_history=False,
    track_bins=False,
    verbose=False,
):
    t_batch_start = time.perf_counter()

    # Fast mode - No history or bin tracking
    if not collect_history:
        batch_pair_sim_sum = defaultdict(float)
        batch_pair_counts = defaultdict(int)
        batch_edge_support = Counter()

        for b in batch_bootstrap_ids:
            res = _run_single_bootstrap(
                b=b,
                spectra_binned=spectra_binned,
                global_bins_arr=global_bins_arr,
                internal_ids=internal_ids,
                id_to_index=id_to_index,
                similarity_metric=similarity_metric,
                seed=seed,
                k=k,
                track_bins=False,
            )

            for key, val in res["pair_sim_sum"].items():
                batch_pair_sim_sum[key] += val

            for key, val in res["pair_counts"].items():
                batch_pair_counts[key] += val

            for key, val in res["edge_support"].items():
                batch_edge_support[key] += val

            if verbose and ((b + 1) % 10 == 0 or (b + 1) == B):
                print(f"[bootstrap {b+1}] done", flush=True)

        t_batch_end = time.perf_counter()
        if verbose:
            print(
                f"Batch {batch_bootstrap_ids[0]+1}-{batch_bootstrap_ids[-1]+1} "
                f"finished in {t_batch_end - t_batch_start:.2f} s",
                flush=True,
            )

        return batch_pair_sim_sum, batch_pair_counts, batch_edge_support
    
    # History mode - Collect detailed results for each bootstrap
    batch_results = []

    for b in batch_bootstrap_ids:
        res = _run_single_bootstrap(
            b=b,
            spectra_binned=spectra_binned,
            global_bins_arr=global_bins_arr,
            internal_ids=internal_ids,
            id_to_index=id_to_index,
            similarity_metric=similarity_metric,
            seed=seed,
            k=k,
            track_bins=track_bins,
        )
        batch_results.append(res)

        if verbose and ((b + 1) % 10 == 0 or (b + 1) == B):
            print(f"[bootstrap {b+1}] done", flush=True)

    t_batch_end = time.perf_counter()
    if verbose:
        print(
            f"Batch {batch_bootstrap_ids[0]+1}-{batch_bootstrap_ids[-1]+1} "
            f"finished in {t_batch_end - t_batch_start:.2f} s",
            flush=True,
        )

    return batch_results


def calculate_boostrapping(
    spectra_binned,
    global_bins,
    B=100,
    k=5,
    similarity_metric=None,
    n_jobs=4,
    batch_size=10,
    seed=42,
    return_history=False,
    track_bins=False,
    label_mode="feature",
    return_label_map=True,
    verbose: bool = True,
):
    if similarity_metric is None:
        raise ValueError("similarity_metric must be provided")

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    batch_size = min(batch_size, B)

    if global_bins is None or not hasattr(global_bins, "__len__"):
        raise TypeError(
            "global_bins must be an array/list of bin m/z values. "
            "Did you accidentally pass the global_bins *function* instead of the computed bins array?"
        )



    def make_unique(labels):
        counts = Counter(labels)
        seen = Counter()
        out = []
        for lab in labels:
            lab = str(lab)
            if counts[lab] == 1:
                out.append(lab)
            else:
                seen[lab] += 1
                out.append(f"{lab}__{seen[lab]}")
        return out

    N = len(spectra_binned)
    global_bins_arr = np.asarray(global_bins)

    pair_sim_sum = defaultdict(float)
    pair_counts = defaultdict(int)
    edge_support = Counter()

    scan_labels = []
    feature_labels = []
    internal_ids = []
    id_to_index = {}

    for idx, spec in enumerate(spectra_binned):
        md = spec.metadata

        scan_number = (
            getattr(spec, "get", lambda x: None)("scans")
            or md.get("scans") or md.get("SCANS")
            or md.get("scan_number") or md.get("SCAN_NUMBER")
            or f"scan_{idx}"
        )

        scan_labels.append(str(scan_number))

        feature_id = md.get("feature_id", md.get("FEATURE_ID", f"feat_{idx}"))
        feature_labels.append(str(feature_id))

        internal_id = f"INTFID_{feature_id}_{idx}"
        internal_ids.append(internal_id)
        id_to_index[internal_id] = idx

    scan_labels_u = make_unique(scan_labels)
    feature_labels_u = make_unique(feature_labels)
    internal_ids_u = make_unique(internal_ids)

    lm = label_mode.lower()

    if lm == "scan":
        out_labels = scan_labels_u
    elif lm == "internal":
        out_labels = internal_ids_u
    else:
        out_labels = feature_labels_u

    label_map = pd.DataFrame(
        {
            "out_label": out_labels,
            "scan": scan_labels,
            "scan_unique": scan_labels_u,
            "feature_id": feature_labels,
            "feature_unique": feature_labels_u,
            "internal_id": internal_ids,
        }
    )

    collect_history = return_history or track_bins

    bootstrap_ids = list(range(B))
    batches = [
        bootstrap_ids[i:i + batch_size]
        for i in range(0, B, batch_size)
    ]

    args = [
        (
            batch,
            spectra_binned,
            global_bins_arr,
            internal_ids,
            id_to_index,
            similarity_metric,
            seed,
            k,
            B,
            collect_history,
            track_bins,
            verbose,
        )
        for batch in batches
    ]

    total_start = time.perf_counter()
    if verbose:
        print(
            f"Running {B} bootstraps in {len(batches)} batches "
            f"(batch_size={batch_size}) with {n_jobs} workers",
            flush=True,
        )

    compute_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(lambda x: run_bootstrap_batch(*x), args))
    compute_end = time.perf_counter()

    if verbose:
        print(
            f"Bootstrap batch execution finished in {compute_end - compute_start:.2f} seconds",
            flush=True,
        )

    # Fast mode - just aggregate batch results without reconstructing history
    if not collect_history:
        if verbose:
            print("Merging batch results...", flush=True)

        merge_start = time.perf_counter()

        for ps, pc, es in results:
            for key, val in ps.items():
                pair_sim_sum[key] += val

            for key, val in pc.items():
                pair_counts[key] += val

            for key, val in es.items():
                edge_support[key] += val

        merge_end = time.perf_counter()
        total_end = time.perf_counter()

        if verbose:
            print(f"Merge finished in {merge_end - merge_start:.2f} seconds", flush=True)
            print(f"Total bootstrapping completed in {total_end - total_start:.2f} seconds", flush=True)

        mean_sim = np.eye(N, dtype="float32")
        for (id_i, id_j), total in pair_sim_sum.items():
            i = id_to_index[id_i]
            j = id_to_index[id_j]
            cnt = pair_counts[(id_i, id_j)]
            mean_sim[i, j] = total / cnt
            mean_sim[j, i] = total / cnt

        edge_mat = np.zeros((N, N), dtype="float32")
        for (id_i, id_j), cnt in edge_support.items():
            i = id_to_index[id_i]
            j = id_to_index[id_j]
            edge_mat[i, j] = cnt / B
            edge_mat[j, i] = cnt / B

        df_mean_sim = pd.DataFrame(mean_sim, index=out_labels, columns=out_labels)
        df_edge_sup = pd.DataFrame(edge_mat, index=out_labels, columns=out_labels)

        if return_label_map:
            return df_mean_sim, df_edge_sup, label_map
        return df_mean_sim, df_edge_sup

    # History mode - reconstruct cumulative history from per-bootstrap results
    if verbose:
        print("Reconstructing cumulative history from per-bootstrap results...", flush=True)

    replay_start = time.perf_counter()

    flat_results = []
    for batch_res in results:
        flat_results.extend(batch_res)

    flat_results.sort(key=lambda x: x["b"])

    hist_mean_sim = []
    hist_edge_sup = []
    hist_sampled_bins = []
    hist_missing_bins = []

    for res in flat_results:
        for key, val in res["pair_sim_sum"].items():
            pair_sim_sum[key] += val

        for key, val in res["pair_counts"].items():
            pair_counts[key] += val

        for key, val in res["edge_support"].items():
            edge_support[key] += val

        cur_mean = np.eye(N, dtype="float32")
        for (id_i, id_j), total in pair_sim_sum.items():
            i = id_to_index[id_i]
            j = id_to_index[id_j]
            cnt = pair_counts[(id_i, id_j)]
            cur_mean[i, j] = total / cnt
            cur_mean[j, i] = total / cnt

        cur_edge = np.zeros((N, N), dtype="float32")
        denom = float(res["b"] + 1)
        for (id_i, id_j), cnt in edge_support.items():
            i = id_to_index[id_i]
            j = id_to_index[id_j]
            cur_edge[i, j] = cnt / denom
            cur_edge[j, i] = cnt / denom

        hist_mean_sim.append(cur_mean)
        hist_edge_sup.append(cur_edge)

        if track_bins:
            hist_sampled_bins.append(res["sampled_bins"])
            hist_missing_bins.append(res["missing_bins"])

    replay_end = time.perf_counter()
    total_end = time.perf_counter()

    if verbose:
        print(f"History reconstruction finished in {replay_end - replay_start:.2f} seconds", flush=True)
        print(f"Total bootstrapping completed in {total_end - total_start:.2f} seconds", flush=True)

    mean_sim = pd.DataFrame(hist_mean_sim[-1], index=out_labels, columns=out_labels)
    edge_sup = pd.DataFrame(hist_edge_sup[-1], index=out_labels, columns=out_labels)

    history = {
        "mean_sim": hist_mean_sim,
        "edge_sup": hist_edge_sup,
    }

    if track_bins:
        history["sampled_bins"] = hist_sampled_bins
        history["missing_bins"] = hist_missing_bins

    if return_label_map:
        history["label_map"] = label_map
        history["label_mode"] = lm

    return mean_sim, edge_sup, history