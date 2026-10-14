"""Runtime reports: measured results only, with explicit latency scope."""
def synchronize():
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


def measure_latency_ms(fn, num_runs=50, warmup=5):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        synchronize()
        times = []
        for _ in range(num_runs):
            synchronize()
            start = time.perf_counter()
            fn()
            synchronize()
            times.append((time.perf_counter() - start) * 1000)
    return {key: float(value) for key, value in {
        'mean': np.mean(times), 'median': np.median(times),
        'p95': np.percentile(times, 95), 'p99': np.percentile(times, 99)}.items()}


def save_checkpoint(name, model, **extra):
    # CoOp saves only the learned context; pretrained CLIP is shared and frozen.
    state = {'ctx': model.ctx.detach().cpu()} if hasattr(model, 'ctx') else {
        key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save({'state_dict': state, 'classes': class_names,
                'fingerprint': manifest['fingerprint'], 'seed': SEED,
                'backbone': MODEL_ID,
                'alpha': getattr(model, 'alpha', None), 'beta': getattr(model, 'beta', None),
                'logit_scale': globals().get('logit_scale_val'), **extra}, output_dir / f'{name}.pt')


def measure_scenarios(scenario_predictors):
    global scenario_latency
    scenarios = ['Raw Image (Full Trainval)', 'TTA 5-view (9:1 Split)', 'TTA 5-view (Full Trainval)']
    scenario_latency = {}
    for method, heads in scenario_predictors.items():
        scenario_latency[method] = []
        for scenario, head in zip(scenarios, heads):
            sample = (test_features_raw if scenario.startswith('Raw') else test_features_tta)[:1].to(device)
            stats = measure_latency_ms(lambda: head(sample))
            scenario_latency[method].append(stats['mean'])
            for row in results.values():
                if row['Method'] == method and row['Scenario'] == scenario:
                    row['Head_Latency_ms'] = stats['mean']
                    row['Head_Throughput_Samples_Sec'] = 1000 / stats['mean']
                    row['Head_Latency_Stats'] = stats


def pareto_frontier(points):
    best, frontier = -float('inf'), []
    for latency, accuracy in sorted((float(x), float(y)) for x, y in points if np.isfinite(x) and np.isfinite(y)):
        if accuracy > best:
            frontier.append((latency, accuracy)); best = accuracy
    return ([p[0] for p in frontier], [p[1] for p in frontier])


def matching_clip_results():
    path = output_dir.parent / 'clip' / 'benchmark_results.json'
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    return data if data['dataset'] == manifest['fingerprint'] else None


def load_matching_clip_scores():
    data = matching_clip_results()
    values = {'Raw': float('nan'), 'TTA_90': float('nan'), 'TTA_Full': float('nan')}
    scenarios = dict(zip(['Raw Image (Full Trainval)', 'TTA 5-view (9:1 Split)', 'TTA 5-view (Full Trainval)'], values))
    if data:
        for row in data['results']:
            if row['Method'] == 'CLIP + Prompt Tuning':
                values[scenarios[row['Scenario']]] = row['Test_Acc']
    else:
        print('CLIP comparison pending: run CLIP on the same dataset split and output root, then rerun this report cell.')
    return values


def matching_clip_latency(scenario):
    data = matching_clip_results()
    if data and data['device'] == str(device):
        for row in data['results']:
            if row['Method'] == 'CLIP + Prompt Tuning' and row['Scenario'] == scenario:
                return row.get('Head_Latency_ms', float('nan'))
    return float('nan')


def build_report(results, predictors, histories, encoder):
    import pandas as pd
    from IPython.display import display
    frame = pd.DataFrame(results.values()).drop(columns=['Head_Latency_Stats'], errors='ignore')
    frame.insert(0, 'STT', range(1, len(frame) + 1))
    frame['Preprocessing'] = np.where(frame.Scenario.str.startswith('Raw'), 'Raw Images', 'TTA 5 Views')
    base = float(frame.iloc[0]['Test_Acc'])
    frame['Delta_vs_Baseline'] = frame['Test_Acc'] - base
    latency = {key: row['Head_Latency_Stats'] for key, row in results.items()}
    # Encoder timing uses synthetic pixels already on device; excludes PIL, OpenCV and IO.
    pixels = torch.zeros(1, 3, 224, 224, device=device)
    latency['encoder_raw'] = measure_latency_ms(lambda: encoder(pixels))
    latency['encoder_5_views'] = measure_latency_ms(lambda: encoder(pixels.expand(5, -1, -1, -1)))
    frame.to_csv(output_dir / 'benchmark_results.csv', index=False)
    display(frame)
    summary = {'dataset': manifest['fingerprint'], 'classes': class_names,
               'device': str(device), 'backbone': MODEL_ID, 'seed': SEED,
               'latency_scope': 'Each scenario head on cached raw/TTA features; encoder measured separately on device-resident synthetic pixels. Not end-to-end.',
               'latency_ms': latency, 'results': list(results.values())}
    (output_dir / 'benchmark_results.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    (output_dir / 'training_history.json').write_text(json.dumps(histories, indent=2), encoding='utf-8')
    (output_dir / 'environment.json').write_text(json.dumps({
        'torch': torch.__version__, 'torchvision': torchvision.__version__,
        'transformers': __import__('transformers').__version__, 'device': str(device),
        'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None}, indent=2), encoding='utf-8')

    # Match the original report: CLIP CoOp / ResNet Cosine, both 9:1 + TTA.
    selected = 'CLIP + Prompt Tuning' if output_dir.name == 'clip' else 'ResNet-50 (Cosine Head)'
    with torch.no_grad():
        preds = predictors[selected](test_features_tta.to(device)).argmax(-1).cpu().numpy()
    truth = test_labels.cpu().numpy()
    matrix = confusion_matrix(truth, preds, labels=list(range(num_classes)))
    report = classification_report(truth, preds, target_names=class_names, output_dict=True, zero_division=0)
    (output_dir / 'classification_report.json').write_text(json.dumps({
        'reported_method': selected, 'report': report,
        'confusion_matrix': matrix.tolist()}, indent=2), encoding='utf-8')
    rows = frame.fillna('—').astype(str)
    table = '| ' + ' | '.join(rows.columns) + ' |\n'
    table += '| ' + ' | '.join(['---'] * len(rows.columns)) + ' |\n'
    table += '\n'.join('| ' + ' | '.join(row) + ' |' for row in rows.to_numpy())
    (output_dir / 'benchmark_report.md').write_text(
        f'# {MODEL_ID} — NewDataset\n\n{table}\n\n'
        'Latency: head-only and encoder-only, excluding image IO and preprocessing. '
        'Full trainval has no independent validation score.\n', encoding='utf-8')
    # Compare only with a matching split fingerprint, never legacy Oxford scores.
    sibling = output_dir.parent / ('resnet50' if output_dir.name == 'clip' else 'clip') / 'benchmark_results.json'
    if sibling.exists():
        other = json.loads(sibling.read_text(encoding='utf-8'))
        if other['dataset'] == manifest['fingerprint']:
            combined = pd.concat([frame.assign(Backbone=MODEL_ID),
                                  pd.DataFrame(other['results']).assign(Backbone=other['backbone'])])
            combined.to_csv(output_dir / 'clip_vs_resnet50.csv', index=False)
            display(combined[['Backbone', 'Method', 'Scenario', 'Val_Acc', 'Test_Acc']])
        else:
            print('Skipped comparison: different split fingerprints')
    print(f'Artifacts saved to {output_dir}')
