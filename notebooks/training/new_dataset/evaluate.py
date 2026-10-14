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


def build_report(results, predictors, histories, encoder):
    import pandas as pd
    from IPython.display import display
    frame = pd.DataFrame(results.values())
    base = float(frame.iloc[0]['Test_Acc'])
    frame['Delta_vs_Baseline'] = frame['Test_Acc'] - base
    frame['Head_90split_Latency_ms'] = None
    latency = {}
    for name, predictor in predictors.items():
        stats = measure_latency_ms(lambda: predictor(test_features_raw[:1].to(device)))
        latency[name] = stats
        frame.loc[frame.Method == name, 'Head_90split_Latency_ms'] = stats['mean']
    # Encoder timing uses synthetic pixels already on device; excludes PIL, OpenCV and IO.
    pixels = torch.zeros(1, 3, 224, 224, device=device)
    latency['encoder_raw'] = measure_latency_ms(lambda: encoder(pixels))
    latency['encoder_5_views'] = measure_latency_ms(lambda: encoder(pixels.expand(5, -1, -1, -1)))
    frame.to_csv(output_dir / 'benchmark_results.csv', index=False)
    display(frame)
    summary = {'dataset': manifest['fingerprint'], 'classes': class_names,
               'device': str(device), 'backbone': MODEL_ID, 'seed': SEED,
               'latency_scope': 'Representative 9:1 head on cached raw features, repeated per method in the table; encoder on device-resident synthetic pixels. Not end-to-end.',
               'latency_ms': latency, 'results': list(results.values())}
    (output_dir / 'benchmark_results.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    (output_dir / 'training_history.json').write_text(json.dumps(histories, indent=2), encoding='utf-8')
    (output_dir / 'environment.json').write_text(json.dumps({
        'torch': torch.__version__, 'torchvision': torchvision.__version__,
        'transformers': __import__('transformers').__version__, 'device': str(device),
        'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None}, indent=2), encoding='utf-8')

    def finish(name):
        plt.tight_layout()
        plt.savefig(output_dir / f'{name}.png', dpi=160, bbox_inches='tight')
        plt.show()

    plt.figure(figsize=(13, 6))
    sns.barplot(data=frame, x='Method', y='Test_Acc', hue='Scenario')
    plt.xticks(rotation=18, ha='right'); plt.ylabel('Test Accuracy (%)')
    finish('01_accuracy')
    for metric, title, filename in [('loss', 'Training Loss', '02_loss'),
                                     ('val_acc', 'Validation Accuracy (%)', '03_validation')]:
        plt.figure(figsize=(10, 5))
        for name, history in histories.items():
            values = history[metric]
            if any(v is not None for v in values):
                plt.plot(range(1, len(values) + 1), values, label=name)
        plt.xlabel('Epoch'); plt.ylabel(title); plt.legend()
        finish(filename)
    plt.figure(figsize=(11, 5))
    plt.bar(latency.keys(), [v['mean'] for v in latency.values()])
    plt.xticks(rotation=25, ha='right'); plt.ylabel('Measured latency (ms; scope in JSON)')
    finish('04_latency')
    plt.figure(figsize=(11, 5))
    sns.barplot(data=frame, x='Method', y='Trainable_Params', hue='Scenario')
    plt.xticks(rotation=20, ha='right'); plt.ylabel('Trainable parameters')
    finish('05_parameters')
    # Select only by validation accuracy; do not choose the winning method on test.
    candidates = frame[frame['Val_Acc'].notna() & frame.Method.isin(predictors)]
    selected = candidates.sort_values('Val_Acc', ascending=False, kind='stable').iloc[0]['Method']
    with torch.no_grad():
        predictor = predictors[selected]
        preds = predictor(test_features_tta.to(device)).argmax(-1).cpu().numpy()
        raw_preds = predictor(test_features_raw.to(device)).argmax(-1).cpu().numpy()
    truth = test_labels.cpu().numpy()
    matrix = confusion_matrix(truth, preds, labels=list(range(num_classes)))
    plt.figure(figsize=(14, 12))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{selected}: validation-selected, TTA, 9:1 support'); plt.xlabel('Predicted'); plt.ylabel('True')
    finish('06_confusion_matrix')
    gains = [(100 * (preds[truth == i] == i).mean() - 100 * (raw_preds[truth == i] == i).mean())
             for i in range(num_classes)]
    order = np.argsort(gains)[-10:]
    plt.figure(figsize=(10, 6))
    plt.barh([class_names[i] for i in order], np.array(gains)[order])
    plt.xlabel('TTA gain vs raw (percentage points), same validation-selected head')
    finish('07_per_class_tta_gain')
    report = classification_report(truth, preds, target_names=class_names, output_dict=True, zero_division=0)
    (output_dir / 'classification_report.json').write_text(json.dumps({
        'selected_by_validation': selected, 'report': report,
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
