import json
import numpy as np
import pandas as pd

labs = ['normal','dr','glaucoma','csc','disc_edema','macular_scar','myopia','retinal_det','retinitis_pig']
core = ['normal','dr','glaucoma','macular_scar','myopia']
rows = []
for s in ('42','43','44'):
    d = f'outputs/runs/stage2/seed_{s}'
    df = pd.read_csv(f'{d}/metrics.csv')
    r = json.load(open(f'{d}/run.json'))
    b = df.loc[df.val_macro_auroc.idxmax()]
    rec = dict(seed=s, best_ep=int(b.epoch), macro=float(b.val_macro_auroc), dr_auroc=float(b.auroc_dr),
               final_macro=float(df.val_macro_auroc.iloc[-1]), loss_first=float(df.train_loss.iloc[0]),
               loss_last=float(df.train_loss.iloc[-1]), epoch_seconds=float(df.seconds.mean()),
               peak_vram=float(df.peak_vram_gb.max()), nonfinite=int(df.nonfinite_batches.sum()),
               hours=round(r['seconds']/3600, 2), reload_delta=r['reload_check']['max_abs_auroc_delta'])
    for l in labs:
        rec['auroc_' + l] = float(b['auroc_' + l])
    rows.append(rec)

t = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print(t[['seed','best_ep','macro','dr_auroc','final_macro','hours','epoch_seconds','peak_vram','nonfinite','reload_delta']].to_string(index=False))
print()
for k in ('macro','dr_auroc'):
    v = t[k].values
    print('%-9s mean %.4f  std %.4f  min %.4f  max %.4f' % (k, v.mean(), v.std(ddof=1), v.min(), v.max()))
print()
print('per-label AUROC at each seed best epoch:')
for l in labs:
    v = t['auroc_' + l].values
    vals = ', '.join('%.4f' % x for x in v)
    role = 'core' if l in core else 'not-in-macro'
    print('  %-15s %.4f +- %.4f  [%s]  %s' % (l, v.mean(), v.std(ddof=1), vals, role))
print()
print('final-epoch macro:', [round(x, 4) for x in t.final_macro])
print('last-5-epoch macro:')
for s in ('42','43','44'):
    df = pd.read_csv(f'outputs/runs/stage2/seed_{s}/metrics.csv')
    print('  ', s, [round(x, 4) for x in df.val_macro_auroc.tail(5)])
t.to_csv('outputs/reports/stage2_3seed_summary.csv', index=False)
print('saved -> outputs/reports/stage2_3seed_summary.csv')
