# SoundMatch-SR results

## Table 1/2 — cross-domain retrieval (TEST split)

| variant | real→real mAP (control) | synth→real mAP | real→synth mAP | **domain gap** | synth→real R@1 | instance MRR |
|---|---|---|---|---|---|---|
| frozen | 0.935 | 0.774 [0.766,0.783] | 0.813 | +0.161 | 0.008 | nan |
| invariant | 0.988 | 0.970 [0.966,0.974] | 0.979 | +0.018 | 0.010 | nan |
| sensitive | 0.206 | 0.157 [0.156,0.158] | 0.156 | +0.049 | 0.004 | nan |

## Fig 2 — domain-gap diagnostics (TEST split)

| variant | Proxy-A-dist | event-probe acc | domain-probe acc | identity−domain | silhouette(event) | silhouette(domain) |
|---|---|---|---|---|---|---|
| frozen | 1.268 | 0.953 | 0.902 | +0.051 | 0.176 | 0.043 |
| invariant | 1.385 | 0.972 | 0.844 | +0.128 | 0.751 | -0.002 |
| sensitive | 1.879 | 0.844 | 0.966 | -0.122 | -0.040 | 0.747 |
