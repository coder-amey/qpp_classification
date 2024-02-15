Introduction

Working with this repo
large_datasets

Param-tuning:
w_size, img_size, epochs, b_size

Results:
Model       Train            Val            Test        (loss/acc/auc)
ws20    .21/.96/.98     .6/.72/.86      1.4/.65/.85
ws30    .09/.96/.99     .09/.94/1       2.7/.6/.66
ws40    .12/.98/.99     .71/.77/.79     3.2/.65/.70
ws50    .05/.99/.999    .52/.89/.91     2.3/.60/.786

Model      ws_20            ws_30            ws_40        (loss/acc/auc)
ws20    .42/.88/.95     1.71/.75/.83    4.2/.64/.72
ws30    .79/.75/.84     .59/.91/.94     .81/.85/.92
ws40    1.1/.74/.79     .84/.80/.90     .74/.90/.93
ws50    1.49/.69/.76    1.2/.77/.84     .87/.83/.90

For small dataset, ws_30 model is most consistent. Results do vary by window_size.
Larger batch size leads to faster convergence.

For large dataset, models tend to overfit after certain epochs.
Larger window size leads to quicker overfitting.
Analysis of the loss curves indicates that 40 is the optimal window size.
And 20 is the optimal batch size.
Smaller window size degrades performance.
"""