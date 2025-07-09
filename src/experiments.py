from scipy.stats import ttest_ind

def aa_test(y_true):
    group_a = y_true.sample(frac=0.5, random_state=42)
    group_b = y_true.drop(group_a.index)
    stat, pval = ttest_ind(group_a, group_b)
    print(f"A/A Test p-value: {pval:.3f} (should be > 0.05)")

def ab_test(y_true, preds_a, preds_b):
    errors_a = abs(y_true - preds_a)
    errors_b = abs(y_true - preds_b)
    stat, pval = ttest_ind(errors_a, errors_b)
    print(f"A/B Test p-value: {pval:.3f} (significant difference if < 0.05)")