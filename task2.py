import pandas as pd

df = pd.read_csv('transactions.txt', header = None)

#1
df.groupby(3).size()

#2
df.groupby(3)[2].mean()

#3
from scipy.stats import sem, t

def conf_int(n, mean_val, st_error):
    delt = st_error*t.ppf((1 + 0.90) / 2, n-1)
    print('Нижняя граница доверительного интервала', mean_val - delt)
    print('Верхняя граница доверительного интервала', mean_val + delt)


print('Cегмент AF')
segm_size = df.groupby(3)[2].size()[0]
segm_mean = df.groupby(3)[2].mean()[0]
segm_st_error = df.groupby(3)[2].sem()[0]
conf_int(segm_size, segm_mean, segm_st_error)

print('Cегмент R')
segm_size = df.groupby(3)[2].size()[1]
segm_mean = df.groupby(3)[2].mean()[1]
segm_st_error = df.groupby(3)[2].sem()[1]
conf_int(segm_size, segm_mean, segm_st_error)

#4
from scipy.stats import ttest_ind

sign_lvl = 0.1
ttest, sign_lvl_hyp = ttest_ind(df.loc[df[3] == 'AF'][2], df.loc[df[3] == 'R'][2])

if sign_lvl_hyp < sign_lvl:
    print('Гипотеза не принята')
else:
    print('Гипотеза принята')