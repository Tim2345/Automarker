import pandas as pd

dataset_dir = './COLA/resources'

train = '/in_domain_train.tsv'
dev_in = '/in_domain_dev.tsv'
dev_out = '/out_of_domain_dev.tsv'

set_types = ('train', 'dev', 'dev')
set_types2 = ('train', 'dev_in', 'dev_out')

data_sets = [pd.read_csv(dataset_dir+set, sep='\t', header=None) for set in (train, dev_in, dev_out)]

data_sets_temp = []
for df, set_type, set_type2 in zip(data_sets, set_types, set_types2):
    df_temp = df
    df_temp['SET'] = set_type
    df_temp['SET_2'] = set_type2

    data_sets_temp.append(df_temp)

dataset = pd.concat(data_sets_temp)
