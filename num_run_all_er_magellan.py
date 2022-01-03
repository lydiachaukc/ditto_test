import os
import time

datasets = """Dirty/iTunes-Amazon
Dirty/Walmart-Amazon
Structured/Amazon-Google
Structured/Walmart-Amazon
Textual/Abt-Buy
Textual/Company""".split('\n')

special_datasets = {
    'Structured/Beer': (40, 8),
    'Structured/iTunes-Amazon': (12, 8),
    'Dirty/iTunes-Amazon': (12, 8),
    'Textual/Company': (32, 8)
}

ops = """swap
swap
swap
swap
swap
swap""".split('\n')


lms = ['bert-base-uncased', 'bert-base-uncased', 'bert-base-uncased', 'bert-base-uncased', 'bert-base-uncased', 'bert-base-uncased']

# lms = ['xlnet', 'roberta', 'roberta', 'roberta', 'xlnet', 'bert',
#        'bert', 'xlnet', 'roberta', 'bert', 'roberta', 'roberta', 'bert']

# lms = """distilbert
# bert
# xlnet
# roberta""".split('\n')

for dataset, op, lm in zip(datasets, ops, lms):
    if dataset in special_datasets:
        batch_size, epochs = special_datasets[dataset]
    else:
        batch_size, epochs = 12, 15

    for da in [True, False]:
        for dk in [True, False]:
            for run_id in range(1):
                cmd = """python train_ditto.py \
              --task %s \
              --logdir results_ditto/ \
              --finetuning \
              --batch_size %d \
              --lr 3e-5 \
              --fp16 \
              --lm %s \
              --n_epochs %d \
              --run_id %d""" % (dataset, batch_size, lm, epochs, run_id)
              --summarize'
                if da:
                    cmd += ' --da %s' % op
                if dk:
                    cmd += ' --dk general'
                print(cmd)
                os.system(cmd)
