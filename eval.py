# coding: utf-8

'''
select = 1
top1 = open('eval/predict.test.top1', 'w')

with open('dev/predict.test') as fp:
    for row in fp:
        if row == '\n':
            top1.write('\n')
        else:
            data = row.split(' ')
            top1.write(data[0] + ' ' + data[1] + ' ' + data[1 + select] + '\n')

top1.close()
'''

for pre1, pre2 in zip(['train', 'valid', 'test'], ['train', 'dev', 'test']):
    with open('data/%s.txt' % pre1) as f1:
        with open('dev/predict.%s' % pre2) as f2:
            with open('eval/predict.%s' % pre2, 'w') as f3:
                for row1, row2 in zip(f1, f2):
                    if row1 == '\n':
                        f3.write(row1)
                    else:
                        token = row1.split(' ')[0]
                        data = row2.split(' ')
                        data[0] = token
                        f3.write(' '.join(data))

