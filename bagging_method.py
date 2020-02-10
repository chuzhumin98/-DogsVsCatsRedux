import os

def bagging_output(results, method='average', path='./submit/bagging.csv', threshold=0.98):
    if method == 'average':
        for qid in results:
            results[qid] = sum(results[qid]) / float(len(results[qid]))
    elif method == 'hard':
        for qid in results:
            if sum(results[qid]) / float(len(results[qid])) > 0.5:
                results[qid] = threshold
            else:
                results[qid] = 1 - threshold

    results = sorted(results.items(), key=lambda item: item[0])

    file_out = open(path, 'w')
    file_out.write('id,label\n')
    for i in range(len(results)):
        file_out.write('{},{}\n'.format(results[i][0], results[i][1]))
    file_out.close()

if __name__ == '__main__':
    path = 'deeps_1'

    results = {}
    for dirpath, dirnames, filenames in os.walk(os.path.join('submit', path)):
        for fn in filenames:
            if not fn.endswith('.csv'):
                continue
            file = open(os.path.join('submit', path, fn), 'r')
            line = file.readline()
            while True:
                line = file.readline()
                if line:
                    qid, score = line.split(',', 1)
                    qid, score = int(qid), float(score)
                    if qid not in results:
                        results[qid] = []

                    results[qid].append(score)
                else:
                    break

            file.close()


    if not os.path.exists('./submit/bagging'):
        os.makedirs('./submit/bagging')

    method = 'hard' # 'average'

    bagging_output(results, method=method, path='./submit/bagging/bagging_{}_{}.csv'.format(path, method))