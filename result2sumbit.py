
def result2submit(path_in, path_out, threshold, reversed=False):
    file_in = open(path_in, 'r')
    file_out = open(path_out, 'w')

    line = file_in.readline()
    file_out.write(line.strip()+'\n')

    while True:
        line = file_in.readline()
        if line:
            id, score = line.split(',', 1)
            score = float(score.strip())
            if score > threshold:
                score = threshold
            if score < 1 - threshold:
                score = 1 - threshold
            if reversed:
                score = 1 - score
            file_out.write('{},{}\n'.format(id.strip(), score))
        else:
            break



    file_out.close()
    file_in.close()

if __name__ == '__main__':
    threshold = 0.98
    run_name = 'Mixture_hidden50'
    is_reversed = False
    # true for traditional method, false for deep
    result2submit('./result/result_{}.csv'.format(run_name), './submit/submit_{}_{}.csv'.format(run_name, threshold), threshold, reversed=is_reversed)