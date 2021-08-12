# logging of training


from datetime import datetime

class Log(object):

    def __init__(self, path):
        self.exp_name = path
        self.dict_metrics = {"loss":"Loss", "acc_all":"Acc all", "acc_mean":"Acc mean", "jacc":"IoU", "dice":"Dice"}

    def start_log(self, lr, n_epochs):
        '''
            start log
        '''
        with open("log/{}.txt".format(self.exp_name), mode='a', encoding='utf-8') as f:
            f.write('#' * 80 + '\n')
            f.write('#' * 29 + '   Start experiment   ' + '#' * 29 + '\n')
            f.write('#' * 80 + '\n')
            f.write('\n')
            f.write('Learning Rate:    {}\n'.format(lr))
            f.write('Number of Epochs: {}\n'.format(n_epochs))
            f.write('\n')


    def add_log(self, epoch, metric, time, learing_type):
        '''
            save log
        '''
        with open("log/{}.txt".format(self.exp_name), mode='a', encoding='utf-8') as f:
            if learing_type == 'train':
                f.write('=' * 80 + '\n')
                f.write('-' * 10 + '\n')
                f.write('Epoch {}\n'.format(epoch))
                f.write('-' * 10 + '\n')
            f.write(learing_type)
            for i in metric:
                f.write(" {}: {:.4f} ".format(self.dict_metrics[i], metric[i]))
            f.write('\n')
            f.write('-' * 10 + '\n')
            if learing_type == 'val':
                f.write('Time {:.0f}m\n'.format(time))


    def start_csv(self):
        '''
            start csv
        '''
        with open("log/{}.csv".format(self.exp_name), mode='a', encoding='utf-8') as f:
            for i in self.dict_metrics:
                f.write("train_{}, ".format(i))
            for i in self.dict_metrics:
                f.write("val_{}, ".format(i))
            f.write('\n')


    def add_csv(self, epoch, metric, time, learing_type):
        '''
        '''
        with open("log/{}.csv".format(self.exp_name), mode='a', encoding='utf-8') as f:
            for i in metric:
                f.write("{:.4f}, ".format(metric[i]))
            if learing_type == 'val':
                f.write('\n')
