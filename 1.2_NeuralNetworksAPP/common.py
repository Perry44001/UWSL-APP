import os
import datetime
import sys


def splic_str(a):
    """
    将列表中的元素拼接成一个字符串。

    参数:
        a (list): 需要拼接的元素列表。

    返回:
        str: 拼接后的字符串。
    """
    str_p = ''
    for ia in range(len(a)):
        str_p = str_p + ' ' + a[ia]
    return str_p

class CheckPoint:
    def __init__(self, args):
        """
        初始化CheckPoint对象。

        参数:
            args: 包含配置参数的对象。
        """
        # 获取当前脚本的命令行参数
        self.getcmd = sys.argv
        # 将splic_str函数赋值给self.splic_str
        self.splic_str = splic_str
        # 将传入的args对象赋值给self.args
        self.args = args
        # 初始化加载模型的路径为空字符串
        self.load_model = ''
        # 获取当前时间并格式化为字符串
        now = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
        # 从数据加载器路径中提取文件名或目录名
        appendix = str.split(self.args.data_loader_path, '/')[-1]
        # 如果args.load为'.'，则设置保存目录为args.save_file
        if args.load == '.':
            # self.dir = args.save_file + args.model + '_' + \
            #            self.args.data_train[1]
            self.dir = args.save_file
        # 否则，设置保存目录为args.save_file + args.load
        else:
            self.dir = args.save_file + args.load

        # 如果保存目录不存在，则创建该目录
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        # 根据log.txt是否存在来决定文件的打开模式
        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        # 打开或创建log.txt文件，并将文件对象赋值给self.log_file
        self.log_file = open(self.dir + '/log.txt', open_type)
        # 打开或创建config.txt文件，并将文件对象赋值给f
        with open(self.dir + '/config.txt', open_type) as f:
            # 将命令行参数拼接后写入config.txt文件
            f.write('python' + self.splic_str(self.getcmd) + '\n')
            # 将当前时间写入config.txt文件
            f.write(now + '\n\n')
            # 遍历args对象的所有属性，并将属性名和属性值写入config.txt文件
            for arg in vars(args):
                # 'args.arg' is equivalent to 'getattr(args, arg)'
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            # 在config.txt文件末尾添加一个换行符
            f.write('\n')
    def write_log(self, log, refresh=False):
        """
        将日志信息写入日志文件。

        参数:
            log (str): 要写入的日志信息。
            refresh (bool, 可选): 是否刷新日志文件。默认为 False。

        返回:
            None
        """
        # 将日志信息写入日志文件，并在末尾添加一个换行符
        self.log_file.write(log + '\n')
        # 如果 refresh 参数为 True，则关闭并重新打开日志文件
        if refresh:
            self.log_file.close()
            self.log_file = open(self.dir + '/log.txt', 'a+')

    def done(self):
        """
        关闭日志文件。

        该方法用于关闭当前CheckPoint对象的日志文件。
        """
        self.log_file.close()

    def load(self):
        """
        加载模型的方法。

        根据不同的条件加载模型：
        - 如果 `self.args.resume` 为 -1，则加载最新的模型文件。
        - 如果 `self.args.resume` 为 0 且 `self.args.pre_train` 不为 '.'，则加载预训练模型。
        - 否则，加载指定 epoch 的模型文件。

        返回:
            str: 加载的模型文件路径。
        """
        # 如果 resume 参数为 -1，表示加载最新的模型
        if self.args.resume == -1:
            # 获取模型目录下的所有文件
            file_list = os.listdir(self.dir + 'model')
            # 对文件列表进行排序
            sorted(file_list)
            # 加载最新的模型文件
            self.load_model = os.path.join(self.dir, 'model', file_list[-1])

        # 如果 resume 参数为 0 且 pre_train 参数不为 '.'，表示加载预训练模型
        elif self.args.resume == 0:
            if self.args.pre_train != '.':
                # 打印加载模型的信息
                print('Loading model from {}.'.format(self.args.pre_train))
                # 设置加载的模型路径
                self.load_model = self.args.pre_train
                # 打印加载模型的模式
                print('Load_model_mode = 1')

        # 其他情况，表示加载指定 epoch 的模型
        else:
            # 构建模型文件路径
            self.load_model = os.path.join(self.dir, 'model',
                                           '{}_epoch_{}.pth'.format(self.args.model, self.args.resume))
            # 打印加载模型的模式
            print('Load_model_mode = 2')

        # 返回加载的模型文件路径
        return self.load_model
