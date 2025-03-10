import os
import sys
import time
import json
import csv
import numpy as np
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtSql import QSqlDatabase
from PyQt5.QtSql import QSqlQuery
from PyQt5.QtWidgets import QMessageBox, QTableWidgetItem, QMenu, QAbstractItemView, QStyleFactory
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from sympy.plotting import plot as symplot
from matplotlib import pyplot as pl
import matplotlib.axes._axes as axes
import matplotlib.figure as figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import datetime
from scipy.io import loadmat
from NetMain import net_main
from qtui import Ui_MainWindow

# 当你定义一个类时，可以通过继承多个父类来获得它们的属性和方法。在这个例子中，AppWidget类继承了QMainWindow和Ui_MainWindow两个类。
# 这样做的好处是可以将UI界面的代码和逻辑代码分离，使代码更加模块化和可维护。同时，通过继承QMainWindow类，AppWidget类也可以获得QMainWindow类的属性和方法，
class AppWidget(QMainWindow, Ui_MainWindow):
    """程序界面设定控制类"""

    def __init__(self):
        # 调用父类的初始化方法
        super().__init__()
        self.setupUi(self)
        # 对Sqlite数据库进行相关初始化
        self.init_db()
        # 调用界面初始化方法（一般会将UI界面的代码封装到另外一个方法函数中，而不直接放到__init__）
        self.init_ui()
        # 加载文件中存储的所有运行信息
        self.load_all_infos()
        self.model_name_list = []

    def init_db(self):
        """对数据库初始化"""
        # 指定要操作的数据库类型
        self.db = QSqlDatabase.addDatabase("QSQLITE")
        # 指定要使用的数据库（如果数据库不存在则会建立新的）
        self.db.setDatabaseName("./infos.db")
        # 打开数据库
        if not self.db.open():
            exit("无法建立数据库")
        # 创建一个用来执行sql语句的对象
        self.query = QSqlQuery()
        # 建立数据表
        sql = "create table if not exists person(" \
              "id integer primary key, " \
              "`model_name` varchar(50), " \
              "run_option varchar(10), " \
              "load_model_path varchar(50), " \
              "num_receiver varchar(10), " \
              "num_frequency varchar(10), " \
              "batchsize varchar(10), " \
              "train_epoch varchar(10), " \
              "data_load_path varchar(50), " \
              "learning_rate varchar(20), " \
              "save_file_path varchar(100), " \
              "save_file_name varchar(50), " \
              "run_date_time varchar(50), " \
              "note varchar(200)," \
              "train_MAE_range varchar(50), " \
              "train_MAE_depth varchar(50), " \
              "test_MA_range varchar(50), " \
              "test_MA_depth varchar(50), " \
              "test_MAE_range varchar(50), " \
              "test_MAE_depth varchar(50)) " 
        # 执行传递给它的 SQL 语句
        self.query.exec(sql)

    def init_ui(self):
        """界面初始化"""
        # RUN BUTTON 事件绑定，每次运行完，都要在数据库中添加一条记录
        self.pushButton_01_run.clicked.connect(self.add_new_run_info)

        # 信息表1(选项卡1中的大表)
        info_culumns = [
            '模型名称', '运行选项', '权重文件路径', '阵元数', '频点数', '批次大小', '训练周期', '数据集路径', 
            '学习率', '保存路径', '保存文件名', '运行时间', '备注', '训练误差-距离', '训练误差-深度',
            '测试平均准确度-距离', '测试平均准确度-深度', '测试平均误差-距离', '测试平均误差-深度'
        ]
        self.tableWidget_01_run_info.setColumnCount(len(info_culumns))
        self.tableWidget_01_run_info.setHorizontalHeaderLabels(info_culumns)

        # 禁用双击编辑单元格
        self.tableWidget_01_run_info.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # 改为选择一行
        self.tableWidget_01_run_info.setSelectionBehavior(QAbstractItemView.SelectRows)
        # 添加右击菜单
        self.tableWidget_01_run_info.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tableWidget_01_run_info.customContextMenuRequested.connect(self.generate_menu)

        # 按类型查找
        # 下拉菜单，查找类型
        find_type = ['模型名称', '运行选项', '权重文件路径', '阵元数', '频点数', '批次大小', '训练周期', '数据集路径', 
            '学习率', '保存路径', '保存文件名', '运行时间', '备注']
        for i, type_temp in enumerate(find_type):
            self.lineEdit_17_seek_type.addItem("")
            self.lineEdit_17_seek_type.setItemText(i, type_temp)
        # 当下拉菜单改变时，触发事件并判断是否显示查找日期范围
        self.lineEdit_17_seek_type.currentIndexChanged.connect(self.adjust_inputs_for_date_search)

        # 查找输入框
        # 查找日期范围（默认不显示，只有当选择查询日期范围时才显示）
        self.line_edit_star_time = QtWidgets.QDateTimeEdit(self.TableFrame)
        self.line_edit_star_time.setCalendarPopup(True)
        self.line_edit_star_time.setDisplayFormat("yyyy-MM-dd")
        self.line_edit_star_time.setVisible(False)
        self.line_edit_end_time = QtWidgets.QDateTimeEdit(self.TableFrame)
        self.line_edit_end_time.setCalendarPopup(True)
        self.line_edit_end_time.setDisplayFormat("yyyy-MM-dd")
        self.line_edit_end_time.setVisible(False)
        # 绑定显示所有信息按钮
        self.pushButton_02_checkall.clicked.connect(self.load_all_infos)
        # 绑定选项卡1大表的查询按钮和查询信息事件
        self.pushButton_03_seek.clicked.connect(self.search_info_show_table1)
        self.pushButton_04_save_table.clicked.connect(self.save_table_csv)

        # 初始化选项卡2 绘图主模块
        # 事件绑定
        # 绑定绘制按钮和绘图事件
        info_culumns_1 = [
            '模型名称', '数据集路径', '保存路径', '保存文件名', '备注'
        ]
        self.pushButton_06_plot.clicked.connect(self.plot_from_files)
        self.lineEdit_19_seek_type.addItems(['数据集路径', '保存路径', '保存文件名', '备注' ])
        # 查询信息表（选项卡2中的上面表格）
        self.tableWidget_02_plot_info.setColumnCount(len(info_culumns_1))
        self.tableWidget_02_plot_info.setHorizontalHeaderLabels(info_culumns_1)
        # 结果分析信息表（选项卡2中的下面表格）
        info_culumns_2 = [
            '模型名称', '训练误差-距离', '训练误差-深度', '测试平均准确度-距离', '测试平均准确度-深度', '测试平均误差-距离', '测试平均误差-深度'
        ]
        self.tableWidget_03_results.setColumnCount(len(info_culumns_2))
        self.tableWidget_03_results.setHorizontalHeaderLabels(info_culumns_2)
        # 绑定查询按钮和查询信息事件（第二个界面）
        self.pushButton_04_seek.clicked.connect(self.search_info_show_table2)

        # 初始化绘图区大小
        try:
            train_range_background = QtWidgets.QLabel(self)
            train_depth_background = QtWidgets.QLabel(self)
            test_range_background = QtWidgets.QLabel(self)
            test_depth_background = QtWidgets.QLabel(self)
            exp_range_background = QtWidgets.QLabel(self)
            exp_depth_background = QtWidgets.QLabel(self)

            self.train_range_layout.addWidget(train_range_background)
            self.train_depth_layout.addWidget(train_depth_background)
            self.test_range_layout.addWidget(test_range_background)
            self.test_depth_layout.addWidget(test_depth_background)
            self.exp_range_layout.addWidget(exp_range_background)
            self.exp_depth_layout.addWidget(exp_depth_background)
        except:
            pass

    def closeEvent(self, *args):
        """关闭窗口时要关闭数据库"""
        self.db.close()
        super().closeEvent(*args)

    def adjust_inputs_for_date_search(self):
        """如果选择了运行日期范围，则显示日期范围输入框，否则隐藏"""
        search_type = self.lineEdit_17_seek_type.currentText()
        if search_type == '运行日期范围':
            self.line_edit_star_time.setVisible(True)
            self.line_edit_end_time.setVisible(True)
            self.horizontalLayout.addWidget(self.line_edit_star_time)
            self.horizontalLayout.addWidget(self.line_edit_end_time)
            self.lineEdit_18_seek_info.setVisible(False)
        else:
            self.line_edit_star_time.setVisible(False)
            self.line_edit_end_time.setVisible(False)
            self.lineEdit_18_seek_info.setVisible(True)

    def search_info_show_table1(self):
        """ 从数据库查询信息(选项卡1的大表) """
        # 查询类型
        search_type = self.lineEdit_17_seek_type.currentText()

        if search_type == '运行日期范围':
            start_time = self.line_edit_star_time.text().split('-')
            stop_time = self.line_edit_end_time.text().split('-')
            start_time = datetime.date(int(start_time[0]), int(start_time[1]), int(start_time[2]))
            stop_time = datetime.date(int(stop_time[0]), int(stop_time[1]), int(stop_time[2]))
            if (stop_time - start_time).days < 0:
                QMessageBox.information(self, '提示', '开始日期不能小于结束日期')
                return

        # 获取要搜索的内容
        search_content = self.lineEdit_18_seek_info.text()
        search_type_map = [
            ('模型名称', 'model_name'),
            ('运行选项', 'run_option'),
            ('权重文件路径', 'load_model_path'),
            ('阵元数', 'num_receiver'),
            ('频点数', 'num_frequency'),
            ('批次大小', 'batchsize'),
            ('训练周期', 'epoch'),
            ('数据集路径', 'data_load_path'),
            ('学习率', 'learning_rate'),
            ('保存路径', 'save_file_path'),
            ('保存文件名', 'save_file_name'),
            ('运行日期范围', 'run_date_time'),
            ('备注', 'note'),
        ]
        for temp in search_type_map:
            if search_type == temp[0]:
                sql = "select count(*) from person where `{}`='{}'".format(temp[1], search_content)
                self.query.exec(sql)
                self.query.next()
                self.tableWidget_01_run_info.setRowCount(self.query.value(0))
                sql = "select * from person where `{}`='{}'".format(temp[1], search_content)
                self.query.exec(sql)
                i = 0
                while self.query.next():  # 每次查询一行记录
                    for j in range(1, 14):
                        new_item = QTableWidgetItem(self.query.value(j))
                        new_item.setTextAlignment(Qt.AlignHCenter)
                        self.tableWidget_01_run_info.setItem(i, j - 1, new_item)

                    i += 1
                break
        else:
            if search_type == "运行日期范围":
                sql = "select count(*) from person where birthday>='{}' and birthday<='{}'".format(start_time,
                                                                                                   stop_time)
                self.query.exec(sql)
                self.query.next()
                self.tableWidget_01_run_info.setRowCount(self.query.value(0))
                sql = "select * from person where birthday>='{}' and birthday<='{}'".format(start_time, stop_time)
                self.query.exec(sql)
                i = 0
                while self.query.next():  # 每次查询一行记录
                    for j in range(1, 14):
                        new_item = QTableWidgetItem(self.query.value(j))
                        new_item.setTextAlignment(Qt.AlignHCenter)
                        self.tableWidget_01_run_info.setItem(i, j - 1, new_item)

                    i += 1

    def search_info_show_table2(self):
        """ 从数据库查询信息(选项卡2的上表) """
        # 查询类型
        search_type = self.lineEdit_19_seek_type.currentText()

        # 获取要搜索的内容
        search_content = self.lineEdit_20_seek_info.text()
        search_type_map = [
            ('数据集路径', 'data_load_path'),
            ('保存路径', 'save_file_path'),
            ('保存文件名', 'save_file_name'),
            ('备注', 'note'),
        ]

        for temp in search_type_map:
            if search_type == temp[0]:
                search_content = search_content.replace("\\", '/')
                sql = "select count(*) from person where `{}`='{}'".format(temp[1], search_content)
                self.query.exec(sql)
                self.query.next()
                self.tableWidget_02_plot_info.setRowCount(self.query.value(0))
                sql = "select * from person where `{}`='{}'".format(temp[1], search_content)
                self.query.exec(sql)
                i = 0
                col_id = [1, 8, 10, 11, 13]
                while self.query.next():  # 每次查询一行记录
                    for j in range(1, len(col_id) + 1):
                        new_item = QTableWidgetItem(self.query.value(col_id[j - 1]))
                        new_item.setTextAlignment(Qt.AlignHCenter)
                        self.tableWidget_02_plot_info.setItem(i, j - 1, new_item)

                    i += 1
                break
        self.model_name_list = []
        model = self.tableWidget_02_plot_info.model()
        for j in range(0, i):
            model_name = model.data(model.index(j, 0))
            self.model_name_list.append(model_name)

    def save_change_info(self):
        """复制配置"""
        model_name = self.lineEdit_01_model_name.currentText()
        run_option = self.lineEdit_02_run_option.currentText()
        load_model_path = self.lineEdit_06_load_model_path.text()
        batchsize = self.lineEdit_07_batchsize.text()
        train_epoch = self.lineEdit_08_train_epoch.text()
        data_load_path = self.lineEdit_10_data_load_path.text()
        learning_rate = self.lineEdit_13_learning_rate.text()
        save_file_path = self.lineEdit_14_save_file_path.text()
        note = self.lineEdit_16_note.toPlainText()

        infos = [model_name, run_option, load_model_path, batchsize, train_epoch,
                 data_load_path, learning_rate, save_file_path, note]
        if "" in infos:
            QMessageBox.information(self, '提示', '修改的信息不能为空')

        sql = "update operating set " \
              "run_option='{1}'," \
              "load_model_path='{2}'," \
              "num_receiver='{3}'," \
              "num_frequency='{4}'," \
              "batchsize='{5}'," \
              "train_epoch='{6}'," \
              "data_load_path='{7}'," \
              "learning_rate='{8}'," \
              "save_file_path='{9}'," \
              "note='{10}' " \
              "where " \
              "model_name='{0}'".format(*infos)
        self.query.exec(sql)

        # 清空文本框的内容
        self.person_no.setText('')
        self.lineEdit_01_model_name.setText('')
        self.lineEdit_02_run_option.setText('')
        self.lineEdit_06_load_model_path.setText('')
        self.lineEdit_07_batchsize.setText('')
        self.lineEdit_08_train_epoch.setText('')
        self.lineEdit_10_data_load_path.setText('')
        self.lineEdit_13_learning_rate.setText('')
        self.lineEdit_14_save_file_path.setText('')
        self.lineEdit_16_note.setPlainText('')

        # 重新加载所有信息
        self.load_all_infos()

        QMessageBox.information(self, '提示', '修改成功')

    def generate_menu(self, pos):
        """右键菜单功能"""
        menu = QMenu()
        time.sleep(0.2)
        item1 = menu.addAction("复制配置")
        item2 = menu.addAction("删除")
        action = menu.exec_(self.tableWidget_01_run_info.mapToGlobal(pos))
        if action == item1:
            print("选择了'复制配置'操作")

            # 从表格中提取需要的数据
            table_selected_index = self.tableWidget_01_run_info.currentIndex().row()
            model = self.tableWidget_01_run_info.model()
            model_name = model.data(model.index(table_selected_index, 0))
            run_option = model.data(model.index(table_selected_index, 1))
            load_model_path = model.data(model.index(table_selected_index, 2))
            batchsize = model.data(model.index(table_selected_index, 5))
            train_epoch = model.data(model.index(table_selected_index, 6))
            data_load_path = model.data(model.index(table_selected_index, 7))
            learning_rate = model.data(model.index(table_selected_index, 8))
            save_file_path = model.data(model.index(table_selected_index, 9))
            run_time = model.data(model.index(table_selected_index, 11))
            note = model.data(model.index(table_selected_index, 12))

            self.lineEdit_01_model_name.setCurrentText(model_name)
            self.lineEdit_02_run_option.setCurrentText(run_option)
            self.lineEdit_06_load_model_path.setText(load_model_path)
            self.lineEdit_07_batchsize.setText(batchsize)
            self.lineEdit_08_train_epoch.setText(train_epoch)
            self.lineEdit_10_data_load_path.setText(data_load_path)
            self.lineEdit_13_learning_rate.setText(learning_rate)
            self.lineEdit_14_save_file_path.setText(save_file_path)
            self.lineEdit_16_note.setPlainText(note)
        elif action == item2:
            print("选择了'删除'操作")
            reply2 = QMessageBox.warning(self, "警告", "您选择了删除，请谨慎操作，防止记录丢失！", QMessageBox.Yes | QMessageBox.No,
                                         QMessageBox.No)
            if not reply2 == 65536:
                table_selected_index = self.tableWidget_01_run_info.currentIndex().row()
                # 获取表格数据模型对象
                model = self.tableWidget_01_run_info.model()
                # 获取选中行的ID
                sql = "delete from person where id={}".format(table_selected_index+1)
                self.query.exec(sql)

                # 更新ID
                update_query = QSqlQuery()
                update_query.prepare("UPDATE person SET id = (SELECT COUNT(*) FROM person AS p WHERE p.id <= person.id)")
                if not update_query.exec():
                    print("更新ID失败:", update_query.lastError().text())
                # 重新加载所有信息
                self.load_all_infos()

                QMessageBox.information(self, '提示', '删除成功')

    def add_new_run_info(self):
        """运行并添加新的运行信息"""
        # oper_no = self.person_no.text()
        model_name = self.lineEdit_01_model_name.currentText()
        run_option = self.lineEdit_02_run_option.currentText()
        load_model_path = self.lineEdit_06_load_model_path.text()
        batchsize = self.lineEdit_07_batchsize.text()
        train_epoch = self.lineEdit_08_train_epoch.text()
        data_load_path = self.lineEdit_10_data_load_path.text()
        learning_rate = self.lineEdit_13_learning_rate.text()
        save_file_path = self.lineEdit_14_save_file_path.text()
        note = self.lineEdit_16_note.toPlainText()

        load_model_path = load_model_path.replace("\\", '/')
        save_file_path = save_file_path.replace("\\", '/')
        data_load_path = data_load_path.replace("\\", '/')
        if save_file_path == '-':
            save_file_path = data_load_path

        config = json.loads(open(os.path.join(data_load_path, 'config.json'), 'r', encoding='utf-8').read())
        num_receiver = len(config['ReceiverDepth'])
        num_frequency = config['Frequency']['NumValues']
        r_max = config['trainset']['SourceRange']['UpperLimit']
        z_max = config['trainset']['SourceDepth']['UpperLimit']

        if model_name == '~all~':
            for model_name in ['mtl_cnn', 'mtl_unet', 'mtl_unet_cbam', 'xception']:
                run_date_time = datetime.datetime.now()
                save_file_name = net_main(data_loader_path=data_load_path, model_name=model_name,
                                      batch_size=int(batchsize), epoch = int(train_epoch), 
                                      run_option=run_option, load_model_path=load_model_path, 
                                      lr=float(learning_rate), save_file=save_file_path)
                mat_path = os.path.abspath(save_file_path + '/b.training_results/' + save_file_name +
                                            '_training_process' + '.mat')
                data = loadmat(mat_path)
                y = np.squeeze(data['MAE_r'])
                if y.ndim > 0:
                    # 如果 y 是一个数组，取最后一个元素
                    train_MAE_range = '%.3f km' % y[-1]
                else:
                    # 如果 y 是一个标量，直接使用这个值
                    train_MAE_range = '%.3f km' % y
                y = np.squeeze(data['MAE_d'])
                if y.ndim > 0:
                    # 如果 y 是一个数组，取最后一个元素
                    train_MAE_depth = '%.3f  m' % y[-1]
                else:
                    # 如果 y 是一个标量，直接使用这个值
                    train_MAE_depth = '%.3f  m' % y
                mat_path = os.path.abspath(save_file_path + '/c.test_results/' + save_file_name +
                                            '_test_accuracy' + '.mat')
                data = loadmat(mat_path)
                y = np.squeeze(data['PRPA']) * 100
                test_MA_range = '%.1f %%' % (np.mean(y))
                y = np.squeeze(data['PDPA']) * 100
                test_MA_depth = '%.1f %%' % (np.mean(y))
                y = np.squeeze(data['Err_r']).flatten()
                y = y[~np.isnan(y)]
                test_MAE_range = '%.3f km' % (np.mean(y))
                y = np.squeeze(data['Err_d']).flatten()
                y = y[~np.isnan(y)]
                test_MAE_depth = '%.3f  m' % (np.mean(y))
                infos = [model_name, run_option, load_model_path, num_receiver, 
                         num_frequency, batchsize, train_epoch, data_load_path, learning_rate,
                         save_file_path, save_file_name, run_date_time, note, 
                        train_MAE_range, train_MAE_depth, test_MA_range, test_MA_depth, test_MAE_range, test_MAE_depth]
                self.info_wr(infos)
        else:
            # 判断有没有填写的信息，如果有则提示，如果没有则存储到文件
            run_date_time = datetime.datetime.now()
            save_file_name = net_main(data_loader_path=data_load_path, model_name=model_name,
                                      batch_size=int(batchsize), epoch = int(train_epoch), 
                                      run_option=run_option, load_model_path=load_model_path, 
                                      lr=float(learning_rate), save_file=save_file_path)
            # save_file_name = model_name + '_2.2datasetTest9_epoch_100.00'
            mat_path = os.path.abspath(save_file_path + '/b.training_results/' + save_file_name +
                                            '_training_process' + '.mat')
            data = loadmat(mat_path)
            y = np.squeeze(data['MAE_r'])
            if y.ndim > 0:
                # 如果 y 是一个数组，取最后一个元素
                train_MAE_range = '%.3f km' % y[-1]
            else:
                # 如果 y 是一个标量，直接使用这个值
                train_MAE_range = '%.3f km' % y
            y = np.squeeze(data['MAE_d'])
            if y.ndim > 0:
                # 如果 y 是一个数组，取最后一个元素
                train_MAE_depth = '%.3f  m' % y[-1]
            else:
                # 如果 y 是一个标量，直接使用这个值
                    train_MAE_depth = '%.3f  m' % y
            mat_path = os.path.abspath(save_file_path + '/c.test_results/' + save_file_name +
                                        '_test_accuracy' + '.mat')
            data = loadmat(mat_path)
            y = np.squeeze(data['PRPA']) * 100
            test_MA_range = '%.1f %%' % (np.mean(y))
            y = np.squeeze(data['PDPA']) * 100
            test_MA_depth = '%.1f %%' % (np.mean(y))
            y = np.squeeze(data['Err_r']).flatten()
            y = y[~np.isnan(y)]
            test_MAE_range = '%.3f km' % (np.mean(y))
            y = np.squeeze(data['Err_d']).flatten()
            y = y[~np.isnan(y)]
            test_MAE_depth = '%.3f  m' % (np.mean(y))
            infos = [model_name, run_option, load_model_path, num_receiver, 
                    num_frequency, batchsize, train_epoch, data_load_path, learning_rate,
                    save_file_path, save_file_name, run_date_time, note, 
                    train_MAE_range, train_MAE_depth, test_MA_range, test_MA_depth, test_MAE_range, test_MAE_depth]
            self.info_wr(infos)
        # 重新加载所有信息
        self.load_all_infos()

        QMessageBox.information(self, '提示', '提交成功')

    def info_wr(self, infos):
        """将信息写入数据库"""
        if "" in infos:
            QMessageBox.information(self, '提示', '输入的信息不能为空')
        else:
            sql = "insert into person(" \
                    "model_name," \
                    "run_option," \
                    "load_model_path," \
                    "num_receiver," \
                    "num_frequency," \
                    "batchsize," \
                    "train_epoch," \
                    "data_load_path," \
                    "learning_rate," \
                    "save_file_path," \
                    "save_file_name," \
                    "run_date_time," \
                    "note," \
                    "train_MAE_range," \
                    "train_MAE_depth," \
                    "test_MA_range," \
                    "test_MA_depth," \
                    "test_MAE_range," \
                    "test_MAE_depth) " \
                     "values(" \
                    "'{0}'," \
                    "'{1}'," \
                    "'{2}'," \
                    "'{3}'," \
                    "'{4}'," \
                    "'{5}'," \
                    "'{6}'," \
                    "'{7}'," \
                    "'{8}'," \
                    "'{9}'," \
                    "'{10}'," \
                    "'{11}'," \
                    "'{12}'," \
                    "'{13}'," \
                    "'{14}'," \
                    "'{15}'," \
                    "'{16}'," \
                    "'{17}'," \
                    "'{18}');".format(*infos)
            self.query.exec(sql)

    def load_all_infos(self):
        """加载所有的人员信息"""
        # 查询信息条数
        sql = "select count(*) from person"
        # 执行查询语句
        self.query.exec(sql)
        # 移动到查询结果的下一行
        self.query.next()
        # 设置表格的行数为查询结果的第一条记录的值（即信息条数）
        self.tableWidget_01_run_info.setRowCount(self.query.value(0))
        # 查询所有信息（如果数据量过大，切记要limit）
        sql = "select * from person"
        # 执行查询语句
        self.query.exec(sql)
        # 初始化行计数器
        i = 0
        # 取tableWidget_01_run_info 的列数
        col_num = self.tableWidget_01_run_info.columnCount()
        # 遍历查询结果的每一行
        while self.query.next():  # 每次查询一行记录
            # 遍历每一行的每一列（从第1列到第19列）
            for j in range(1, col_num+1):
                # 创建一个新的表格项，内容为查询结果的第j列的值
                new_item = QTableWidgetItem(self.query.value(j))
                # 设置表格项的文本对齐方式为水平居中
                new_item.setTextAlignment(Qt.AlignHCenter)
                # 将表格项设置到表格的第i行第j-1列
                self.tableWidget_01_run_info.setItem(i, j - 1, new_item)
            # 行计数器加1
            i += 1

    def save_table_csv(self):
        """保存表格为csv文件"""
        # 获取表格的行数和列数
        row_count = self.tableWidget_01_run_info.rowCount()
        col_count = self.tableWidget_01_run_info.columnCount()

        # 打开文件对话框，让用户选择保存文件的路径和文件名
        file_path, _ = QFileDialog.getSaveFileName(self, "保存文件", os.getcwd(), "CSV Files (*.csv);;All Files (*)")
        if file_path:# 如果用户选择了保存文件，则执行保存操作
            try:
                with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    # 写入表头
                    header = []
                    for col in range(col_count):
                        item = self.tableWidget_01_run_info.horizontalHeaderItem(col)
                        if item is not None:
                            header.append(item.text())
                        else:
                            header.append(f'Column{col}')
                    writer.writerow(header)

                    # 写入表格数据
                    for row in range(row_count):
                        row_data = []
                        for col in range(col_count):
                            item = self.tableWidget_01_run_info.item(row, col)
                            if item is not None:
                                row_data.append(item.text())
                            else:
                                row_data.append('')
                        writer.writerow(row_data)
            except Exception as e:
                QMessageBox.critical(self, "保存失败", f"保存文件时出现错误: {str(e)}")
            else:
                QMessageBox.information(self, "保存成功", "文件已成功保存。")

    def plot_from_files(self):
        """选项卡2的绘图功能"""
        self.clear_draw()
        # 创建动画图形并且显示在窗口右侧
        pl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        pl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        pl.rcParams['font.size'] = '16'  # 设置字体大小
        # 图 1 1
        self.fig_train_range = pl.figure()  # type:figure.Figure
        self.ax_train_range = pl.subplot()  # type:axes.Axes
        self.canvans_train_range = FigureCanvas(self.fig_train_range)
        self.toolbar_train_range = NavigationToolbar(self.canvans_train_range, self)
        self.train_range_layout.addWidget(self.canvans_train_range)
        self.train_range_layout.addWidget(self.toolbar_train_range)
        # self.ax_train_range.plot(xy[0], xy[1])

        # 图 1 2
        self.fig_train_depth = pl.figure()  # type:figure.Figure
        self.ax_train_depth = pl.subplot()  # type:axes.Axes
        self.canvans_train_depth = FigureCanvas(self.fig_train_depth)
        self.toolbar_train_depth = NavigationToolbar(self.canvans_train_depth, self)
        self.train_depth_layout.addWidget(self.canvans_train_depth)
        self.train_depth_layout.addWidget(self.toolbar_train_depth)
        # self.ax_train_depth.plot(xy[0], xy[1])

        # 图 2 1
        self.fig_test_range = pl.figure()  # type:figure.Figure
        self.ax_test_range = pl.subplot()  # type:axes.Axes
        self.canvans_test_range = FigureCanvas(self.fig_test_range)
        self.toolbar_test_range = NavigationToolbar(self.canvans_test_range, self)
        self.test_range_layout.addWidget(self.canvans_test_range)
        self.test_range_layout.addWidget(self.toolbar_test_range)
        # self.ax_test_range.plot(xy[0], xy[1])

        # 图 2 2
        self.fig_test_depth = pl.figure()  # type:figure.Figure
        self.ax_test_depth = pl.subplot()  # type:axes.Axes
        self.canvans_test_depth = FigureCanvas(self.fig_test_depth)
        self.toolbar_test_depth = NavigationToolbar(self.canvans_test_depth, self)
        self.test_depth_layout.addWidget(self.canvans_test_depth)
        self.test_depth_layout.addWidget(self.toolbar_test_depth)
        # self.ax_test_depth.plot(xy[0], xy[1])

        # 图 3 1
        self.fig_exp_range = pl.figure()  # type:figure.Figure
        self.ax_exp_range = pl.subplot()  # type:axes.Axes
        self.canvans_exp_range = FigureCanvas(self.fig_exp_range)
        self.toolbar_exp_range = NavigationToolbar(self.canvans_exp_range, self)
        self.exp_range_layout.addWidget(self.canvans_exp_range)
        self.exp_range_layout.addWidget(self.toolbar_exp_range)
        # self.ax_exp_range.plot(xy[0], xy[1])

        # 图 3 2
        self.fig_exp_depth = pl.figure()  # type:figure.Figure
        self.ax_exp_depth = pl.subplot()  # type:axes.Axes
        self.canvans_exp_depth = FigureCanvas(self.fig_exp_depth)
        self.toolbar_exp_depth = NavigationToolbar(self.canvans_exp_depth, self)
        self.exp_depth_layout.addWidget(self.canvans_exp_depth)
        self.exp_depth_layout.addWidget(self.toolbar_exp_depth)
        # self.ax_exp_depth.plot(xy[0], xy[1])

        # model_name_list = ['mtl_cnn', 'mtl_unet', 'mtl_unet_cbam', 'xception', ]
        model = self.tableWidget_02_plot_info.model()
        # 获取 model 的行数
        row_count = model.rowCount()

        # training loss
        dpi = 200
        color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        line_style = ['-', '--', '-.', ':']
        self.tableWidget_03_results.setRowCount(len(self.model_name_list))
        model_name = []
        for i in range(0, row_count):
            save_file = model.data(model.index(i, 2))
            save_file_name = model.data(model.index(i, 3))
            mat_path = os.path.abspath(save_file + '/b.training_results/' + save_file_name +
                                        '_training_process' + '.mat')
            plot_data = loadmat(mat_path)
            x = np.squeeze(plot_data['epoch'])
            y = np.squeeze(plot_data['MAE_r'])
            model_name.append((str(plot_data['model_name']))[2:-2])
            self.ax_train_range.plot(x, y, color[i] + line_style[i],
                                        label=model_name[i])
            new_item = QTableWidgetItem(model_name[i])
            new_item.setTextAlignment(Qt.AlignHCenter)
            self.tableWidget_03_results.setItem(i, 0, new_item)
            # 检查 y 是否是一个数组
            if y.ndim > 0:
                # 如果 y 是一个数组，取最后一个元素
                train_loss_r_str = '%.3f km' % y[-1]
            else:
                # 如果 y 是一个标量，直接使用这个值
                train_loss_r_str = '%.3f km' % y
            new_item = QTableWidgetItem(train_loss_r_str)
            new_item.setTextAlignment(Qt.AlignHCenter)
            self.tableWidget_03_results.setItem(i, 1, new_item)
        self.tableWidget_03_results.update()

        self.ax_train_range.legend()
        self.ax_train_range.set_xlabel("epoch")
        self.ax_train_range.set_ylabel("MAE_r (km)")

        for i in range(0, row_count):
            save_file = model.data(model.index(i, 2))
            save_file_name = model.data(model.index(i, 3))
            mat_path = os.path.abspath(save_file + '/b.training_results/' + save_file_name +
                                        '_training_process' + '.mat')
            plot_data = loadmat(mat_path)
            x = np.squeeze(plot_data['epoch'])
            y = np.squeeze(plot_data['MAE_d'])
            self.ax_train_depth.plot(x, y, color[i] + line_style[i],
                                        label=model_name[i])
            # 检查 y 是否是一个数组
            if y.ndim > 0:
                # 如果 y 是一个数组，取最后一个元素
                train_loss_d_str = '%.3f m' % y[-1]
            else:
                # 如果 y 是一个标量，直接使用这个值
                train_loss_d_str = '%.3f m' % y
            new_item = QTableWidgetItem(train_loss_d_str)
            new_item.setTextAlignment(Qt.AlignHCenter)
            self.tableWidget_03_results.setItem(i, 2, new_item)

        self.ax_train_depth.legend()
        self.ax_train_depth.set_xlabel("epoch")
        self.ax_train_depth.set_ylabel("MAE_d (m)")

        # test accuracy
        for i in range(0, row_count):
            save_file = model.data(model.index(i, 2))
            save_file_name = model.data(model.index(i, 3))
            mat_path = os.path.abspath(save_file + '/c.test_results/' + save_file_name +
                                        '_test_accuracy' + '.mat')
            plot_data = loadmat(mat_path)
            x = np.squeeze(plot_data['para'])
            y = np.squeeze(plot_data['PRPA']) * 100
            Es = np.squeeze(plot_data['Es'])
            # 将Es按照最后一个维度分为Es_r和Es_d，并将第一个维度平铺到第二个维度上去
            Es_r = Es[:, :, 0].flatten()
            Es_d = Es[:, :, 1].flatten()
            x1 = np.arange(0, len(Es_r))

            Err_r = np.squeeze(plot_data['Err_r']).flatten()
            Err_r = Err_r[~np.isnan(Err_r)]
            Err_r_mean = np.mean(Err_r)
            Err_d = np.squeeze(plot_data['Err_d']).flatten()
            Err_d = Err_d[~np.isnan(Err_d)]
            Err_d_mean = np.mean(Err_d)
            Err_r_mean_str = '%.3f km' % Err_r_mean
            Err_d_mean_str = '%.3f m' % Err_d_mean
            
            self.ax_exp_range.plot(x1, Es_r, color[i] + line_style[i],
                                    label=model_name[i])
            self.ax_exp_depth.plot(x1, Es_d, color[i] + line_style[i],
                                    label=model_name[i])
    
            self.ax_test_range.plot(x, y, color[i] + line_style[i],
                                    label=model_name[i])
            test_m_acc_r_str = '%.1f %%' % np.mean(y)
            new_item = QTableWidgetItem(test_m_acc_r_str)
            new_item.setTextAlignment(Qt.AlignHCenter)
            self.tableWidget_03_results.setItem(i, 3, new_item)
            new_item = QTableWidgetItem(Err_r_mean_str)
            new_item.setTextAlignment(Qt.AlignHCenter)
            self.tableWidget_03_results.setItem(i, 5, new_item)
            new_item = QTableWidgetItem(Err_d_mean_str)
            new_item.setTextAlignment(Qt.AlignHCenter)
            self.tableWidget_03_results.setItem(i, 6, new_item)

        self.ax_test_range.legend()
        self.ax_test_range.set_xlabel("Parameter")
        self.ax_test_range.set_ylabel("Proportion (%)")

        self.ax_exp_depth.legend()
        self.ax_exp_depth.set_xlabel("Sample index")
        self.ax_exp_depth.set_ylabel("Depth Estimation (m)")

        for i in range(0, row_count):
            save_file = model.data(model.index(i, 2))
            save_file_name = model.data(model.index(i, 3))
            mat_path = os.path.abspath(save_file + '/c.test_results/' + save_file_name +
                                        '_test_accuracy' + '.mat')
            plot_data = loadmat(mat_path)
            x = np.squeeze(plot_data['para'])
            y = np.squeeze(plot_data['PDPA']) * 100
            self.ax_test_depth.plot(x, y, color[i] + line_style[i],
                                    label=model_name[i])
            test_m_acc_d_str = '%.1f %%' % np.mean(y)
            new_item = QTableWidgetItem(test_m_acc_d_str)
            new_item.setTextAlignment(Qt.AlignHCenter)
            self.tableWidget_03_results.setItem(i, 4, new_item)

        self.ax_test_depth.legend()
        self.ax_test_depth.set_xlabel("Parameter")
        self.ax_test_depth.set_ylabel("Proportion (%)")


    def clear_draw(self):
        """清除绘图"""
        try:
            # 清除右侧画布内容
            for i in range(self.train_range_layout.count()):
                self.train_range_layout.itemAt(i).widget().deleteLater()
            for i in range(self.train_depth_layout.count()):
                self.train_depth_layout.itemAt(i).widget().deleteLater()
            for i in range(self.test_range_layout.count()):
                self.test_range_layout.itemAt(i).widget().deleteLater()
            for i in range(self.test_depth_layout.count()):
                self.test_depth_layout.itemAt(i).widget().deleteLater()
            for i in range(self.exp_range_layout.count()):
                self.exp_range_layout.itemAt(i).widget().deleteLater()
            for i in range(self.exp_depth_layout.count()):
                self.exp_depth_layout.itemAt(i).widget().deleteLater()
        except:
            pass


if __name__ == '__main__':
    # 创建应用程序对象
    # QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)

    # 设置窗口窗口风格
    QApplication.setStyle(QStyleFactory.create("Fusion"))
    print(QStyleFactory.keys())

    # 创建一个要显示的窗口对象
    app_widget = AppWidget()
    app_widget.show()

    # 让应用程序一直运行，在这期间会显示UI界面、检测事件等操作
    sys.exit(app.exec())
